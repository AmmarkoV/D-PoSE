"""
MHR Head Module for D-Pose MHR port.

This module replaces SMPLXCamHead and provides:
1. MHR forward pass to compute vertices from MHR parameters
2. Camera projection for 3D to 2D joint projection
3. Skeleton state extraction for joint positions
"""

import os
import sys
import types
import pickle
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse


def _stub_chumpy():
    if 'chumpy' not in sys.modules:
        _c = types.ModuleType('chumpy')
        _ch = types.ModuleType('chumpy.ch')
        class _Ch: pass
        _c.Ch = _Ch
        _c.array = lambda *a, **k: None
        _ch.Ch = _Ch
        sys.modules['chumpy'] = _c
        sys.modules['chumpy.ch'] = _ch


class MHRHead(nn.Module):
    """Head that applies MHR model and projects joints to 2D.

    This replaces SMPLXCamHead and works directly with MHR parameters:
    - identity_coeffs: [B, 45] body shape blendshapes
    - lbs_model_params: [B, ~144] LBS model parameters
    - face_expr_coeffs: [B, 72] facial expressions

    Args:
        mhr_model: Pre-loaded MHR model instance
        img_res: Image resolution for normalization
    """

    def __init__(self, mhr_model, img_res=224):
        super(MHRHead, self).__init__()
        self.mhr_model = mhr_model
        self.img_res = img_res

        # MHR uses centimeters, convert to meters for consistency
        self.cm_to_m = 0.01

        # Register MHR faces for rendering
        # These come from the MHR character mesh
        if hasattr(mhr_model, 'character') and hasattr(mhr_model.character, 'mesh'):
            faces = mhr_model.character.mesh.faces
            if isinstance(faces, torch.Tensor):
                self.register_buffer('faces', faces)
            else:
                self.register_buffer('faces', torch.tensor(faces, dtype=torch.long))
        else:
            # Fallback - will be set later
            self.faces = None

        # Load mhr2smpl mapping + SMPL J_regressor for differentiable joint computation
        _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)

        _mapping = np.load(os.path.join(_proj_root, 'assets', 'mhr2smpl_mapping.npz'))
        _triangle_ids = _mapping['triangle_ids'].astype(np.int64)   # [6890]
        _baryc = _mapping['baryc_coords'].astype(np.float32)        # [6890, 3]

        from load_mhr_portable import load_portable
        _portable = load_portable(
            os.path.join(_proj_root, 'mhr_portable_dump', 'mhr_dump_lod1.pt')
        )
        _mhr_faces = _portable['interesting']['character.mesh.faces']
        if isinstance(_mhr_faces, torch.Tensor):
            _mhr_faces = _mhr_faces.cpu().numpy()
        else:
            _mhr_faces = np.array(_mhr_faces)
        _tri_vids = _mhr_faces[_triangle_ids].astype(np.int64)      # [6890, 3]

        # Stub chumpy only for this pickle load, then restore original state
        _saved = {k: sys.modules.get(k) for k in ('chumpy', 'chumpy.ch')}
        _stub_chumpy()
        try:
            _smpl_pkl = os.path.join(
                _proj_root,
                'data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_NEUTRAL.pkl'
            )
            with open(_smpl_pkl, 'rb') as _f:
                _smpl_data = pickle.load(_f, encoding='latin1')
        finally:
            for _k, _v in _saved.items():
                if _v is None:
                    sys.modules.pop(_k, None)
                else:
                    sys.modules[_k] = _v
        _J_reg = _smpl_data['J_regressor']
        if scipy.sparse.issparse(_J_reg):
            _J_reg = np.array(_J_reg.todense()).astype(np.float32)
        else:
            _J_reg = np.array(_J_reg).astype(np.float32)            # [24, 6890]

        self.register_buffer('_smpl_tri_vids', torch.tensor(_tri_vids, dtype=torch.long))
        self.register_buffer('_smpl_baryc', torch.tensor(_baryc, dtype=torch.float32))
        self.register_buffer('_smpl_J_reg', torch.tensor(_J_reg, dtype=torch.float32))

    def forward(self, identity_coeffs, lbs_model_params, face_expr_coeffs,
                cam, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h,
                normalize_joints2d=False):
        """
        Forward pass through MHR model and camera projection.

        Args:
            identity_coeffs: [B, 45] identity blendshape coefficients
            lbs_model_params: [B, ~144] LBS model parameters
            face_expr_coeffs: [B, 72] face expression coefficients
            cam: [B, 3] camera parameters (scale, tx, ty)
            cam_intrinsics: [B, 3, 3] camera intrinsic matrix
            bbox_scale: [B] bounding box scale
            bbox_center: [B, 2] bounding box center
            img_w: [B] image width
            img_h: [B] image height
            normalize_joints2d: Whether to normalize 2D joints to [-1, 1]

        Returns:
            Dictionary with:
            - vertices: [B, V, 3] mesh vertices in meters
            - joints3d: [B, J, 3] 3D joint positions in meters
            - joints2d: [B, J, 2] projected 2D joint positions
            - pred_cam_t: [B, 3] camera translation
            - skel_state: [B, J, 16] skeleton state (4x4 transforms)
        """
        batch_size = identity_coeffs.shape[0]
        device = identity_coeffs.device

        # Ensure face_expr_coeffs exists
        if face_expr_coeffs is None:
            face_expr_coeffs = torch.zeros(
                batch_size, 72, device=device, dtype=identity_coeffs.dtype
            )

        # Forward through MHR model
        # MHR outputs vertices in centimeters
        verts_cm, skel_state = self.mhr_model(
            identity_coeffs=identity_coeffs,
            model_parameters=lbs_model_params,
            face_expr_coeffs=face_expr_coeffs,
            apply_correctives=True
        )

        # Convert from centimeters to meters
        verts_m = verts_cm * self.cm_to_m

        # Compute SMPL joints from MHR vertices via barycentric interpolation (differentiable)
        v0 = verts_m[:, self._smpl_tri_vids[:, 0]]  # [B, 6890, 3]
        v1 = verts_m[:, self._smpl_tri_vids[:, 1]]
        v2 = verts_m[:, self._smpl_tri_vids[:, 2]]
        w = self._smpl_baryc  # [6890, 3]
        smpl_verts = (w[:, 0].unsqueeze(0).unsqueeze(-1) * v0
                      + w[:, 1].unsqueeze(0).unsqueeze(-1) * v1
                      + w[:, 2].unsqueeze(0).unsqueeze(-1) * v2)  # [B, 6890, 3]
        smpl_joints3d = torch.einsum('jv,bvd->bjd', self._smpl_J_reg, smpl_verts)  # [B, 24, 3]

        # Extract joint positions from skeleton state.
        # skel_state is [B, 127, 8]: each joint stores [tx, ty, tz, qw, qx, qy, qz, 1]
        # where the first 3 values are the global translation in centimetres.
        joints3d_cm = skel_state[:, :, :3]  # [B, 127, 3]
        joints3d_m = joints3d_cm * self.cm_to_m

        # Compute camera translation from PARE-style camera params
        cam_t = convert_pare_to_full_img_cam(
            pare_cam=cam,
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
            crop_res=self.img_res,
        )

        # Project 3D joints to 2D
        joints2d = perspective_projection(
            points=joints3d_m,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )

        if normalize_joints2d:
            joints2d = joints2d / (self.img_res / 2.)

        return {
            'vertices': verts_m,
            'joints3d': joints3d_m,
            'joints3d_smpl': smpl_joints3d,
            'joints2d': joints2d,
            'pred_cam_t': cam_t,
            'skel_state': skel_state,
        }


def perspective_projection(points, rotation, translation, cam_intrinsics):
    """
    Project 3D points to 2D using perspective projection.

    Args:
        points: [B, N, 3] 3D points
        rotation: [B, 3, 3] rotation matrix
        translation: [B, 3] camera translation
        cam_intrinsics: [B, 3, 3] camera intrinsic matrix

    Returns:
        [B, N, 2] projected 2D points
    """
    K = cam_intrinsics

    # Apply rotation
    points = torch.einsum('bij,bkj->bki', rotation, points)

    # Apply translation
    points = points + translation.unsqueeze(1)

    # Perspective division
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points.float())

    return projected_points[:, :, :-1]


def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):
    """
    Convert PARE-style camera parameters to full image camera translation.

    Args:
        pare_cam: [B, 3] camera parameters (scale, tx, ty)
        bbox_height: [B] bounding box height in pixels
        bbox_center: [B, 2] bounding box center
        img_w: [B] image width
        img_h: [B] image height
        focal_length: [B] focal length
        crop_res: Crop resolution (default 224)

    Returns:
        [B, 3] camera translation
    """
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t
