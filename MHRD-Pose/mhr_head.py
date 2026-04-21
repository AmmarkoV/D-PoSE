"""
MHR Head Module for D-Pose MHR port.

MAPPED FROM: train/models/head/smplx_cam_head.py:SMPLXCamHead

This module replaces SMPLXCamHead and provides:
1. MHR forward pass to compute vertices from MHR parameters
2. Camera projection for 3D to 2D joint projection
3. Skeleton state extraction for joint positions

File correspondence:
  mhr_head.py:MHRHead          ←  train/models/head/smplx_cam_head.py:SMPLXCamHead
  mhr_head.py:perspective_projection  ←  train/models/head/smplx_cam_head.py:perspective_projection  (identical)
  mhr_head.py:convert_pare_to_full_img_cam  ←  train/models/head/smplx_cam_head.py:convert_pare_to_full_img_cam  (identical)

Key differences from original SMPLXCamHead:
  - Input params: original takes (rotmat, shape, cam) → SMPLX forward
    MHR takes (identity_coeffs, lbs_model_params, face_expr_coeffs, cam) → MHR forward
  - Original: SMPLX model instance with num_betas=11, runs SMPLX forward with pose2rot=False
    MHR: mhr_model instance (pre-loaded TorchScript), runs MHR forward with identity/lbs/expr
  - Original output: {vertices, joints3d, joints2d, pred_cam_t, left_hand_3d, right_hand_3d, head_3d, ...}
    MHR output: {vertices, joints3d (127 MHR skel joints), joints3d_smpl (24), joints2d (127), joints2d_smpl (24), pred_cam_t, skel_state}
  - Original: joints3d = smpl_output.joints[:22] (body joints from SMPLX)
    MHR: joints3d = skel_state[:, :, :3] (first 3 values of MHR skeleton transforms)
  - New: MHR head computes SMPL joints via barycentric interpolation (mhr2smpl_mapping.npz + SMPL J_regressor)
    This provides differentiable SMPL-ordered joints for loss computation
  - New: Two 2D projection outputs — joints2d (MHR 127-joint ordering, viz only) and joints2d_smpl (SMPL 24-joint ordering, used in loss)
  - Original SMPLXCamHeadBodyOnly (L8-60): simplified version without hands/head split — MHRHead is closer to this
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

        class _Ch:
            pass

        _c.Ch = _Ch
        _c.array = lambda *a, **k: None
        _ch.Ch = _Ch
        sys.modules['chumpy'] = _c
        sys.modules['chumpy.ch'] = _ch


class MHRHead(nn.Module):
    # MAPPED FROM: train/models/head/smplx_cam_head.py:SMPLXCamHead (L66)
    """Head that applies MHR model and projects joints to 2D.

    This replaces SMPLXCamHead and works directly with MHR parameters:
    - identity_coeffs: [B, 45] body shape blendshapes
    - lbs_model_params: [B, ~204] LBS model parameters
    - face_expr_coeffs: [B, 72] facial expressions

    Args:
        mhr_model: Pre-loaded MHR model instance
        img_res: Image resolution for normalization
    """

    def __init__(self, mhr_model, img_res=224):
        # MAPPED FROM: SMPLXCamHead.__init__ (smplx_cam_head.py:L67)
        #
        # Original:
        #   self.smplx = SMPLX(config.SMPLX_MODEL_DIR, num_betas=11)
        #   self.add_module('smplx', self.smplx)
        #   self.img_res = img_res
        #
        # New:
        #   self.mhr_model = mhr_model (pre-loaded from checkpoint, not loaded from disk)
        #   self.img_res = img_res
        #
        # New in MHRHead:
        #   - Barycentric mapping (mhr2smpl_mapping.npz) for SMPL joint extraction
        #   - SMPL J_regressor from SMPL_NEUTRAL.pkl for differentiable joint computation
        #   - These enable computing SMPL-ordered 24-joint predictions from MHR vertices
        super(MHRHead, self).__init__()
        self.mhr_model = mhr_model
        self.img_res = img_res

        # MHR uses centimeters, convert to meters for consistency
        self.cm_to_m = 0.01

        # Register MHR faces for rendering
        # These come from the MHR character mesh
        # ← original: self.smplx.faces (SMPLX faces from model instance)
        # ← new: mhr_model.character.mesh.faces (MHR faces from loaded model)
        if hasattr(mhr_model, 'character') and hasattr(mhr_model.character,
                                                       'mesh'):
            faces = mhr_model.character.mesh.faces
            if isinstance(faces, torch.Tensor):
                self.register_buffer('faces', faces)
            else:
                self.register_buffer('faces',
                                     torch.tensor(faces, dtype=torch.long))
        else:
            # Fallback - will be set later
            self.faces = None

        # Load mhr2smpl mapping + SMPL J_regressor for differentiable joint computation
        # ← original: no such mapping — SMPLX model directly outputs SMPL-ordered joints
        # ← new: MHR vertices are in a different mesh space, need barycentric interpolation
        #   to project onto SMPL vertex topology, then J_regressor for joint extraction
        _proj_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)

        _mapping = np.load(
            os.path.join(_proj_root, 'assets', 'mhr2smpl_mapping.npz'))
        _triangle_ids = _mapping['triangle_ids'].astype(np.int64)  # [6890]
        _baryc = _mapping['baryc_coords'].astype(np.float32)  # [6890, 3]

        from load_mhr_portable import load_portable
        _portable = load_portable(
            os.path.join(_proj_root, 'mhr_portable_dump', 'mhr_dump_lod1.pt'))
        _mhr_faces = _portable['interesting']['character.mesh.faces']
        if isinstance(_mhr_faces, torch.Tensor):
            _mhr_faces = _mhr_faces.cpu().numpy()
        else:
            _mhr_faces = np.array(_mhr_faces)
        _tri_vids = _mhr_faces[_triangle_ids].astype(np.int64)  # [6890, 3]

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
            _J_reg = np.array(_J_reg).astype(np.float32)  # [24, 6890]

        self.register_buffer('_smpl_tri_vids',
                             torch.tensor(_tri_vids, dtype=torch.long))
        self.register_buffer('_smpl_baryc',
                             torch.tensor(_baryc, dtype=torch.float32))
        self.register_buffer('_smpl_J_reg',
                             torch.tensor(_J_reg, dtype=torch.float32))

    def forward(self,
                identity_coeffs,
                lbs_model_params,
                face_expr_coeffs,
                cam,
                cam_intrinsics,
                bbox_scale,
                bbox_center,
                img_w,
                img_h,
                normalize_joints2d=False):
        # MAPPED FROM: SMPLXCamHead.forward (smplx_cam_head.py:L73)
        #
        # Original flow:
        #   1. self.smplx(betas=shape, body_pose=rotmat[:,1:22], global_orient=rotmat[:,0])
        #      → smpl_output.vertices [B, V, 3], smpl_output.joints [B, ~118, 3]
        #   2. Split joints: body[:22], left_hand[22:37], right_hand[37:52], head[52:]
        #   3. Convert camera params → cam_t via convert_pare_to_full_img_cam
        #   4. perspective_projection each group separately
        #   5. Return {vertices, joints3d (body only), joints2d (body only), pred_cam_t,
        #              left_hand_3d, right_hand_3d, head_3d, left_hand_2d, right_hand_2d, head_2d, ...}
        #
        # New flow:
        #   1. self.mhr_model(identity_coeffs, model_parameters=lbs_model_params, face_expr_coeffs)
        #      → verts_cm [B, V, 3], skel_state [B, 127, 8]
        #   2. Barycentric interpolation: MHR vertices → SMPL vertices → SMPL joints [B, 24, 3]
        #   3. skel_state[:, :, :3] → MHR joints [B, 127, 3]
        #   4. Convert camera params → cam_t via convert_pare_to_full_img_cam (identical)
        #   5. perspective_projection both MHR joints (127) and SMPL joints (24)
        #   6. Return {vertices, joints3d (127 MHR), joints3d_smpl (24 SMPL), joints2d (127),
        #              joints2d_smpl (24), pred_cam_t, skel_state}
        """
        Forward pass through MHR model and camera projection.

        Args:
            identity_coeffs: [B, 45] identity blendshape coefficients
            lbs_model_params: [B, ~204] LBS model parameters
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
            - joints3d: [B, 127, 3] MHR skeleton 3D joints (cm→m)
            - joints3d_smpl: [B, 24, 3] SMPL-ordered 3D joints (via barycentric interp)
            - joints2d: [B, 127, 2] projected MHR 2D joints (viz only)
            - joints2d_smpl: [B, 24, 2] projected SMPL 2D joints (used in loss)
            - pred_cam_t: [B, 3] camera translation
            - skel_state: [B, 127, 8] skeleton state (4x4 transforms)
        """
        batch_size = identity_coeffs.shape[0]
        device = identity_coeffs.device

        # Ensure face_expr_coeffs exists
        if face_expr_coeffs is None:
            face_expr_coeffs = torch.zeros(batch_size,
                                           72,
                                           device=device,
                                           dtype=identity_coeffs.dtype)

        # Forward through MHR model
        # MHR outputs vertices in centimeters
        # ← original: self.smplx(betas=shape, body_pose=rotmat[:,1:22], global_orient=rotmat[:,0], pose2rot=False)
        #   SMPLX expects (betas, body_pose, global_orient) as separate args
        # ← new: self.mhr_model(identity_coeffs, model_parameters=lbs_model_params, face_expr_coeffs, apply_correctives=True)
        #   MHR expects (identity, lbs_model_params, face_expr_coeffs) as separate args
        verts_cm, skel_state = self.mhr_model(
            identity_coeffs=identity_coeffs,
            model_parameters=lbs_model_params,
            face_expr_coeffs=face_expr_coeffs,
            apply_correctives=True)

        # Convert from centimeters to meters
        verts_m = verts_cm * self.cm_to_m

        # Compute SMPL joints from MHR vertices via barycentric interpolation (differentiable)
        # ← original: smpl_output.joints from SMPLX forward — native SMPL joint output
        # ← new: MHR mesh has different topology, need to project to SMPL vertex space
        #   then apply SMPL J_regressor to get joints
        v0 = verts_m[:, self._smpl_tri_vids[:, 0]]  # [B, 6890, 3]
        v1 = verts_m[:, self._smpl_tri_vids[:, 1]]
        v2 = verts_m[:, self._smpl_tri_vids[:, 2]]
        w = self._smpl_baryc  # [6890, 3]
        smpl_verts = (w[:, 0].unsqueeze(0).unsqueeze(-1) * v0 +
                      w[:, 1].unsqueeze(0).unsqueeze(-1) * v1 +
                      w[:, 2].unsqueeze(0).unsqueeze(-1) * v2)  # [B, 6890, 3]
        smpl_joints3d = torch.einsum('jv,bvd->bjd', self._smpl_J_reg,
                                     smpl_verts)  # [B, 24, 3]

        # Extract joint positions from skeleton state.
        # skel_state is [B, 127, 8]: each joint stores [tx, ty, tz, qw, qx, qy, qz, 1]
        # where the first 3 values are the global translation in centimetres.
        joints3d_cm = skel_state[:, :, :3]  # [B, 127, 3]
        joints3d_m = joints3d_cm * self.cm_to_m

        # Compute camera translation from PARE-style camera params
        # ← identical to original SMPLXCamHead.forward (smplx_cam_head.py:L99-107)
        cam_t = convert_pare_to_full_img_cam(
            pare_cam=cam,
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
            crop_res=self.img_res,
        )

        # Project MHR skel_state joints (all 127) to 2D.
        # This is kept for backward-compat and visualization but its joint ordering
        # is MHR-specific and does NOT match GT keypoint annotations in BEDLAM/3DPW,
        # so it cannot be used directly for the supervised 2D keypoint loss.
        # ← original: perspective_projection(joints3d, ...) with SMPLX body joints (22)
        # ← new: perspective_projection(joints3d_m, ...) with MHR skeleton joints (127)
        _rot = torch.eye(3, device=device).unsqueeze(0).expand(
            batch_size, -1, -1)
        joints2d = perspective_projection(
            points=joints3d_m,
            rotation=_rot,
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )

        # Project the 24 SMPL-mapped joints to 2D.
        # smpl_joints3d was computed above via barycentric interpolation of MHR
        # vertices using mhr2smpl_mapping.npz + SMPL J_regressor, so it carries
        # gradients back through the MHR vertex computation.
        # Crucially, these 24 joints follow the standard SMPL joint ordering
        # (0=pelvis, 1=L-hip, 2=R-hip, …, 23=R-wrist), which matches the GT
        # keypoint annotations stored in BEDLAM/3DPW batches. This makes
        # joints2d_smpl the correct tensor to use in the 2D keypoint loss.
        # ← original: perspective_projection(joints_body, ...) with SMPLX body joints (22)
        # ← new: perspective_projection(smpl_joints3d, ...) with 24 SMPL-ordered joints
        joints2d_smpl = perspective_projection(
            points=smpl_joints3d,
            rotation=_rot,
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )

        if normalize_joints2d:
            # ← original: joints2d / (self.img_res / 2.)  [same]
            joints2d = joints2d / (self.img_res / 2.)
            joints2d_smpl = joints2d_smpl / (self.img_res / 2.)

        return {
            # ← original: output = {vertices, joints3d (body 22), joints2d (body 22), pred_cam_t,
            #                       left_hand_3d, right_hand_3d, head_3d, left_hand_2d, ...}
            # ← new: simplified output with MHR-specific keys + SMPL-mapped fallbacks
            'vertices': verts_m,
            'joints3d': joints3d_m,
            'joints3d_smpl': smpl_joints3d,
            # joints2d uses MHR skel_state ordering (127 joints) — for viz only.
            'joints2d': joints2d,
            # joints2d_smpl uses SMPL ordering (24 joints) — used in 2D keypoint loss.
            'joints2d_smpl': joints2d_smpl,
            'pred_cam_t': cam_t,
            'skel_state': skel_state,
        }


def perspective_projection(points, rotation, translation, cam_intrinsics):
    # MAPPED FROM: smplx_cam_head.py:perspective_projection (L172)
    # Identical to original — no changes.
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
    projected_points = torch.einsum('bij,bkj->bki', K,
                                    projected_points.float())

    return projected_points[:, :, :-1]


def convert_pare_to_full_img_cam(pare_cam,
                                 bbox_height,
                                 bbox_center,
                                 img_w,
                                 img_h,
                                 focal_length,
                                 crop_res=224):
    # MAPPED FROM: smplx_cam_head.py:convert_pare_to_full_img_cam (L183)
    # Identical to original — no changes.
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
