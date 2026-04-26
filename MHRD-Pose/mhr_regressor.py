"""
MHR Regressor for D-Pose MHR port.

MAPPED FROM: train/models/head/refit_regressor.py:Regressor (ReFit-based SMPL regressor)

This module replaces the SMPL-based Regressor with an MHR-based version
that predicts MHR parameters (identity, expression, pose) instead of
SMPL betas and rotation matrices.

The regressor follows the same iterative refinement approach as the
original ReFit-based regressor, but outputs MHR-compatible parameters.

MHR Parameter Structure:
- identity_coeffs: [B, 45] - body shape blendshapes
- lbs_model_params: [B, 204] - LBS model parameters (flat vector, not per-joint)
- face_expr_coeffs: [B, 72] - facial expressions

File correspondence:
  mhr_regressor.py:MultiLinear       ←  train/models/head/refit_regressor.py:MultiLinear  (identical)
  mhr_regressor.py:MHRRegressor       ←  train/models/head/refit_regressor.py:Regressor

Key differences from original Regressor:
  - Output: original returns {pred_pose (rotmat B×22×3×3), pred_shape (betas B×11), pred_cam (B×3)}
    MHR returns {pred_pose (lbs_params B×204), pred_identity (B×45), pred_expr (B×72), pred_cam (B×3)}
  - Pose branch: original uses MultiLinear (per-joint, 22 heads) with 6D pose → rotmat
    MHR uses flat MLP (no MultiLinear) with 204-D lbs_model_params
  - Shape branch: original decshape → 11-D betas, init from SMPL_MEAN_PARAMS
    MHR: decshape → 45-D identity, decexpr → 72-D expr, both init to zeros
  - init_pose: original loads from SMPL_MEAN_PARAMS['pose'] (6D per joint)
    MHR: zeros (204-D lbs_model_params neutral pose)
  - init_shape: original loads from SMPL_MEAN_PARAMS['shape'] (11 betas + 1 pad)
    MHR: zeros (45 identity + 72 expr)
  - pose_input: original (720+48+3)+6, shape_input (720+48+3)*22+11
    MHR: flat_dim+204, flat_dim+45, flat_dim+72
  - decpose: original MultiLinear(22, hidden_dim, 6) → per-joint 6D residuals
    MHR: nn.Linear(hidden_dim, 204) → flat lbs residuals
  - decshape: original nn.Linear(hidden_dim, 11)
    MHR: nn.Linear(hidden_dim, 45)
  - New: decexpr = nn.Linear(hidden_dim, 72) — no original equivalent
  - forward: original hpose.transpose(1,2) (permute C,J→J,C), MHR: hpose.flatten(1) (no transpose)
  - No SMPL_MEAN_PARAMS or rot6d_to_rotmat dependency (MHR params are direct regression targets)
"""

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np

from mhr_constants import (
    NUM_IDENTITY_BLENDSHAPES,
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_LBS_MODEL_PARAMS,
    NUM_REGRESSOR_FEATURE_JOINTS,
)


class MultiLinear(torch.nn.Module):
    # MAPPED FROM: train/models/head/refit_regressor.py:MultiLinear (L13)
    # Identical to original — unchanged. Used by _make_multilinear which is
    # NOT used by MHRRegressor (pose branch uses flat MLP instead).
    """Multi-headed linear layer for per-joint processing.

    This is the same as in the original Regressor, used for processing
    multiple joints in parallel with shared weights.
    """
    __constants__ = ['n_head', 'in_features', 'out_features']
    n_head: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                 n_head: int,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        # MAPPED FROM: refit_regressor.py:MultiLinear.__init__ (L22)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiLinear, self).__init__()
        self.n_head = n_head
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty((n_head, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(
                torch.empty(n_head, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # MAPPED FROM: refit_regressor.py:MultiLinear.reset_parameters (L38)
        # Identical to original (uses np.sqrt(5) = math.sqrt(5))
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # MAPPED FROM: refit_regressor.py:MultiLinear.forward (L48)
        # Identical to original
        out = torch.einsum('kij, bkj -> bki', self.weight, input)
        if self.bias is not None:
            out += self.bias
        return out.contiguous()

    def extra_repr(self) -> str:
        # MAPPED FROM: refit_regressor.py:MultiLinear.extra_repr (L54)
        # Identical to original
        return 'n_head={}, in_features={}, out_features={}, bias={}'.format(
            self.n_head, self.in_features, self.out_features, self.bias
            is not None)


class MHRRegressor(nn.Module):
    # MAPPED FROM: train/models/head/refit_regressor.py:Regressor (L59)
    """Regressor that outputs MHR parameters.

    This regressor replaces the SMPL-based Regressor and outputs:
    - Identity coefficients (45 dims): Body shape blendshapes
    - Face expression coefficients (72 dims): Facial expressions
    - LBS model parameters (204 dims): Joint poses and rigid transforms
    - Camera parameters (3 dims): Camera translation/scale

    The architecture follows the same iterative refinement pattern as
    the original ReFit regressor, adapted for MHR's parameter structure.

    Unlike SMPL which has 22 joints with 6D pose each, MHR uses a flat
    204-dimensional lbs_model_params vector that encodes the full pose.

    Args:
        input_dim: Dimension of input features (from backbone)
        hidden_dim: Hidden layer dimension
        num_layer: Number of layers in the MLP
    """

    def __init__(self,
                 input_dim=32,
                 hidden_dim=64, #This controls how fat the regressor is!
                 num_layer=1,
                 use_depth=False):
        # MAPPED FROM: Regressor.__init__ (refit_regressor.py:L60)
        super(MHRRegressor, self).__init__()

        # Add bbox info dimension (and depth_feats channels if used)
        # ← original: input_dim = input_dim + 3  [same]
        input_dim = input_dim + 3
        if use_depth:
            input_dim = input_dim + 48  # UNET intermediate features are 48 channels

        # Feature-tensor J dim — fixed at 22 by the upstream HRNet/attention
        # pretraining. NOT a semantic body-joint count.
        # ← original: hardcoded 22 in MultiLinear(n_head=22)
        num_joints = NUM_REGRESSOR_FEATURE_JOINTS

        # Flattened feature dimension: (C+3) * J
        flat_dim = input_dim * num_joints

        # Input dimensions for each branch.
        # MHR lbs_model_params is a heterogeneous 204-D vector
        # ([tx,ty,tz, rx,ry,rz, per-joint-angles..., scale-params...]), NOT
        # per-joint-6D.  The pose branch therefore uses a flat MLP instead of
        # the per-joint MultiLinear used for SMPL.
        # ← original: pose_input = (720+48+3)+6, shape_input = (720+48+3)*22+11, cam_input = (720+48+3)*22+3
        #   These used flat_dim*22 (transposed then flattened) + param_dim
        # ← new: flat_dim + param_dim (already flat, no transpose needed)
        pose_input = flat_dim + NUM_LBS_MODEL_PARAMS  # flat features + current pose
        shape_input = flat_dim + NUM_IDENTITY_BLENDSHAPES
        expr_input = flat_dim + NUM_FACE_EXPRESSION_BLENDSHAPES
        cam_input = flat_dim + 3

        # Initialize with neutral values (MHR default pose)
        # ← original: init_pose loaded from SMPL_MEAN_PARAMS['pose'] (6D per joint, 22 joints)
        # ← original: init_shape loaded from SMPL_MEAN_PARAMS['shape'] (11 betas + 1 pad)
        # ← new: all zeros — MHR lbs_model_params identity pose has no mean-file equivalent
        init_identity = torch.zeros(NUM_IDENTITY_BLENDSHAPES).unsqueeze(0)
        init_expr = torch.zeros(NUM_FACE_EXPRESSION_BLENDSHAPES).unsqueeze(0)
        init_pose = torch.zeros(NUM_LBS_MODEL_PARAMS).unsqueeze(0)
        init_cam = torch.tensor([[0.9, 0.0, 0.0]])

        self.register_buffer('init_identity', init_identity)
        self.register_buffer('init_expr', init_expr)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_cam', init_cam)

        # MLP layers for each branch (all flat)
        # ← original: self.p = self._make_multilinear(num_layer, 22, pose_input, hidden_dim)
        #   MultiLinear with 22 heads (one per SMPL joint) for per-joint 6D pose residuals
        # ← new: flat MLP (no MultiLinear) — MHR lbs_model_params is a flat 204-D vector
        self.p = self._make_linear(num_layer, pose_input, hidden_dim)
        self.s = self._make_linear(num_layer, shape_input, hidden_dim)
        self.e = self._make_linear(num_layer, expr_input, hidden_dim)
        self.c = self._make_linear(num_layer, cam_input, hidden_dim)

        # Output decoders
        # ← original: decpose = MultiLinear(22, hidden_dim, 6) → per-joint 6D residuals
        # ← new: decpose = nn.Linear(hidden_dim, 204) → flat lbs_model_params residuals
        self.decpose = nn.Linear(hidden_dim, NUM_LBS_MODEL_PARAMS)  # 204 flat
        # ← original: decshape = nn.Linear(hidden_dim, 11) → 11-D betas
        # ← new: 45-D identity blendshapes
        self.decshape = nn.Linear(hidden_dim, NUM_IDENTITY_BLENDSHAPES)  # 45
        # ← NEW: no original equivalent — MHR has separate expression branch
        self.decexpr = nn.Linear(hidden_dim,
                                 NUM_FACE_EXPRESSION_BLENDSHAPES)  # 72
        # ← original: deccam = nn.Linear(hidden_dim, 3)  [same]
        self.deccam = nn.Linear(hidden_dim, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1))
        self.num_joints = num_joints

    def forward(self, hpose, hshape, hcam, bbox_info, depth_feats=None):
        # MAPPED FROM: Regressor.forward (refit_regressor.py:L85)
        #
        # Key flow differences:
        #   ORIGINAL:
        #     1. bbox/depth appended per-joint → hpose.transpose(1,2) (C,J→J,C)
        #     2. hshape/hcam transposed, hshape/hcam flattened
        #     3. pred_pose init: [B, 22, 6] (reshaped from flat init_pose)
        #     4. cat([pred_pose, hpose], dim=2) — per-joint feature concat
        #     5. pred_pose + MultiLinear(22) residual → [B, 22, 6]
        #     6. pred_shape + Linear residual → [B, 11]
        #     7. rot6d_to_rotmat(d_pose) → [B, 22, 3, 3]
        #     8. Returns {pred_pose: rotmat, pred_shape: betas, pred_cam, pred_pose_6d}
        #
        #   MHR:
        #     1. bbox/depth appended → hpose/hshape/hcam flattened via .flatten(1) (no transpose)
        #     2. pred_pose init: [B, 204], pred_identity: [B, 45], pred_expr: [B, 72]
        #     3. cat([pred_pose, hpose_flat], dim=1) — flat feature concat
        #     4. pred_pose + Linear residual → [B, 204]
        #     5. pred_identity + Linear residual → [B, 45]
        #     6. pred_expr + Linear residual → [B, 72]
        #     7. Returns {pred_identity, pred_expr, pred_pose (flat lbs), pred_cam}
        """Forward pass predicting MHR parameters.

        Args:
            hpose: Pose features [B, C, J] where J is number of joints
            hshape: Shape features [B, C, J]
            hcam: Camera features [B, C, J]
            bbox_info: Bounding box info [B, 3]
            depth_feats: Optional depth features [B, C, H, W]

        Returns:
            Dictionary with MHR parameters:
            - pred_identity: [B, 45] identity blendshape coefficients
            - pred_expr: [B, 72] face expression coefficients
            - pred_pose: [B, 204] LBS model parameters (flat)
            - pred_cam: [B, 3] camera parameters
        """
        BN = hpose.shape[0]

        # Add bbox info to features (broadcast across joints)
        # ← original: bbox_info.unsqueeze(-1).repeat(1,1,22) — hardcoded 22
        # ← new: uses self.num_joints (22, same value)
        hpose = torch.cat(
            [hpose,
             bbox_info.unsqueeze(-1).repeat(1, 1, self.num_joints)], 1)
        hshape = torch.cat(
            [hshape,
             bbox_info.unsqueeze(-1).repeat(1, 1, self.num_joints)], 1)
        hcam = torch.cat(
            [hcam, bbox_info.unsqueeze(-1).repeat(1, 1, self.num_joints)], 1)

        if depth_feats is not None:
            depth_feats = self.avgpool(depth_feats)
            hpose = torch.cat(
                [hpose,
                 depth_feats.squeeze(-1).repeat(1, 1, self.num_joints)], 1)
            hshape = torch.cat([
                hshape,
                depth_feats.squeeze(-1).repeat(1, 1, self.num_joints)
            ], 1)
            hcam = torch.cat(
                [hcam,
                 depth_feats.squeeze(-1).repeat(1, 1, self.num_joints)], 1)

        # Flatten [B, C, J] → [B, C*J] for all branches
        # ← original: hpose.transpose(1,2) then hshape/hcam.transpose(1,2) then flatten
        #   MultiLinear expects [B, J, C] (transposed), outputs [B, J, out]
        # ← new: .flatten(1) keeps [B, C*J], no transpose — flat MLP doesn't need per-joint axis
        hpose_flat = hpose.flatten(1)
        hshape_flat = hshape.flatten(1)
        hcam_flat = hcam.flatten(1)

        # Initialize predictions with neutral values
        # ← original: pred_pose = init_pose.repeat(BN,1).reshape(BN,22,-1) → [B,22,6]
        # ← new: pred_identity [B,45], pred_expr [B,72], pred_pose [B,204]
        pred_identity = self.init_identity.repeat(BN, 1)
        pred_expr = self.init_expr.repeat(BN, 1)
        pred_pose = self.init_pose.repeat(BN, 1)  # [B, 204]
        pred_cam = self.init_cam.repeat(BN, 1)

        # Iteratively refine predictions (single iteration)
        # ← original: for i in range(1): ...  [same loop count]
        for _ in range(1):
            # Pose branch: flat features + current pose estimate
            # ← original: pose_feats_pred = torch.cat([pred_pose, hpose], 2) — per-joint concat
            # ← new: pose_feats_pred = torch.cat([pred_pose, hpose_flat], 1) — flat concat
            pose_feats_pred = torch.cat([pred_pose, hpose_flat], 1)
            shape_feats_pred = torch.cat([pred_identity, hshape_flat], 1)
            expr_feats_pred = torch.cat([pred_expr, hshape_flat], 1)
            cam_feats_pred = torch.cat([pred_cam, hcam_flat], 1)

            # Residual updates
            # ← original: pred_pose + MultiLinear(22) residual → [B,22,6], then rot6d_to_rotmat
            # ← new: pred_pose + Linear residual → [B,204] (flat lbs params)
            pred_pose = pred_pose + self.decpose(self.p(pose_feats_pred))
            pred_identity = pred_identity + self.decshape(
                self.s(shape_feats_pred))
            pred_expr = pred_expr + self.decexpr(self.e(expr_feats_pred))
            pred_cam = pred_cam + self.deccam(self.c(cam_feats_pred))

        # ← original: rotm = rot6d_to_rotmat(d_pose).view(BN,22,3,3)
        #   Returns {pred_pose: rotmat, pred_shape: betas, pred_cam, pred_pose_6d}
        # ← new: returns flat lbs_model_params directly, no Rodrigues conversion
        return {
            'pred_identity': pred_identity,  # [B, 45]
            'pred_expr': pred_expr,  # [B, 72]
            'pred_pose': pred_pose,  # [B, 204]
            'pred_cam': pred_cam,  # [B, 3]
        }

    def _make_linear(self, num, input_dim, hidden_dim):
        # MAPPED FROM: Regressor._make_linear (refit_regressor.py:L130)
        # Identical to original
        """Create a sequential MLP."""
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [nn.Linear(plane, hidden_dim), nn.ReLU(inplace=True)]
            layers.extend(layer)
            plane = hidden_dim
        return nn.Sequential(*layers)

    def _make_multilinear(self, num, n_head, input_dim, hidden_dim):
        # MAPPED FROM: Regressor._make_multilinear (refit_regressor.py:L142)
        # Identical to original — NOT used by MHRRegressor (pose branch uses flat MLP)
        """Create a sequential multi-linear network."""
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [
                MultiLinear(n_head, plane, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            layers.extend(layer)
            plane = hidden_dim
        return nn.Sequential(*layers)
