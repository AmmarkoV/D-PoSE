"""
MHR Regressor for D-Pose MHR port.

This module replaces the SMPL-based Regressor with an MHR-based version
that predicts MHR parameters (identity, expression, pose) instead of
SMPL betas and rotation matrices.

The regressor follows the same iterative refinement approach as the
original ReFit-based regressor, but outputs MHR-compatible parameters.

MHR Parameter Structure:
- identity_coeffs: [B, 45] - body shape blendshapes
- lbs_model_params: [B, 144] - rigid transforms + joint poses
- face_expr_coeffs: [B, 72] - facial expressions
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
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        out = torch.einsum('kij, bkj -> bki', self.weight, input)
        if self.bias is not None:
            out += self.bias
        return out.contiguous()

    def extra_repr(self) -> str:
        return 'n_head={}, in_features={}, out_features={}, bias={}'.format(
            self.n_head, self.in_features, self.out_features, self.bias
            is not None)


class MHRRegressor(nn.Module):
    """Regressor that outputs MHR parameters.

    This regressor replaces the SMPL-based Regressor and outputs:
    - Identity coefficients (45 dims): Body shape blendshapes
    - Face expression coefficients (72 dims): Facial expressions
    - LBS model parameters (144 dims): Joint poses and rigid transforms
    - Camera parameters (3 dims): Camera translation/scale

    The architecture follows the same iterative refinement pattern as
    the original ReFit regressor, adapted for MHR's parameter structure.

    Unlike SMPL which has 22 joints with 6D pose each, MHR uses a flat
    144-dimensional lbs_model_params vector that encodes the full pose.

    Args:
        input_dim: Dimension of input features (from backbone)
        hidden_dim: Hidden layer dimension
        num_layer: Number of layers in the MLP
    """

    def __init__(self,
                 input_dim=32,
                 hidden_dim=1024,
                 num_layer=1,
                 use_depth=False):
        super(MHRRegressor, self).__init__()

        # Add bbox info dimension (and depth_feats channels if used)
        input_dim = input_dim + 3
        if use_depth:
            input_dim = input_dim + 48  # UNET intermediate features are 48 channels

        # Feature-tensor J dim — fixed at 22 by the upstream HRNet/attention
        # pretraining. NOT a semantic body-joint count.
        num_joints = NUM_REGRESSOR_FEATURE_JOINTS

        # Flattened feature dimension: (C+3) * J
        flat_dim = input_dim * num_joints

        # Input dimensions for each branch.
        # MHR lbs_model_params is a heterogeneous 204-D vector
        # ([tx,ty,tz, rx,ry,rz, per-joint-angles..., scale-params...]), NOT
        # per-joint-6D.  The pose branch therefore uses a flat MLP instead of
        # the per-joint MultiLinear used for SMPL.
        pose_input = flat_dim + NUM_LBS_MODEL_PARAMS  # flat features + current pose
        shape_input = flat_dim + NUM_IDENTITY_BLENDSHAPES
        expr_input = flat_dim + NUM_FACE_EXPRESSION_BLENDSHAPES
        cam_input = flat_dim + 3

        # Initialize with neutral values (MHR default pose)
        init_identity = torch.zeros(NUM_IDENTITY_BLENDSHAPES).unsqueeze(0)
        init_expr = torch.zeros(NUM_FACE_EXPRESSION_BLENDSHAPES).unsqueeze(0)
        init_pose = torch.zeros(NUM_LBS_MODEL_PARAMS).unsqueeze(0)
        init_cam = torch.tensor([[1.0, 0.0, 0.0]])

        self.register_buffer('init_identity', init_identity)
        self.register_buffer('init_expr', init_expr)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_cam', init_cam)

        # MLP layers for each branch (all flat)
        self.p = self._make_linear(num_layer, pose_input, hidden_dim)
        self.s = self._make_linear(num_layer, shape_input, hidden_dim)
        self.e = self._make_linear(num_layer, expr_input, hidden_dim)
        self.c = self._make_linear(num_layer, cam_input, hidden_dim)

        # Output decoders
        self.decpose = nn.Linear(hidden_dim, NUM_LBS_MODEL_PARAMS)  # 204 flat
        self.decshape = nn.Linear(hidden_dim, NUM_IDENTITY_BLENDSHAPES)  # 45
        self.decexpr = nn.Linear(hidden_dim,
                                 NUM_FACE_EXPRESSION_BLENDSHAPES)  # 72
        self.deccam = nn.Linear(hidden_dim, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1))
        self.num_joints = num_joints

    def forward(self, hpose, hshape, hcam, bbox_info, depth_feats=None):
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
        hpose_flat = hpose.flatten(1)
        hshape_flat = hshape.flatten(1)
        hcam_flat = hcam.flatten(1)

        # Initialize predictions with neutral values
        pred_identity = self.init_identity.repeat(BN, 1)
        pred_expr = self.init_expr.repeat(BN, 1)
        pred_pose = self.init_pose.repeat(BN, 1)  # [B, 204]
        pred_cam = self.init_cam.repeat(BN, 1)

        # Iteratively refine predictions (single iteration)
        for _ in range(1):
            # Pose branch: flat features + current pose estimate
            pose_feats_pred = torch.cat([pred_pose, hpose_flat], 1)
            shape_feats_pred = torch.cat([pred_identity, hshape_flat], 1)
            expr_feats_pred = torch.cat([pred_expr, hshape_flat], 1)
            cam_feats_pred = torch.cat([pred_cam, hcam_flat], 1)

            # Residual updates
            pred_pose = pred_pose + self.decpose(self.p(pose_feats_pred))
            pred_identity = pred_identity + self.decshape(
                self.s(shape_feats_pred))
            pred_expr = pred_expr + self.decexpr(self.e(expr_feats_pred))
            pred_cam = pred_cam + self.deccam(self.c(cam_feats_pred))

        return {
            'pred_identity': pred_identity,  # [B, 45]
            'pred_expr': pred_expr,  # [B, 72]
            'pred_pose': pred_pose,  # [B, 204]
            'pred_cam': pred_cam,  # [B, 3]
        }

    def _make_linear(self, num, input_dim, hidden_dim):
        """Create a sequential MLP."""
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [nn.Linear(plane, hidden_dim), nn.ReLU(inplace=True)]
            layers.extend(layer)
            plane = hidden_dim
        return nn.Sequential(*layers)

    def _make_multilinear(self, num, n_head, input_dim, hidden_dim):
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
