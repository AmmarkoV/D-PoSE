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
    NUM_MHR_SKELETON_JOINTS,
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

    def __init__(self, n_head: int, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiLinear, self).__init__()
        self.n_head = n_head
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty(
            (n_head, out_features, in_features), **factory_kwargs
        ))
        if bias:
            self.bias = Parameter(torch.empty(
                n_head, out_features, **factory_kwargs
            ))
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
            self.n_head, self.in_features, self.out_features,
            self.bias is not None
        )


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

    def __init__(self, input_dim=32, hidden_dim=32, num_layer=1):
        super(MHRRegressor, self).__init__()

        # Add bbox info dimension
        input_dim = input_dim + 3

        # Number of MHR skeleton joints (144 / 6 = 24 joints)
        num_joints = NUM_MHR_SKELETON_JOINTS

        # Input dimensions for each branch
        # Following the original ReFit architecture but adapted for MHR
        # Pose: Each joint gets 6D output, processed with multi-linear
        pose_per_joint_input = input_dim + 6
        # Shape: Global features + identity params
        shape_input = input_dim * num_joints + NUM_IDENTITY_BLENDSHAPES
        # Expression: Global features + expression params
        expr_input = input_dim * num_joints + NUM_FACE_EXPRESSION_BLENDSHAPES
        # Camera: Global features + cam params
        cam_input = input_dim * num_joints + 3

        # Initialize with neutral values (MHR default pose)
        # Identity coefficients: neutral body shape (zeros)
        init_identity = torch.zeros(NUM_IDENTITY_BLENDSHAPES).unsqueeze(0)
        # Face expression coefficients: neutral expression (zeros)
        init_expr = torch.zeros(NUM_FACE_EXPRESSION_BLENDSHAPES).unsqueeze(0)
        # LBS model params: neutral pose (zeros)
        init_pose = torch.zeros(NUM_LBS_MODEL_PARAMS).unsqueeze(0)
        # Camera: default camera position [scale, tx, ty]
        init_cam = torch.tensor([[1.0, 0.0, 0.0]])  # Shape: [1, 3]

        self.register_buffer('init_identity', init_identity)
        self.register_buffer('init_expr', init_expr)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_cam', init_cam)

        # MLP layers for each branch
        self.p = self._make_multilinear(num_layer, num_joints,
                                         pose_per_joint_input, hidden_dim)
        self.s = self._make_linear(num_layer, shape_input, hidden_dim)
        self.e = self._make_linear(num_layer, expr_input, hidden_dim)
        self.c = self._make_linear(num_layer, cam_input, hidden_dim)

        # Output decoders
        # Pose: 6D per joint, reshaped to flat 144
        self.decpose = MultiLinear(num_joints, hidden_dim, 6)
        # Shape: 45 identity blendshapes
        self.decshape = nn.Linear(hidden_dim, NUM_IDENTITY_BLENDSHAPES)
        # Expression: 72 face expression blendshapes
        self.decexpr = nn.Linear(hidden_dim, NUM_FACE_EXPRESSION_BLENDSHAPES)
        # Camera: 3 params
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
            - pred_pose: [B, 144] LBS model parameters
            - pred_cam: [B, 3] camera parameters
        """
        BN = hpose.shape[0]

        # Add bbox info to features (broadcast across joints)
        hpose = torch.cat([hpose, bbox_info.unsqueeze(-1).repeat(1, 1, self.num_joints)], 1)
        hshape = torch.cat([hshape, bbox_info.unsqueeze(-1).repeat(1, 1, self.num_joints)], 1)
        hcam = torch.cat([hcam, bbox_info.unsqueeze(-1).repeat(1, 1, self.num_joints)], 1)

        if depth_feats is not None:
            depth_feats = self.avgpool(depth_feats)
            hpose = torch.cat([hpose, depth_feats.squeeze(-1).repeat(1, 1, self.num_joints)], 1)
            hshape = torch.cat([hshape, depth_feats.squeeze(-1).repeat(1, 1, self.num_joints)], 1)
            hcam = torch.cat([hcam, depth_feats.squeeze(-1).repeat(1, 1, self.num_joints)], 1)

        # Transpose for processing: [B, C, J] -> [B, J, C]
        hpose = hpose.transpose(1, 2)
        hshape = hshape.transpose(1, 2)
        hcam = hcam.transpose(1, 2)

        # Flatten spatial dimensions for shape/cam branches
        hshape_flat = hshape.flatten(1)
        hcam_flat = hcam.flatten(1)

        # Initialize predictions with neutral values
        pred_identity = self.init_identity.repeat(BN, 1)
        pred_expr = self.init_expr.repeat(BN, 1)
        pred_pose = self.init_pose.repeat(BN, 1)
        pred_cam = self.init_cam.repeat(BN, 1)

        # Reshape pose for per-joint processing: [B, 144] -> [B, 24, 6]
        pred_pose_reshaped = pred_pose.view(BN, self.num_joints, -1)

        # Iteratively refine predictions
        for i in range(1):
            # Pose branch: concatenate per-joint features
            pose_feats_pred = torch.cat([pred_pose_reshaped, hpose], 2)

            # Shape branch: global features + current prediction
            shape_feats_pred = torch.cat([pred_identity, hshape_flat], 1)

            # Expression branch: global features + current prediction
            expr_feats_pred = torch.cat([pred_expr, hshape_flat], 1)

            # Camera branch: global features + current prediction
            cam_feats_pred = torch.cat([pred_cam, hcam_flat], 1)

            # Update predictions using residual connections
            pose_update = self.decpose(self.p(pose_feats_pred))
            pred_pose_reshaped = pred_pose_reshaped + pose_update

            pred_identity = pred_identity + self.decshape(self.s(shape_feats_pred))
            pred_expr = pred_expr + self.decexpr(self.e(expr_feats_pred))
            pred_cam = pred_cam + self.deccam(self.c(cam_feats_pred))

        # Flatten pose back to [B, 144]
        pred_pose = pred_pose_reshaped.flatten(1)

        return {
            'pred_identity': pred_identity,    # [B, 45]
            'pred_expr': pred_expr,            # [B, 72]
            'pred_pose': pred_pose,            # [B, 144]
            'pred_cam': pred_cam,              # [B, 3]
        }

    def _make_linear(self, num, input_dim, hidden_dim):
        """Create a sequential MLP."""
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [nn.Linear(plane, hidden_dim),
                     nn.ReLU(inplace=True)]
            layers.extend(layer)
            plane = hidden_dim
        return nn.Sequential(*layers)

    def _make_multilinear(self, num, n_head, input_dim, hidden_dim):
        """Create a sequential multi-linear network."""
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [MultiLinear(n_head, plane, hidden_dim),
                     nn.ReLU(inplace=True)]
            layers.extend(layer)
            plane = hidden_dim
        return nn.Sequential(*layers)
