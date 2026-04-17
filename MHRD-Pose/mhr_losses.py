"""
MHR Loss Functions for D-Pose MHR port.

This module provides loss functions for training the MHR-based HMR model.
It replaces the SMPL-based losses and works directly with MHR parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mhr_constants import (
    NUM_IDENTITY_BLENDSHAPES,
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_LBS_MODEL_PARAMS,
)


def mhr_losses(
        pred_identity,
        pred_expr,
        pred_pose,
        gt_identity,
        gt_expr,
        gt_pose,
        criterion,
):
    """
    Compute parameter regression losses for MHR.

    Args:
        pred_identity: [B, 45] predicted identity coefficients
        pred_expr: [B, 72] predicted expression coefficients
        pred_pose: [B, 144] predicted LBS model parameters
        gt_identity: [B, 45] ground truth identity coefficients
        gt_expr: [B, 72] ground truth expression coefficients
        gt_pose: [B, 144] ground truth LBS model parameters
        criterion: Loss function (e.g., MSELoss)

    Returns:
        loss_regr_identity: Identity coefficient loss
        loss_regr_expr: Expression coefficient loss
        loss_regr_pose: Pose parameter loss
    """
    # Identity loss
    if len(pred_identity) > 0 and gt_identity is not None:
        loss_regr_identity = criterion(pred_identity, gt_identity)
    else:
        loss_regr_identity = torch.FloatTensor(1).fill_(0.).to(pred_identity.device)

    # Expression loss
    if len(pred_expr) > 0 and gt_expr is not None:
        loss_regr_expr = criterion(pred_expr, gt_expr)
    else:
        loss_regr_expr = torch.FloatTensor(1).fill_(0.).to(pred_expr.device)

    # Pose loss
    if len(pred_pose) > 0 and gt_pose is not None:
        loss_regr_pose = criterion(pred_pose, gt_pose)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(pred_pose.device)

    return loss_regr_identity, loss_regr_expr, loss_regr_pose


def projected_keypoint_loss(
        pred_keypoints_2d,
        gt_keypoints_2d,
        criterion,
):
    """
    Compute 2D projected keypoint loss.

    Args:
        pred_keypoints_2d: [B, J, 2] predicted 2D keypoints
        gt_keypoints_2d: [B, J, 3] ground truth 2D keypoints with visibility
        criterion: Loss function

    Returns:
        Loss weighted by visibility confidence
    """
    conf = gt_keypoints_2d[:, :, -1]
    conf[conf == -2] = 0
    conf = conf.unsqueeze(-1)
    loss = conf * criterion(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])
    return loss


def keypoint_3d_loss(
        pred_keypoints_3d,
        gt_keypoints_3d,
        criterion,
):
    """
    Compute 3D keypoint loss.

    Args:
        pred_keypoints_3d: [B, J, 3] predicted 3D keypoints
        gt_keypoints_3d: [B, J, 3] ground truth 3D keypoints
        criterion: Loss function

    Returns:
        3D keypoint regression loss
    """
    gt_keypoints_3d = gt_keypoints_3d.clone()
    pred_keypoints_3d = pred_keypoints_3d

    if len(gt_keypoints_3d) > 0:
        return criterion(pred_keypoints_3d, gt_keypoints_3d)
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_keypoints_3d.device)


def shape_loss(
        pred_vertices,
        gt_vertices,
        criterion,
):
    """
    Compute per-vertex shape reconstruction loss.

    Args:
        pred_vertices: [B, V, 3] predicted vertices
        gt_vertices: [B, V, 3] ground truth vertices
        criterion: Loss function (typically L1)

    Returns:
        Per-vertex reconstruction loss
    """
    if len(gt_vertices) > 0:
        return criterion(pred_vertices, gt_vertices)
    else:
        return torch.FloatTensor(1).fill_(0.).to(pred_vertices.device)


class MHRLoss(nn.Module):
    """Loss function for MHR-based HMR training.

    This loss function computes multiple components:
    1. Parameter regression losses (identity, expression, pose)
    2. 2D projected keypoint loss
    3. 3D keypoint loss
    4. Per-vertex shape loss
    5. Camera parameter loss

    Args:
        hparams: Hyperparameters containing loss weights
    """

    def __init__(self, hparams=None):
        super(MHRLoss, self).__init__()
        self.criterion_mse = nn.MSELoss()
        self.criterion_mse_noreduce = nn.MSELoss(reduction='none')
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l1_noreduce = nn.L1Loss(reduction='none')
        self.hparams = hparams

        # Loss weights from config
        self.loss_weight = hparams.MODEL.LOSS_WEIGHT if hparams else 1.0
        self.shape_loss_weight = hparams.MODEL.SHAPE_LOSS_WEIGHT if hparams else 1.0
        self.pose_loss_weight = hparams.MODEL.POSE_LOSS_WEIGHT if hparams else 1.0
        self.joint_loss_weight = hparams.MODEL.JOINT_LOSS_WEIGHT if hparams else 5.0
        self.keypoint_loss_weight_2d = hparams.MODEL.KEYPOINT_LOSS_WEIGHT if hparams else 10.0
        self.identity_loss_weight = getattr(hparams.MODEL, 'IDENTITY_LOSS_WEIGHT', 1.0) if hparams else 1.0
        self.expr_loss_weight = getattr(hparams.MODEL, 'EXPR_LOSS_WEIGHT', 0.5) if hparams else 0.5

        # Number of joints for MHR (from skeleton state)
        self.num_joints = 24

    def forward(self, pred, gt):
        """
        Compute total loss for MHR training.

        Args:
            pred: Dictionary with predicted MHR outputs
            gt: Dictionary with ground truth data

        Returns:
            loss: Total loss
            loss_dict: Dictionary with individual loss components
        """
        # Select criterion
        if self.hparams and self.hparams.TRIAL.criterion == 'mse':
            criterion = self.criterion_mse
            criterion_noreduce = self.criterion_mse_noreduce
        else:
            criterion = self.criterion_l1
            criterion_noreduce = self.criterion_l1_noreduce

        # Extract predictions
        pred_cam = pred['pred_cam']
        pred_identity = pred['pred_identity']
        pred_expr = pred['pred_expr']
        pred_pose = pred['pred_pose']
        pred_joints = pred['joints3d'][:, :self.num_joints]
        pred_keypoints_2d = pred['joints2d'][:, :self.num_joints]
        pred_vertices = pred['vertices']

        # Extract ground truth
        gt_identity = gt.get('identity_coeffs')
        gt_expr = gt.get('face_expr_coeffs')
        gt_pose = gt.get('lbs_model_params')
        # Prefer MHR skeleton joints; fall back to SMPL joints if not present
        gt_joints = gt.get('joints3d_mhr', gt.get('joints3d'))
        gt_vertices = gt.get('vertices')

        # Compute image size for scaling
        img_size = gt['orig_shape'].rot90().T.unsqueeze(1)

        # Compute 2D keypoint loss
        if self.hparams and self.hparams.TRIAL.bedlam_bbox:
            # Use full image keypoints
            pred_keypoints_2d[:, :, :2] = 2 * (pred_keypoints_2d[:, :, :2] / img_size) - 1
            gt_keypoints_2d = gt['keypoints_orig']
            gt_keypoints_2d[:, :, :2] = 2 * (gt_keypoints_2d[:, :, :2] / img_size) - 1
        else:
            gt_keypoints_2d = gt['keypoints']

        loss_keypoints = projected_keypoint_loss(
            pred_keypoints_2d,
            gt_keypoints_2d,
            criterion=criterion_noreduce,
        )

        if self.hparams and self.hparams.TRIAL.bedlam_bbox:
            loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] * 200.).unsqueeze(-1)
            loss_keypoints = loss_keypoints * loss_keypoints_scale.unsqueeze(1)
            loss_keypoints = loss_keypoints.mean()
        else:
            loss_keypoints = loss_keypoints.mean()

        # Compute MHR parameter losses
        loss_regr_identity, loss_regr_expr, loss_regr_pose = mhr_losses(
            pred_identity,
            pred_expr,
            pred_pose,
            gt_identity,
            gt_expr,
            gt_pose,
            criterion=criterion,
        )

        # Compute 3D keypoint loss
        if gt_joints is not None:
            loss_keypoints_3d = keypoint_3d_loss(
                pred_joints,
                gt_joints[:, :self.num_joints],
                criterion=criterion,
            )
        else:
            loss_keypoints_3d = torch.tensor(0.0, device=pred_joints.device)

        # Compute per-vertex shape loss
        if gt_vertices is not None:
            loss_shape = shape_loss(
                pred_vertices,
                gt_vertices,
                criterion=self.criterion_l1,
            )
        else:
            loss_shape = torch.tensor(0.0, device=pred_vertices.device)

        # Apply loss weights
        loss_shape *= self.shape_loss_weight
        loss_keypoints *= self.keypoint_loss_weight_2d
        loss_keypoints_3d *= self.joint_loss_weight
        loss_regr_pose *= self.pose_loss_weight
        loss_regr_identity *= self.identity_loss_weight
        loss_regr_expr *= self.expr_loss_weight

        # Camera loss (encourage reasonable camera parameters)
        pred_cam_clipped = torch.clamp(pred_cam[:, 0], min=-0.5, max=0.5)
        loss_cam = ((torch.exp(-pred_cam_clipped * 10)) ** 2).mean()

        # Build loss dictionary
        loss_dict = {
            'loss/loss_keypoints': loss_keypoints,
            'loss/loss_keypoints_3d': loss_keypoints_3d,
            'loss/loss_regr_identity': loss_regr_identity,
            'loss/loss_regr_expr': loss_regr_expr,
            'loss/loss_regr_pose': loss_regr_pose,
            'loss/loss_shape': loss_shape,
            'loss/loss_cam': loss_cam,
        }

        # Compute total loss
        loss = sum(loss_dict.values())
        loss *= self.loss_weight

        loss_dict['loss/loss'] = loss

        return loss, loss_dict
