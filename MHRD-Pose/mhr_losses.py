"""
MHR Loss Functions for D-Pose MHR port.

MAPPED FROM: train/losses/losses.py (HMRLoss — SMPL/SMPLX version)

This module provides loss functions for training the MHR-based HMR model.
It replaces the SMPL-based losses and works directly with MHR parameters.

File correspondence:
  mhr_losses.py:mhr_losses              ←  train/losses/losses.py:smpl_losses
  mhr_losses.py:projected_keypoint_loss  ←  train/losses/losses.py:projected_keypoint_loss  (identical)
  mhr_losses.py:keypoint_3d_loss         ←  train/losses/losses.py:keypoint_3d_loss        (identical)
  mhr_losses.py:shape_loss               ←  train/losses/losses.py:shape_loss              (identical)
  mhr_losses.py:MHRLoss                  ←  train/losses/losses.py:HMRLoss

Key differences from original HMRLoss:
  - Original predicts: pred_pose (rotmat), pred_shape (betas), pred_cam
    MHR predicts: pred_identity, pred_expr, pred_pose (lbs_params), pred_cam
  - Original loss: smpl_losses(pred_rotmat, pred_betas, ...) → loss_regr_pose, loss_regr_betas
    MHR loss: mhr_losses(pred_identity, pred_expr, pred_pose, ...) → loss_regr_identity, loss_regr_expr, loss_regr_pose
  - Original weight: beta_loss_weight for betas
    MHR weight: identity_loss_weight, expr_loss_weight (split from single beta weight)
  - Original 3D keypoint loss: pred_joints[:, :self.num_joints] vs gt_joints[:, :self.num_joints]
    MHR 3D keypoint loss: tries joints3d_smpl first, falls back to MHR-to-SMPL index mapping
  - Original has many conditional loss branches (losses_abl variants, SUPERVISE_SEGM, SUPERVISE_DEPTH)
    MHR: simplified — always returns the same loss_dict structure, no conditional branches
  - Original: self.num_joints = 49 (SMPL) or 24 (SMPLX) based on TRIAL.version
    MHR: self.num_joints = NUM_MHR_SKELETON_JOINTS (127)
  - Original: uses batch_rodrigues to convert gt_pose rotvec → rotmat internally in smpl_losses
    MHR: lbs_model_params are direct regression targets, no Rodrigues conversion
  - Original: has depth_loss, ssim_loss, projected_verts_loss, reconstruction_error, generate_part_labels
    MHR: no depth/segm/SSIM losses (those are handled separately if needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mhr_constants import (
    NUM_IDENTITY_BLENDSHAPES,
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_LBS_MODEL_PARAMS,
    NUM_MHR_SKELETON_JOINTS,
    MHR_TO_SMPL_JOINT_INDICES,
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
    # MAPPED FROM: train/losses/losses.py:smpl_losses (L577-596)
    #
    # Original smpl_losses:
    #   - Converts pred_rotmat (rot6d or rotmat) and gt_pose (rotvec) to rotmat via batch_rodrigues
    #   - Returns (loss_regr_pose, loss_regr_betas)
    #   - Uses gt_pose.reshape(-1, 3) → batch_rodrigues for pose conversion
    #
    # MHR version:
    #   - Direct L2 regression on lbs_model_params (no Rodrigues conversion needed)
    #   - Returns (loss_regr_identity, loss_regr_expr, loss_regr_pose) — three-way split
    #   - No smplx-specific batch_rodrigues dependency
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
        loss_regr_identity = torch.FloatTensor(1).fill_(0.).to(
            pred_identity.device)

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
    # MAPPED FROM: train/losses/losses.py:HMRLoss (L234)
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
        # MAPPED FROM: HMRLoss.__init__ (train/losses/losses.py:L235)
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
        # ← original: self.beta_loss_weight = self.hparams.MODEL.BETA_LOSS_WEIGHT
        #   Replaced by two separate weights for identity and expression
        self.identity_loss_weight = getattr(
            hparams.MODEL, 'IDENTITY_LOSS_WEIGHT', 1.0) if hparams else 1.0
        self.expr_loss_weight = getattr(hparams.MODEL, 'EXPR_LOSS_WEIGHT',
                                        0.5) if hparams else 0.5

        # ← original: self.num_joints = 49 (SMPL) or 24 (SMPLX) based on TRIAL.version
        #   MHR uses the full skeleton joint count
        self.num_joints = NUM_MHR_SKELETON_JOINTS

    def forward(self, pred, gt):
        # MAPPED FROM: HMRLoss.forward (train/losses/losses.py:L286)
        #
        # Key structural differences from original:
        #   ORIGINAL:
        #     - pred: pred_cam, pred_betas, pred_rotmat, pred_joints, pred_keypoints_2d, pred_vertices
        #     - gt: betas, joints3d, vertices, pose, keypoints_orig, ...
        #     - smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas) → pose + beta losses
        #     - Conditional loss branches (losses_abl, SUPERVISE_SEGM, SUPERVISE_DEPTH, proj_verts)
        #     - depth_loss + ssim_loss for depth supervision
        #     - cross_entropy_loss for segmentation
        #     - projected_verts_loss for 3D-to-2D projected vertex loss
        #
        #   MHR:
        #     - pred: pred_cam, pred_identity, pred_expr, pred_pose (lbs), pred_joints, pred_vertices
        #     - gt: identity_coeffs, face_expr_coeffs, lbs_model_params, joints3d_mhr, joints3d, vertices
        #     - mhr_losses(pred_identity, pred_expr, pred_pose, ...) → identity + expr + pose losses
        #     - Simplified: no conditional branches, always same loss_dict structure
        #     - No depth/segm/SSIM/proj-verts losses (handled separately if needed)
        #     - 3D joint loss: prefers joints3d_smpl, falls back to MHR-to-SMPL index mapping
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
        # Use SMPL-ordered 2D joints when available (added in mhr_head.py).
        # joints2d_smpl is projected from smpl_joints3d whose 24-joint ordering
        # (0=pelvis … 23=R-wrist) matches the GT keypoint annotations in
        # BEDLAM/3DPW, making the 2D keypoint loss meaningful.
        # Fall back to the 127-joint MHR skel_state projection only if the new
        # key is absent (e.g. running against an older checkpoint or test path).
        _joints2d_src = pred.get('joints2d_smpl', pred['joints2d'])
        pred_keypoints_2d = _joints2d_src[:, :self.num_joints]
        pred_vertices = pred['vertices']

        # Extract ground truth
        gt_identity = gt.get('identity_coeffs')
        gt_expr = gt.get('face_expr_coeffs')
        gt_pose = gt.get('lbs_model_params')
        # Prefer MHR skeleton joints; fall back to SMPL joints if not present.
        # Guard against cached joints3d_mhr with wrong shape (e.g. size 0 at last dim).
        gt_joints = gt.get('joints3d_mhr', gt.get('joints3d'))
        if gt_joints is not None and (gt_joints.ndim != 3
                                      or gt_joints.shape[-1] < 3):
            gt_joints = gt.get('joints3d')  # fall back to SMPL joints
        gt_vertices = gt.get('vertices')

        # Compute image size for scaling
        img_size = gt['orig_shape'].rot90().T.unsqueeze(1)

        # Compute 2D keypoint loss
        if self.hparams and self.hparams.TRIAL.bedlam_bbox:
            # Normalize to [-1, 1] using full image dimensions.
            #
            # IMPORTANT: use out-of-place arithmetic for pred_keypoints_2d.
            # pred_keypoints_2d is a view of joints2d_smpl which has
            # requires_grad=True. An in-place write (x[:,...] = ...) increments
            # the tensor's version counter, causing PyTorch autograd to silently
            # zero out the gradient on the backward pass. The loss value would
            # still be computed correctly but no gradient would reach the network,
            # so the 2D supervision would have zero effect and the reprojection
            # loss would only increase as other losses push the params.
            # Out-of-place creates a new node in the computation graph, keeping
            # the gradient path intact.
            pred_keypoints_2d = 2 * (pred_keypoints_2d / img_size) - 1

            # GT keypoints_orig is [B, J, 3]: (x, y, confidence). img_size is
            # [B, 1, 2], so we can only normalize the xy channels; dividing the
            # full [B,J,3] tensor would fail with a shape mismatch at dim 2.
            # projected_keypoint_loss expects [..., -1] to be the confidence
            # weight, so we reconstruct the 3-channel tensor after normalizing.
            _gt_orig = gt['keypoints_orig']
            gt_keypoints_2d = torch.cat([
                2 * (_gt_orig[:, :, :2] / img_size) - 1,
                _gt_orig[:, :, 2:3],
            ],
                                        dim=-1)
        else:
            gt_keypoints_2d = gt['keypoints']

        # GT keypoints may have more joints than pred (SMPL 24 vs MHR 22); align.
        num_kp = pred_keypoints_2d.shape[1]
        loss_keypoints = projected_keypoint_loss(
            pred_keypoints_2d,
            gt_keypoints_2d[:, :num_kp],
            criterion=criterion_noreduce,
        )

        if self.hparams and self.hparams.TRIAL.bedlam_bbox:
            loss_keypoints_scale = img_size.squeeze(1) / (gt['scale'] *
                                                          200.).unsqueeze(-1)
            loss_keypoints = loss_keypoints * loss_keypoints_scale.unsqueeze(1)
            loss_keypoints = loss_keypoints.mean()
        else:
            loss_keypoints = loss_keypoints.mean()

        # Compute MHR parameter losses
        # ← original: smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, criterion)
        #   which returned (loss_regr_pose, loss_regr_betas)
        #   Inside: batch_rodrigues(gt_pose.reshape(-1,3)) to convert rotvec→rotmat
        # ← new: mhr_losses(pred_identity, pred_expr, pred_pose, gt_identity, gt_expr, gt_pose, criterion)
        #   which returns (loss_regr_identity, loss_regr_expr, loss_regr_pose)
        #   Direct regression on lbs_model_params — no Rodrigues conversion
        loss_regr_identity, loss_regr_expr, loss_regr_pose = mhr_losses(
            pred_identity,
            pred_expr,
            pred_pose,
            gt_identity,
            gt_expr,
            gt_pose,
            criterion=criterion,
        )

        # Compute 3D keypoint loss using differentiable SMPL joints when available
        gt_joints_smpl = gt.get('joints3d')
        pred_joints_smpl = pred.get('joints3d_smpl')
        if pred_joints_smpl is not None and gt_joints_smpl is not None:
            # Pelvis-center both (average of left hip [1] and right hip [2])
            pred_pel = (pred_joints_smpl[:, [1]] +
                        pred_joints_smpl[:, [2]]) / 2.0
            gt_pel = (gt_joints_smpl[:, [1]] + gt_joints_smpl[:, [2]]) / 2.0
            loss_keypoints_3d = keypoint_3d_loss(
                pred_joints_smpl - pred_pel,
                (gt_joints_smpl - gt_pel)[:, :24],
                criterion=criterion,
            )
        elif gt_joints is not None:
            # Fallback: map raw 127-joint MHR skel_state → SMPL ordering.
            # The first 24 MHR joints are foot/twist procedurals, so direct
            # [:, :24] slicing would compare anatomically unrelated joints.
            idx = torch.as_tensor(
                MHR_TO_SMPL_JOINT_INDICES,
                dtype=torch.long,
                device=pred['joints3d'].device,
            )
            pred_full = pred['joints3d']  # [B, 127, 3]
            pred_mapped = pred_full.index_select(1, idx)
            if gt_joints.shape[1] >= max(MHR_TO_SMPL_JOINT_INDICES) + 1:
                gt_mapped = gt_joints.index_select(1, idx.to(gt_joints.device))
            else:
                gt_mapped = gt_joints[:, :self.num_joints]
            loss_keypoints_3d = keypoint_3d_loss(
                pred_mapped,
                gt_mapped,
                criterion=criterion,
            )
        else:
            loss_keypoints_3d = torch.tensor(0.0,
                                             device=pred['joints3d'].device)

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
        loss_cam = ((torch.exp(-pred_cam_clipped * 10))**2).mean()

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
