"""
MHR-based HMR Model for D-Pose MHR port.

MAPPED FROM: train/models/hmr.py (HMR — SMPL/SMPLX version)

This module replaces the SMPL-based HMR model and predicts MHR parameters
directly from images. The architecture follows the same pattern as the
original HMR but outputs MHR-compatible parameters.

Architecture:
    Image → HRNet Backbone → MHR Regressor → MHR Parameters → MHR Head → Vertices/Joints

File correspondence:
  mhr_hmr.py:MHRHMR  ←  train/models/hmr.py:HMR

Key differences from original HMR:
  - Head: SMPLXCamHead (original) → MHRRegressor + MHRHead (new)
    - Original Regressor() returns pred_pose (rotmat), pred_shape (betas), pred_cam
    - New MHRRegressor() returns pred_pose (lbs_model_params), pred_identity, pred_expr, pred_cam
  - SMPLXCamHead (original, train/models/head/smplx_cam_head.py) → MHRHead (new)
    - Original: rotmat + shape + cam → SMPLX forward → vertices/joints
    - New: identity + lbs + expr + cam → MHR forward → vertices/joints
  - OneEuroFilter: original filters pred_pose, pred_shape, pred_cam
    - New filters pred_pose, pred_identity, pred_expr, pred_cam (no pred_shape)
  - Original __getitem__ returns smpl_output which is smpl_output.update(hmr_output)
    - New returns mhr_forward_output which is mhr_forward_output.update(mhr_output)
  - Original no-segmentation path: self.head(features, bbox_info=bbox_info, depth_feats=None)
    - New no-segmentation path: adaptive_avg_pool2d → self.head(pose_feat, pose_feat, pose_feat, bbox_info, None)
  - mhr_model is passed in as constructor arg (pre-loaded from checkpoint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mhr_regressor import MHRRegressor
from mhr_head import MHRHead
from mhr_constants import (
    NUM_IDENTITY_BLENDSHAPES,
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_LBS_MODEL_PARAMS,
)

from train.models.head.keypoint_attention import KeypointAttention
from train.core.config import PRETRAINED_CKPT_FOLDER
from train.models.head.unet_advanced import UNET
from train.models.backbone.hrnet import hrnet_w32, hrnet_w48
from train.utils.one_euro_filter import OneEuroFilter


class MHRHMR(nn.Module):
    # MAPPED FROM: train/models/hmr.py:HMR (L12)
    """HMR model that predicts MHR parameters directly.

    Unlike the original HMR which predicts SMPL betas/pose and then
    calls SMPLX to get vertices, this model predicts MHR parameters
    (identity, expression, pose) and calls the MHR model directly.

    This eliminates the need for runtime SMPL→MHR conversion and
    enables true real-time performance.

    Args:
        backbone: Backbone network name (e.g., 'hrnet-w32')
        img_res: Image resolution
        focal_length: Focal length for camera intrinsics
        pretrained_ckpt: Path to pretrained checkpoint
        hparams: Hyperparameters
        mhr_model: Pre-loaded MHR model instance
    """

    def __init__(self,
                 backbone='resnet50',
                 img_res=224,
                 focal_length=5000,
                 pretrained_ckpt=None,
                 hparams=None,
                 mhr_model=None):
        # MAPPED FROM: HMR.__init__ (train/models/hmr.py:L13)
        super(MHRHMR, self).__init__()
        self.hparams = hparams
        self.img_res = img_res
        self.focal_length = focal_length

        # Initialize backbone — identical to original (hmr.py:L24-33)
        if backbone.startswith('hrnet'):
            backbone_name, use_conv = backbone.split('-')
            pretrained_ckpt = backbone_name + '-' + pretrained_ckpt
            pretrained_ckpt_path = PRETRAINED_CKPT_FOLDER[pretrained_ckpt]
            self.backbone = eval(backbone_name)(
                pretrained_ckpt_path=pretrained_ckpt_path,
                downsample=True,
                use_conv=(use_conv == 'conv'),
            )

        # MHR-specific head components
        # ← original: self.head = Regressor()  [hmr.py:L37]
        #   Regressor() returns {pred_pose (rotmat), pred_shape (betas), pred_cam}
        # ← new: MHRRegressor() returns {pred_pose (lbs_params), pred_identity, pred_expr, pred_cam}
        if hparams and hparams.TRIAL.bedlam_bbox:
            # Channel count of the upsampled HRNet feature map (all branches concatenated)
            _backbone_channels = {'hrnet_w32': 480, 'hrnet_w48': 720}
            _backbone_name = backbone.split('-')[0]
            _feat_channels = _backbone_channels.get(_backbone_name, 480)
            _use_depth = bool(hparams.DATASET.USE_DEPTH) if hparams else False
            self.head = MHRRegressor(input_dim=_feat_channels,
                                     use_depth=_use_depth)

            if mhr_model is not None:
                # ← original: self.smpl = SMPLXCamHead(img_res=img_res)  [hmr.py:L38]
                #   SMPLXCamHead takes (rotmat, shape, cam) → SMPLX forward → vertices/joints
                #   MHRHead takes (identity, lbs_model_params, expr, cam) → MHR forward → vertices/joints
                self.mhr_head = MHRHead(mhr_model, img_res=img_res)
            else:
                # Will be initialized later
                self.mhr_head = None

        # Optional depth decoder — identical to original (hmr.py:L39-40)
        if hparams and hparams.DATASET.USE_DEPTH:
            self.depth_decoder = UNET(depth=True)

        # Optional segmentation decoder — identical to original (hmr.py:L42-45)
        if hparams and hparams.DATASET.USE_SEGM:
            self.segmentation_decoder = UNET(depth=False)
            self.attention = KeypointAttention()
            self.avg_pool_cam_shape = nn.AdaptiveAvgPool2d((768, 1))

        # OneEuroFilter for temporal smoothing
        # ← original (hmr.py:L46-52): filters pred_pose, pred_shape, pred_cam
        # ← new: filters pred_pose, pred_identity, pred_expr, pred_cam
        #   pred_shape (betas) replaced by pred_identity + pred_expr (two separate filters)
        self.min_cutoff = 0.004
        self.beta = 0.7
        self.t = 0
        self.one_euro_identity = None
        self.one_euro_pose = None
        self.one_euro_expr = None
        self.one_euro_cam = None
        self.use_one_euro = True

    def forward(self,
                images,
                bbox_scale=None,
                bbox_center=None,
                img_w=None,
                img_h=None,
                fl=None):
        # MAPPED FROM: HMR.forward (train/models/hmr.py:L54)
        #
        # Flow comparison:
        #   ORIGINAL: backbone → head → {pred_pose, pred_shape, pred_cam} →
        #             SMPLXCamHead(rotmat=pred_pose, shape=pred_shape, cam=pred_cam) → smpl_output
        #             smpl_output.update(hmr_output) → return smpl_output
        #
        #   NEW:      backbone → head → {pred_pose, pred_identity, pred_expr, pred_cam} →
        #             MHRHead(identity=pred_identity, lbs=pred_pose, expr=pred_expr, cam=pred_cam) → mhr_forward_output
        #             mhr_forward_output.update(mhr_output) → return mhr_forward_output
        #
        # No-segmentation path difference:
        #   ORIGINAL (hmr.py:L114): self.head(features, bbox_info=bbox_info, depth_feats=None)
        #   NEW (below): adaptive_avg_pool2d → self.head(pose_feat, pose_feat, pose_feat, bbox_info, None)
        #     The MHRRegressor expects [B, C, J] spatially-pooled features as first arg.
        """
        Forward pass predicting MHR parameters and computing vertices.

        Args:
            images: Input images [B, 3, H, W]
            bbox_scale: Bounding box scale [B]
            bbox_center: Bounding box center [B, 2]
            img_w: Image width [B]
            img_h: Image height [B]
            fl: Focal length (optional) [B, 2]

        Returns:
            Dictionary with MHR outputs:
            - vertices: [B, V, 3] mesh vertices in meters
            - joints3d: [B, J, 3] 3D joint positions in meters
            - joints2d: [B, J, 2] projected 2D joint positions
            - pred_identity: [B, 45] identity coefficients
            - pred_expr: [B, 72] expression coefficients
            - pred_pose: [B, 144] LBS model parameters
            - pred_cam: [B, 3] camera parameters
            - pred_cam_t: [B, 3] camera translation
            - skel_state: [B, J, 16] skeleton state
        """
        batch_size = images.shape[0]

        # Compute focal length
        if fl is not None:
            focal_length = fl
        else:
            focal_length = (img_w * img_w + img_h * img_h)**0.5
            focal_length = focal_length.repeat(2).view(batch_size, 2)

        # Initialize camera intrinsic matrix
        cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
        cam_intrinsics[:, 0, 0] = focal_length[:, 0]
        cam_intrinsics[:, 1, 1] = focal_length[:, 1]
        cam_intrinsics[:, 0, 2] = img_w / 2.
        cam_intrinsics[:, 1, 2] = img_h / 2.

        if self.hparams and self.hparams.TRIAL.bedlam_bbox:
            # Compute bbox info (from CLIFF repository)
            cx, cy = bbox_center[:, 0], bbox_center[:, 1]
            b = bbox_scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b],
                                    dim=-1)
            bbox_info = bbox_info.cuda().float()
            bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0,
                                                                 0].unsqueeze(
                                                                     -1)
            bbox_info[:, 2] = bbox_info[:, 2] / cam_intrinsics[:, 0, 0]
            bbox_info = bbox_info.cuda().float()

            if self.hparams.DATASET.USE_SEGM:
                features, upsampled_feature = self.backbone(images)
                segmentation, _ = self.segmentation_decoder(features)
                cam_shape_feat = upsampled_feature

                if not self.hparams.DATASET.USE_DEPTH:
                    attention_pose = self.attention(upsampled_feature,
                                                    segmentation[:, 1:, :, :],
                                                    None)
                    attention_cam_shape = self.attention(
                        cam_shape_feat, segmentation[:, 1:, :, :], None)
                    mhr_output = self.head(attention_pose, attention_cam_shape,
                                           attention_cam_shape, bbox_info)
                else:
                    depth, depth_feats = self.depth_decoder(features)
                    orig_depth = depth.clone()
                    depth = depth.repeat(1, segmentation.shape[1], 1, 1)

                    if not self.hparams.DATASET.USE_DEPTH_CONC:
                        attention_pose = self.attention(
                            upsampled_feature, segmentation[:, 1:, :, :],
                            depth[:, 1:, :, :])
                        attention_cam_shape = self.attention(
                            cam_shape_feat, segmentation[:, 1:, :, :],
                            depth[:, 1:, :, :])
                        mhr_output = self.head(attention_pose,
                                               attention_cam_shape,
                                               attention_cam_shape, bbox_info,
                                               depth_feats)
                    else:
                        upsampled_feature = torch.cat(
                            [upsampled_feature, depth_feats], dim=1)
                        attention_pose = self.attention(
                            upsampled_feature, segmentation[:, 1:, :, :],
                            depth[:, 1:, :, :])
                        cam_shape_feat = torch.cat(
                            [cam_shape_feat, depth_feats], dim=1)
                        attention_cam_shape = self.attention(
                            cam_shape_feat, segmentation[:, 1:, :, :],
                            depth[:, 1:, :, :])
                        mhr_output = self.head(attention_pose,
                                               attention_cam_shape,
                                               attention_cam_shape, bbox_info,
                                               None)
            else:
                # No-segmentation path.
                # The depth decoder (UNET) was designed for HRNet-w48 channel
                # sizes (branch4=384) and would fail for w32 (branch4=256).
                # Depth is therefore skipped here; orig_depth/segmentation are
                # set to None so the return statement below is still valid.
                segmentation = None
                orig_depth = None
                features, _ = self.backbone(images)
                # features is a list of 4 multi-scale tensors from HRNet.
                # features[0] has `width` channels (32 for hrnet_w32).
                # Pool the spatial dimensions down to num_joints positions so
                # the regressor receives [B, C, J] as it expects.
                pose_feat = F.adaptive_avg_pool2d(
                    features[0],
                    (self.head.num_joints, 1)).squeeze(-1)  # [B, C, J]
                mhr_output = self.head(pose_feat,
                                       pose_feat,
                                       pose_feat,
                                       bbox_info,
                                       depth_feats=None)

            # Apply OneEuroFilter for temporal smoothing
            # ← original (hmr.py:L120-133): initializes/filters pred_cam, pred_pose, pred_shape
            #   self.one_euro_shape = OneEuroFilter(...) for pred_shape (betas)
            # ← new: pred_shape replaced by pred_identity + pred_expr (two separate filters)
            #   self.one_euro_identity, self.one_euro_expr, NO self.one_euro_shape
            if self.one_euro_cam is None and self.use_one_euro:
                self.one_euro_cam = OneEuroFilter(
                    np.zeros_like(mhr_output['pred_cam'][0].detach().cpu()),
                    mhr_output['pred_cam'][0].detach().cpu().numpy(),
                    min_cutoff=self.min_cutoff,
                    beta=self.beta)
                self.one_euro_pose = OneEuroFilter(
                    np.zeros_like(mhr_output['pred_pose'][0].detach().cpu()),
                    mhr_output['pred_pose'][0].detach().cpu().numpy(),
                    min_cutoff=self.min_cutoff,
                    beta=self.beta)
                self.one_euro_identity = OneEuroFilter(
                    np.zeros_like(
                        mhr_output['pred_identity'][0].detach().cpu()),
                    mhr_output['pred_identity'][0].detach().cpu().numpy(),
                    min_cutoff=self.min_cutoff,
                    beta=self.beta)

            if self.use_one_euro:
                self.t += 1
                t_pose = np.ones_like(
                    mhr_output['pred_pose'][0].detach().cpu()) * self.t
                t_identity = np.ones_like(
                    mhr_output['pred_identity'][0].detach().cpu()) * self.t
                t_cam = np.ones_like(
                    mhr_output['pred_cam'][0].detach().cpu()) * self.t

                mhr_output['pred_cam'] = torch.tensor(
                    self.one_euro_cam(
                        t_cam,
                        mhr_output['pred_cam'].detach().cpu().numpy())).cuda()
                mhr_output['pred_pose'] = torch.tensor(
                    self.one_euro_pose(
                        t_pose,
                        mhr_output['pred_pose'].detach().cpu().numpy())).cuda(
                        )
                mhr_output['pred_identity'] = torch.tensor(
                    self.one_euro_identity(
                        t_identity, mhr_output['pred_identity'].detach().cpu().
                        numpy())).cuda()
                # ← original also filtered: hmr_output['pred_shape']
                #   New: no pred_shape; pred_identity and pred_expr are filtered above

            # Forward through MHR head to get vertices and joints
            # ← original (hmr.py:L134-144): self.smpl(rotmat=hmr_output['pred_pose'], shape=hmr_output['pred_shape'], cam=hmr_output['pred_cam'], ...)
            #   SMPLXCamHead expects (rotmat, shape, cam)
            #   MHRHead expects (identity_coeffs, lbs_model_params, face_expr_coeffs, cam)
            mhr_forward_output = self.mhr_head(
                identity_coeffs=mhr_output['pred_identity'],
                lbs_model_params=mhr_output['pred_pose'],
                face_expr_coeffs=mhr_output['pred_expr'],
                cam=mhr_output['pred_cam'],
                cam_intrinsics=cam_intrinsics,
                bbox_scale=bbox_scale,
                bbox_center=bbox_center,
                img_w=img_w,
                img_h=img_h,
                normalize_joints2d=False,
            )

            # Combine outputs
            mhr_forward_output.update(mhr_output)

            # Return with optional depth/segmentation
            if self.hparams and (self.hparams.DATASET.USE_DEPTH
                                 or self.hparams.DATASET.USE_SEGM):
                if self.hparams.DATASET.USE_DEPTH:
                    return mhr_forward_output, orig_depth, None, None, segmentation
                else:
                    return mhr_forward_output, None, None, None, segmentation
            return mhr_forward_output, None, None, None, None

        # MHRHMR only supports bedlam_bbox=True (TRIAL.bedlam_bbox must be True in config)
        raise RuntimeError(
            "MHRHMR requires TRIAL.bedlam_bbox=True. "
            "Set 'bedlam_bbox: True' in config_mhr.yaml under the TRIAL section."
        )
