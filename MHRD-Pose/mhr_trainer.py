"""
MHR Trainer Module for D-Pose MHR port.

MAPPED FROM: train/core/hmr_trainer.py (HMRTrainer — SMPL/SMPLX version)

This module provides a PyTorch Lightning module for training the MHR-based HMR model.
It replaces HMRTrainer and is completely independent of SMPL dependencies.

File correspondence:
  mhr_trainer.py:MHRTrainer  ←  train/core/hmr_trainer.py:HMRTrainer

Key differences from original HMRTrainer:
  - No smplx.SMPLX / smplx.SMPL module instances (replaced by mhr_model loaded from checkpoint)
  - No smplx2smpl face/vertex mapping tensor (MHR vertices are used directly)
  - No J_regressor buffer (SMPL joints derived from MHR vertices via barycentric interpolation)
  - Training GT: smplx(gt_betas, gt_pose) → barycentric interp of gt vertices → SMPL joints
  - Validation GT: unified MHR→SMPL mapping instead of dataset-specific branches (bedlam/rich/h36m/3dpw)
  - Validation_epoch_end → on_validation_epoch_end (PL v2.0 API: outputs arg removed, accumulate manually)
  - test_epoch_end → on_test_epoch_end (same PL v2.0 reason)
  - Renderer faces loaded from mhr_portable_dump instead of smplx.faces
  - DataLoader multiprocessing_context='spawn' added for CUDA safety in workers
"""

import os
import torch
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from mhr_constants import NUM_MHR_SKELETON_JOINTS, MHR_TO_SMPL_JOINT_INDICES
from dataset_wrapper import DatasetHMR
from train.utils.eval_utils import reconstruction_error
from train.utils.renderer import Renderer


class MHRTrainer(pl.LightningModule):
    """PyTorch Lightning module for MHR-based HMR training.

    This trainer wraps the MHR-based HMR model and provides:
    1. Training step with MHR loss computation
    2. Validation step with MPJPE/PA-MPJPE metrics
    3. MHR-specific forward pass without SMPL dependencies

    Unlike the original HMRTrainer, this module:
    - Uses MHR parameters directly (no SMPL→MHR conversion at runtime)
    - Computes losses on MHR parameters
    - Extracts joints from MHR skeleton state
    """

    def __init__(self, hparams):
        # MAPPED FROM: HMRTrainer.__init__ (train/core/hmr_trainer.py:20)
        super(MHRTrainer, self).__init__()

        self.hparams.update(hparams)

        # Import here to avoid circular imports
        from mhr_hmr import MHRHMR        # ← original: from ..models.hmr import HMR
        from mhr_losses import MHRLoss    # ← original: from ..losses.losses import HMRLoss
        from config import MHR_MODEL_PT

        # Load MHR model
        # ← original: self.smplx = smplx.SMPLX(...) and self.smpl = smplx.SMPL(...)
        #   SMPLX/SMPL model instances replaced by loading MHR model from checkpoint
        self.mhr_model = torch.load(MHR_MODEL_PT,
                                    map_location='cpu',
                                    weights_only=False)
        self.add_module('mhr_model', self.mhr_model)

        # Initialize MHR-based HMR model
        # ← original: self.model = HMR(...)  [line 24 of hmr_trainer.py]
        self.model = MHRHMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
            mhr_model=self.mhr_model,  # ← new param: passes MHR model to HMR wrapper
        )

        # MHR loss function
        # ← original: self.loss_fn = HMRLoss(hparams=self.hparams)
        self.loss_fn = MHRLoss(hparams=self.hparams)

        # Dataset
        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()
        self.val_ds = self.val_dataset()

        self.save_itr = 0
        self._val_step_outputs = [
        ]  # accumulated by validation_step; read by on_validation_epoch_end
        # ← original had no _val_step_outputs — PL v1.x passed `outputs` to validation_epoch_end
        #   PL v2.0 removed that arg, so we accumulate manually (like PL docs recommend)

        # ← original: self.smplx2smpl = pickle.load(...); self.smplx2smpl = torch.tensor(...).cuda()
        #   No longer needed — MHR vertices are already in the right space, no SMPLX→SMPL conversion

        # ← original: self.register_buffer('J_regressor', torch.from_numpy(...))
        #   No longer needed — SMPL joints derived from MHR vertices via barycentric interp in training/validation

        # Renderer for visualization
        # ← original: faces=self.smplx.faces  [line 53 of hmr_trainer.py]
        #   mhr_model is a TorchScript RecursiveScriptModule; .character is not
        #   accessible via __getattr__. Load faces from the portable dump instead.
        import sys as _sys
        _proj_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        if _proj_root not in _sys.path:
            _sys.path.insert(0, _proj_root)
        from load_mhr_portable import load_portable
        _portable = load_portable(
            os.path.join(_proj_root, 'mhr_portable_dump', 'mhr_dump_lod1.pt'))
        _faces = _portable['interesting']['character.mesh.faces']
        if isinstance(_faces, torch.Tensor):
            self.faces = _faces.cpu().numpy()
        else:
            self.faces = np.array(_faces)

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.faces,  # ← original: faces=self.smplx.faces
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )

        # Load pretrained weights from a checkpoint (e.g. original D-Pose SMPL ckpt)
        # ← original: this was done in train.py via TRAINING.RESUME + manual loading
        #   MHR: use TRAINING.PRETRAINED_LIT to point to a .ckpt file to load
        #   compatible weights from before training starts.
        self._load_pretrained_model()

        # H36M J_regressor for paper-comparable 14-joint validation metrics.
        #
        # Original (hmr_trainer.py:45-48):
        #   self.register_buffer('J_regressor',
        #       torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float())
        # The original D-Pose paper reports MPJPE/PA-MPJPE on 14 H36M joints,
        # computed by applying this [17, 6890] regressor to SMPL vertices and
        # then selecting the 14-joint subset via H36M_TO_J14.
        #
        # MHR validation uses SMPL vertices derived from barycentric interpolation
        # of MHR mesh vertices (mhr2smpl_mapping.npz), so the same regressor can
        # be applied here for an apples-to-apples comparison with the paper.
        # When the file is absent, validation falls back to SMPL-24 protocol
        # (different joint set and pelvis, numbers NOT comparable to the paper).
        _h36m_path = os.path.join(_proj_root, 'data/utils/J_regressor_h36m.npy')
        if os.path.isfile(_h36m_path):
            self.register_buffer(
                'J_regressor_h36m',
                torch.from_numpy(np.load(_h36m_path)).float())
            logger.info(
                f'Loaded H36M J_regressor from {_h36m_path} — '
                'validation will use H36M-14 protocol (paper-comparable).')
        else:
            self.J_regressor_h36m = None
            logger.warning(
                f'J_regressor_h36m.npy not found at {_h36m_path}. '
                'Validation will use SMPL-24 protocol — MPJPE/PA-MPJPE '
                'will NOT be directly comparable to the D-Pose paper. '
                'Provide data/utils/J_regressor_h36m.npy (shape [17, 6890]) '
                'to enable paper-comparable H36M-14 metrics.')

    def forward(self, x, bbox_center, bbox_scale, img_w, img_h, fl=None):
        # MAPPED FROM: HMRTrainer.forward (hmr_trainer.py:57)
        """Forward pass through MHR-based HMR model."""
        return self.model(x,
                          bbox_center=bbox_center,
                          bbox_scale=bbox_scale,
                          img_w=img_w,
                          img_h=img_h,
                          fl=fl)

    def on_train_epoch_start(self):
        """Reconfigure optimizer when unfreezing pretrained layers."""
        # Log the learning rate for every optimizer param group at epoch start.
        # With differential LR (head at base_lr, backbone at base_lr/10), two
        # groups exist before unfreeze and one after.  Logging both makes the
        # LR jump at UNFREEZE_EPOCH visible on TensorBoard without guessing from
        # epoch numbers alone.  LR is epoch-level (doesn't change within epoch),
        # so on_epoch=True / on_step=False is the right granularity.
        _opt = self.optimizers()
        if isinstance(_opt, list):
            _opt = _opt[0]
        if _opt is not None:
            for _i, _pg in enumerate(_opt.param_groups):
                self.log(f'train/lr_group_{_i}', float(_pg['lr']),
                         on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if self.current_epoch == self.hparams.TRAINING.UNFREEZE_EPOCH:
            # Log a 1.0 spike so the unfreeze boundary is visible on every
            # TensorBoard scalar chart without needing to compute it mentally
            # from epoch numbers.  All other epochs log 0.0 implicitly (absent).
            self.log('train/unfreeze_event', 1.0,
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
            if self.hparams.TRAINING.FREEZE_PRETRAINED:
                logger.info('=' * 60)
                logger.info('UNFREEZE — switching optimizer to uniform LR for all layers')
                logger.info('=' * 60)
                base_lr = self.hparams.OPTIMIZER.LR
                # Mutate the existing optimizer's param_groups in-place
                # so Lightning's references stay valid. Adam state is
                # keyed by tensor id, not index, so it transfers cleanly.
                opt = self.optimizers()
                if isinstance(opt, list):
                    opt = opt[0]
                for pg in opt.param_groups:
                    pg['lr'] = base_lr
                logger.info(f'  All params now use lr={base_lr:.1e}')
                logger.info('=' * 60)

    def on_train_start(self):
        """Probe whether the MHR TorchScript model supports autograd.

        Checks two output paths:
        1. vertices — used by loss_shape and loss_keypoints_3d (via barycentric
           interpolation into SMPL joints). This is the primary gradient path.
        2. skel_state — used for joints3d extraction (mhr_head.py:259-261).
           joints3d (127-joint raw MHR ordering) is NOT used in any loss —
           only joints3d_smpl (24-joint barycentric from vertices) is.
           If skel_state is non-differentiable, it has NO impact on training.
        """
        logger.info("=" * 60)
        logger.info("MHR GRAD PROBE — checking if MHR model is differentiable")
        mhr_head = getattr(self.model, 'mhr_head', None)
        if mhr_head is None:
            logger.warning("  mhr_head not found — skipping probe")
            logger.info("=" * 60)
            return

        device = next(self.model.parameters()).device
        from mhr_constants import NUM_LBS_MODEL_PARAMS, NUM_IDENTITY_BLENDSHAPES, NUM_FACE_EXPRESSION_BLENDSHAPES
        B = 2
        identity = torch.zeros(B, NUM_IDENTITY_BLENDSHAPES,  device=device, requires_grad=True)
        pose     = torch.zeros(B, NUM_LBS_MODEL_PARAMS,      device=device, requires_grad=True)
        expr     = torch.zeros(B, NUM_FACE_EXPRESSION_BLENDSHAPES, device=device, requires_grad=True)

        try:
            verts_cm, skel_state = mhr_head.mhr_model(
                identity_coeffs=identity,
                model_parameters=pose,
                face_expr_coeffs=expr,
                apply_correctives=True,
            )
            verts_m = verts_cm * 0.01
            logger.info(f"  verts_m: shape={tuple(verts_m.shape)}"
                        f"  requires_grad={verts_m.requires_grad}"
                        f"  grad_fn={type(verts_m.grad_fn).__name__ if verts_m.grad_fn else None}")

            # Check skel_state differentiability (joints3d extraction path).
            # joints3d = skel_state[:, :, :3] is used for visualization and
            # as a fallback in loss_keypoints_3d, but the primary path uses
            # joints3d_smpl (barycentric from vertices) which IS differentiable.
            logger.info(f"  skel_state: shape={tuple(skel_state.shape)}"
                        f"  requires_grad={skel_state.requires_grad}"
                        f"  grad_fn={type(skel_state.grad_fn).__name__ if skel_state.grad_fn else None}")
            joints3d_probe = skel_state[:, :, :3] * 0.01
            logger.info(f"  joints3d (skel_state[:,:3]*cm_to_m): shape={tuple(joints3d_probe.shape)}"
                        f"  requires_grad={joints3d_probe.requires_grad}"
                        f"  grad_fn={type(joints3d_probe.grad_fn).__name__ if joints3d_probe.grad_fn else None}")

            if verts_m.requires_grad:
                verts_m.sum().backward()
                logger.info(f"  identity.grad: {'OK — grads reach identity_coeffs' if identity.grad is not None else 'NONE — dead gradient path!'}")
                logger.info(f"  pose.grad:     {'OK — grads reach lbs_model_params' if pose.grad is not None else 'NONE — dead gradient path!'}")
                logger.info(f"  expr.grad:     {'OK — grads reach face_expr_coeffs' if expr.grad is not None else 'NONE — dead gradient path!'}")
            else:
                logger.warning("  verts_m.requires_grad=False — MHR model does NOT support autograd")
                logger.warning("  loss_keypoints_3d and loss_shape provide ZERO gradient to the network")

            if not skel_state.requires_grad:
                logger.info("  skel_state is NOT differentiable — this is OK because"
                            " joints3d (127-joint raw MHR ordering) is not used in"
                            " any loss. The primary path uses joints3d_smpl computed"
                            " via barycentric interpolation of vertices, which IS"
                            " differentiable (see verts_m check above).")

        except Exception as e:
            logger.error(f"  MHR grad probe raised: {e}")
        logger.info("=" * 60)

    def training_step(self, batch, batch_nb, dataloader_nb=0):
        # MAPPED FROM: HMRTrainer.training_step (hmr_trainer.py:60)
        #
        # Key difference in GT computation:
        #   ORIGINAL (hmr_trainer.py:76-82): self.smplx(betas=gt_betas, body_pose=..., global_orient=...)
        #     → runs SMPLX model forward to get gt vertices and joints
        #   MHR (below): uses barycentric interpolation of GT MHR vertices
        #     → _smpl_tri_vids + _smpl_baryc map MHR→SMPL vertices
        #     → _smpl_J_reg projects SMPL vertices→joints (same J_regressor as original)
        #     → No need for smplx module at all
        #
        # Original also extracted gt_betas, gt_pose from batch — not used here
        # (MHR has no pose/betas params; skeleton state is the output)
        """Training step with MHR loss computation.

        Args:
            batch: Batch of training data with MHR ground truth
            batch_nb: Batch index
            dataloader_nb: Dataloader index

        Returns:
            Dictionary with loss
        """
        # GT data
        images = batch['img']
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]
        fl = batch['focal_length']
        batch_size = batch['img'].shape[0]

        # Forward pass
        pred, depth, _, _, part_segms = self(images,
                                             bbox_center=bbox_center,
                                             bbox_scale=bbox_scale,
                                             img_w=img_w,
                                             img_h=img_h,
                                             fl=fl)
        pred['depth'] = depth
        pred['part_segms'] = part_segms

        # --- Prediction statistics (logged every step) ---
        # Three common silent failure modes that MPJPE won't reveal until
        # several epochs in:
        #
        #   cam_scale_min < 0  →  at least one sample in the batch has an
        #     inverted camera (person "behind" the image plane). The
        #     exp(-cam*10)^2 camera loss clamps to 10.0 but does not
        #     prevent negative scale from persisting if the pose gradient
        #     is stronger.  Visible here before it shows up as high MPJPE.
        #
        #   cam_scale_std → 0  →  camera branch collapsed; all samples in
        #     the batch receive the same scale regardless of bounding-box
        #     size.  Typically caused by the loss_cam term dominating and
        #     driving all predictions toward the mean.
        #
        #   pose_std → 0  →  regressor mode collapse: all samples get
        #     nearly identical lbs_model_params.  Happens when the pose-
        #     regression loss overwhelms the image-feature signal early in
        #     training (before unfreeze), pushing params toward the GT mean
        #     without the backbone providing discriminative features.
        #
        #   identity_std → 0  →  shape branch not learning.  Common when
        #     identity_coeffs are all-zero in the precomputed dataset
        #     (preconvert_to_mhr.py failed to extract blendshape coeffs)
        #     because criterion(zeros, zeros) = 0 gives no gradient.
        with torch.no_grad():
            _cam = pred['pred_cam'][:, 0]  # scale only; tx/ty are unbounded
            self.log('pred/cam_scale_mean', _cam.mean(),
                     on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('pred/cam_scale_std',  _cam.std(),
                     on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('pred/cam_scale_min',  _cam.min(),
                     on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('pred/pose_std', pred['pred_pose'].std(),
                     on_step=True, on_epoch=False, logger=True, sync_dist=True)
            self.log('pred/identity_std', pred['pred_identity'].std(),
                     on_step=True, on_epoch=False, logger=True, sync_dist=True)

        # --- GT key availability (logged every step) ---
        # mhr_losses.py fetches every GT tensor with .get(), so a missing key
        # silently returns 0 loss and that parameter branch receives no
        # gradient.  A dataset caching bug, wrong file path, or failed
        # precomputation would manifest here as a 1.0 spike rather than as
        # a mysteriously stalled loss component.
        # Log as binary 0/1 per step; TensorBoard's smoothing makes even
        # occasional missing batches visible as a non-zero mean.
        for _gt_key in ('identity_coeffs', 'face_expr_coeffs',
                        'lbs_model_params', 'vertices'):
            self.log(f'data/gt_{_gt_key}_missing',
                     float(batch.get(_gt_key) is None),
                     on_step=True, on_epoch=False, logger=True, sync_dist=True)

        # Compute GT SMPL-ordered joints from GT MHR vertices.
        #
        # GT source: batch['vertices'] comes from dataset_wrapper.py, which converts
        # SMPL→MHR via SMPLX forward + Conversion + MHR forward. These are MHR mesh
        # vertices (18439 vertices) in meters, stored in camera space (transl=0).
        #
        # Pipeline: GT MHR vertices → barycentric interp → SMPL vertices (6890)
        #           → SMPL J_regressor → SMPL joints (24 joints, SMPL ordering).
        #
        # This matches exactly how validation computes GT joints (validation_step
        # lines 370-382), so training and validation use the same GT source.
        # The barycentric mapping (mhr2smpl_mapping.npz) and J_regressor (SMPL_NEUTRAL.pkl)
        # are buffers in mhr_head, shared between training and validation.
        #
        # torch.no_grad is essential — GT must be a constant, not part of the graph.
        # Gradients flow only through pred['joints3d_smpl'], which is computed via
        # the same barycentric mapping from pred['vertices'] (mhr_head.py:247-255).
        gt_verts = batch.get('vertices')
        mhr_head = getattr(self.model, 'mhr_head', None)
        if gt_verts is not None and mhr_head is not None and 'joints3d_smpl' in pred:
            with torch.no_grad():
                tv = mhr_head._smpl_tri_vids  # [6890, 3]
                w = mhr_head._smpl_baryc  # [6890, 3]
                gv0 = gt_verts[:, tv[:, 0]]
                gv1 = gt_verts[:, tv[:, 1]]
                gv2 = gt_verts[:, tv[:, 2]]
                gt_smpl_v = (w[:, 0].unsqueeze(0).unsqueeze(-1) * gv0 +
                             w[:, 1].unsqueeze(0).unsqueeze(-1) * gv1 +
                             w[:, 2].unsqueeze(0).unsqueeze(-1) * gv2)
                batch['joints3d'] = torch.einsum('jv,bvd->bjd',
                                                 mhr_head._smpl_J_reg,
                                                 gt_smpl_v)  # [B, 24, 3]

        # Compute loss
        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)

        # --- NaN/Inf guard ---
        # If the MHR forward produces degenerate geometry (e.g. extreme
        # lbs_model_params) or a loss weight is misconfigured, loss can go
        # NaN/Inf.  Lightning detects NaN and skips the optimizer step, but
        # it does NOT zero the gradients first — stale .grad values from the
        # previous step persist, and Adam's running-mean buffers accumulate
        # NaN permanently, making recovery impossible without resetting state.
        # Returning a zero-loss tensor (with a live grad_fn via pred_cam) lets
        # the backward run cleanly (grad = 0 everywhere) and leaves optimizer
        # state intact.  We also log which component caused the NaN so the
        # root cause is immediately traceable in TensorBoard.
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f'NaN/Inf loss at epoch {self.current_epoch} batch {batch_nb}! '
                f'Skipping backward to protect optimizer state.')
            for _k, _v in loss_dict.items():
                if torch.isnan(_v) or torch.isinf(_v):
                    logger.error(f'  BAD  {_k} = {_v.item()}')
                else:
                    logger.error(f'  ok   {_k} = {_v.item():.6f}')
            self.log('train/nan_loss_event', 1.0,
                     on_step=True, on_epoch=False, logger=True, sync_dist=True)
            # Zero via pred_cam.mean() keeps a live grad_fn so Lightning's
            # backward hook does not raise "element 0 of tensors does not
            # require grad" while still producing zero updates.
            return {'loss': pred['pred_cam'].mean() * 0.0}

        # Log losses
        self.log('train_loss', loss, logger=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(k, v, logger=True, sync_dist=True)

        # --- Loss contribution ratios ---
        # Absolute loss values are hard to interpret as training progresses
        # because the overall scale changes as components converge.  Ratios
        # show the *relative weight* of each term, making imbalances obvious:
        # e.g. loss_shape at 80% of total means vertex reconstruction is
        # dominating and pose/keypoint signals are being drowned out.
        # Summing all ratios (excluding 'loss/loss') should give ~1.0 modulo
        # the loss_weight scaling applied inside MHRLoss.forward.
        with torch.no_grad():
            _total_abs = loss.abs().clamp(min=1e-8)
            for _k, _v in loss_dict.items():
                if _k != 'loss/loss':
                    self.log(f'ratio/{_k.split("/")[-1]}',
                             _v.abs() / _total_abs,
                             on_step=True, on_epoch=False,
                             logger=True, sync_dist=True)

        if batch_nb == 0 and self.current_epoch == 0:
            logger.info("=" * 60)
            logger.info("E0 B0 LOSS DEBUG")
            logger.info(f"  Config loss_weight={self.loss_fn.loss_weight}"
                        f"  joint={self.loss_fn.joint_loss_weight}"
                        f"  shape={self.loss_fn.shape_loss_weight}"
                        f"  kp2d={self.loss_fn.keypoint_loss_weight_2d}"
                        f"  pose={self.loss_fn.pose_loss_weight}"
                        f"  identity={self.loss_fn.identity_loss_weight}"
                        f"  expr={self.loss_fn.expr_loss_weight}")
            for k, v in loss_dict.items():
                logger.info(f"  {k} = {v.item():.6f}")
            logger.info("=" * 60)

            # Backward to inspect gradients, then let Lightning run
            # its own backward. We do this before Lightning's backward
            # so we can inspect; Lightning will zero grads first anyway.
            # Actually — calling .backward() twice would accumulate.
            # Instead, just check pred tensor grad status pre-backward.
            logger.info("--- E0 B0 PRED TENSOR GRAD CHECK ---")
            for key in ('joints3d', 'joints3d_smpl', 'vertices'):
                t = pred.get(key)
                if t is not None:
                    logger.info(
                        f"  {key}: shape={tuple(t.shape)}"
                        f"  requires_grad={t.requires_grad}"
                        f"  grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}"
                    )
            logger.info("--- END E0 B0 CHECK ---")

            # --- Barycentric mapping sanity (E0 B0, real data) ---
            # PLAN.md §Critical Finding: the 3D keypoint loss uses GT SMPL joints
            # derived from GT MHR vertices via barycentric interpolation
            # (mhr2smpl_mapping.npz → SMPL J_regressor).  If that mapping is
            # inaccurate, the model trains toward systematically wrong joint
            # targets and PA-MPJPE will plateau at a floor set by the mapping
            # error, not by the model's pose estimation capability.
            #
            # This check runs once at E0 B0 on the first real training batch,
            # where batch['joints3d'] has just been populated by the GT computation
            # block above.  It checks three proxies for mapping quality:
            #   1. Position range: joints should be within ±1.5 m of the pelvis
            #      for any plausible human pose.  Larger values mean the mapping
            #      returned positions in centimetres (forgot * cm_to_m) or the
            #      barycentric interpolation landed on completely wrong triangles.
            #   2. Mean dist from pelvis: should be 0.3–0.6 m for adult body joints.
            #      Values < 0.05 m mean joints are all at the origin (mapping
            #      returned zeros, or GT vertices were T-pose with no displacement).
            #   3. Per-joint spread: std across joints per sample reflects
            #      whether the mapping is producing a full-body spread or a degenerate
            #      configuration where multiple joints collapse to the same point.
            _gt_j3d = batch.get('joints3d')
            if (_gt_j3d is not None and _gt_j3d.ndim == 3
                    and _gt_j3d.shape[-1] == 3 and _gt_j3d.shape[1] >= 3):
                with torch.no_grad():
                    _pel = (_gt_j3d[:, [1]] + _gt_j3d[:, [2]]) / 2.0
                    _centered = _gt_j3d - _pel
                    _dist = _centered.norm(dim=-1)  # [B, J] dist from pelvis
                logger.info("--- BARYCENTRIC MAPPING SANITY (E0 B0) ---")
                logger.info(f"  GT joints3d: shape={tuple(_gt_j3d.shape)}")
                logger.info(f"  Position range : [{_gt_j3d.min():.4f}, "
                            f"{_gt_j3d.max():.4f}] m")
                logger.info(f"  Mean dist/pelvis: {_dist.mean():.4f} m "
                            f"(expected 0.30–0.60 m)")
                logger.info(f"  Max  dist/pelvis: {_dist.max():.4f} m "
                            f"(expected < 1.50 m)")
                logger.info(f"  Per-joint std   : "
                            f"{_centered.std(dim=1).mean():.4f} m")
                if _dist.max() > 2.0:
                    logger.warning(
                        "  *** WARN: joints > 2 m from pelvis — check "
                        "mhr2smpl_mapping.npz units (should be metres, not cm)")
                elif _dist.mean() < 0.05:
                    logger.warning(
                        "  *** WARN: joints near origin — barycentric "
                        "mapping may have returned zeros or GT vertices "
                        "are all T-pose (no shape/pose displacement)")
                else:
                    logger.info("  Mapping looks reasonable.")
                logger.info("--- END BARYCENTRIC SANITY ---")
            else:
                logger.warning(
                    "  Barycentric sanity skipped: batch['joints3d'] not "
                    "available at E0 B0 — GT joint computation may have "
                    "failed (check mhr_head._smpl_tri_vids / _smpl_baryc "
                    "buffers and GT vertices in batch)")

        if batch_nb % 200 == 0:
            loss_summary = {
                k.split('/')[-1]: f'{v.item():.4f}'
                for k, v in loss_dict.items()
            }
            logger.info(
                f'Epoch {self.current_epoch} batch {batch_nb}: {loss_summary}')

        return {'loss': loss}

    def on_after_backward(self):
        """Inspect gradient norms after each backward pass, periodically.

        Runs at E0 B0 (initial connectivity check) and every 500 steps
        thereafter (ongoing health monitoring).

        Why periodic gradient norms are useful beyond the initial check:
          - After UNFREEZE_EPOCH, the backbone starts receiving gradients for
            the first time.  A sudden spike in backbone grad norm (> 10× its
            first value) means the learning rate is too high for the unfrozen
            layers and features will be destroyed.  A drop to near-zero means
            the backbone is not contributing to the loss.
          - If head (regressor) grad norm shrinks to near-zero after many epochs,
            the MHRRegressor has saturated — lbs_model_params converged to a
            fixed point and the model stopped learning pose.  This is expected
            near the end of training but premature early on.
          - mhr_head should always show zero grad: its TorchScript MHR model
            parameters are not nn.Parameters (they are TorchScript internals),
            and the barycentric buffers (_smpl_tri_vids, _smpl_baryc, etc.) are
            registered buffers with requires_grad=False.  Any non-zero value here
            means the model structure changed unexpectedly.

        Note: global_step == 1 during the first batch's on_after_backward
        because Lightning increments global_step after the optimizer step,
        which runs after on_after_backward.  So step 0 → global_step=1 here.
        """
        _is_first    = (self.current_epoch == 0 and self.global_step == 1)
        # Fire every 500 optimizer steps after the initial check.
        # 500 steps ≈ every ~22 min at 1355 steps/epoch; enough resolution
        # to catch a gradient collapse without flooding the log.
        _is_periodic = (self.global_step % 500 == 0 and self.global_step > 1)
        if not (_is_first or _is_periodic):
            return

        _label = 'E0 B0' if _is_first else f'step {self.global_step}'
        logger.info("=" * 60)
        logger.info(f"POST-BACKWARD GRADIENT NORM CHECK ({_label})")
        groups = {'backbone': [], 'head (regressor)': [], 'mhr_head': [], 'other': []}
        for name, param in self.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm().item()
                bare = name.replace('model.model.', '').replace('model.', '')
                if bare.startswith('backbone.'):
                    groups['backbone'].append((bare, norm))
                elif bare.startswith('head.'):
                    groups['head (regressor)'].append((bare, norm))
                elif bare.startswith('mhr_head.'):
                    groups['mhr_head'].append((bare, norm))
                else:
                    groups['other'].append((bare, norm))
        for group, params in groups.items():
            if params:
                total = sum(n for _, n in params)
                nonzero = sum(1 for _, n in params if n > 0)
                logger.info(f"  {group}: {nonzero}/{len(params)} params "
                            f"have non-zero grad, total norm={total:.6f}")
                for name, norm in sorted(params, key=lambda x: -x[1])[:3]:
                    logger.info(f"    top: {name} norm={norm:.6f}")
                # Log total group norm to TensorBoard for trend analysis.
                # Key sanitised: spaces→underscore, parens removed.
                _tb_key = (group.replace(' ', '_')
                                .replace('(', '').replace(')', ''))
                self.log(f'grad_norm/{_tb_key}', total,
                         on_step=True, on_epoch=False,
                         logger=True, sync_dist=True)
            else:
                logger.warning(f"  {group}: NO parameters found")
        logger.info("=" * 60)

    def validation_step(self,
                        batch,
                        batch_nb,
                        dataloader_nb=0,
                        vis=False,
                        save=True,
                        mesh_save_dir=None):
        # MAPPED FROM: HMRTrainer.validation_step (hmr_trainer.py:93)
        #
        # Key difference — original had dataset-specific branches (hmr_trainer.py:116-173):
        #   'bedlam':    smplx(gt) → gt vertices/joints, pred[:24]
        #   'rich':      batch['vertices'], smplx2smpl matrix convert pred vertices, J_regressor
        #   'h36m':      batch['vertices'], J_regressor, joint_mapper_h36m, smplx2smpl
        #   else (3dpw): batch['vertices'], J_regressor, joint_mapper_h36m, smplx2smpl
        #   All used self.smplx2smpl (SMPLX→SMPL vertex mapping matrix) and self.J_regressor
        #
        # MHR version: unified approach using barycentric MHR→SMPL vertex mapping
        #   (_smpl_tri_vids, _smpl_baryc, _smpl_J_reg) — no dataset branches needed
        #   because MHR vertices are already in a common coordinate space.
        #
        # Also: PL v2.0 — this returns a dict that gets accumulated in _val_step_outputs
        #   (original had no return accumulation; validation_epoch_end received `outputs`)
        """Validation step computing MPJPE/PA-MPJPE metrics.

        Args:
            batch: Batch of validation data
            batch_nb: Batch index
            dataloader_nb: Dataloader index
            vis: Whether to visualize
            save: Whether to save results
            mesh_save_dir: Directory to save meshes

        Returns:
            Dictionary with metrics per dataset
        """
        images = batch['img']
        batch_size = images.shape[0]
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        dataset_names = batch['dataset_name']
        dataset_index = batch['dataset_index'].detach().cpu().numpy()
        val_dataset_names = self.hparams.DATASET.VAL_DS.split('_')
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]

        # Forward pass
        pred, depth, _, _, part_segms = self(images,
                                             bbox_center=bbox_center,
                                             bbox_scale=bbox_scale,
                                             img_w=img_w,
                                             img_h=img_h)
        pred['part_segms'] = part_segms
        pred['depth'] = depth
        pred_cam_vertices = pred['vertices']

        # Get ground truth joints and vertices
        gt_cam_vertices = batch.get('vertices', None)

        # Validation joint protocol.
        #
        # Two outer paths:
        #   Primary: barycentric MHR→SMPL vertices are available (normal case).
        #            Sub-protocol selected by J_regressor_h36m availability:
        #     H36M-14: apply J_regressor_h36m [17,6890] to SMPL verts → 17 H36M
        #              joints → pelvis at H36M index 0 → select 14 via H36M_TO_J14.
        #              Matches original D-Pose paper protocol exactly.
        #              ← hmr_trainer.py:104,161-173 (3DPW/else branch)
        #     SMPL-24: use barycentric joints directly; pelvis = avg(Lhip[1], Rhip[2]).
        #              Used when J_regressor_h36m.npy is absent. Numbers are NOT
        #              comparable to the D-Pose paper.
        #   Fallback: raw MHR skel_state → SMPL index mapping (no SMPL vertices).
        #             Same SMPL-24 pelvis convention as the non-H36M primary path.
        #
        # H36M_TO_J14 maps into the 17-joint H36M regressor output (NOT SMPL-24).
        # Joint 0 of the 17-joint set is the pelvis; it is not included in J14,
        # it is used only for centering — matching hmr_trainer.py:162:
        #   gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
        # ← original: train/core/constants.py:H36M_TO_J17[:14]
        # Sentinel: GT SMPL-24 joints for the validation loss breakdown below.
        # Set inside the barycentric branch where gt_smpl_joints is computed.
        # Remains None in the MHR-skel fallback branch (barycentric path
        # unavailable), in which case the val-loss breakdown is skipped.
        _gt_smpl_joints_for_loss = None

        H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

        pred_smpl = pred.get('joints3d_smpl')
        mhr_head = getattr(self.model, 'mhr_head', None)
        gt_verts_val = batch.get('vertices')
        if pred_smpl is not None and mhr_head is not None and gt_verts_val is not None:
            tv = mhr_head._smpl_tri_vids  # [6890, 3]
            w  = mhr_head._smpl_baryc     # [6890, 3]

            # GT SMPL vertices via barycentric interpolation of GT MHR vertices.
            # Shared by both sub-protocols below.
            gv0 = gt_verts_val[:, tv[:, 0]]
            gv1 = gt_verts_val[:, tv[:, 1]]
            gv2 = gt_verts_val[:, tv[:, 2]]
            gt_smpl_v = (w[:, 0].unsqueeze(0).unsqueeze(-1) * gv0 +
                         w[:, 1].unsqueeze(0).unsqueeze(-1) * gv1 +
                         w[:, 2].unsqueeze(0).unsqueeze(-1) * gv2)  # [B, 6890, 3]
            gt_smpl_joints = torch.einsum(
                'jv,bvd->bjd', mhr_head._smpl_J_reg, gt_smpl_v)  # [B, 24, 3]
            # Capture for the validation loss breakdown; detach so the loss
            # function sees a constant target (same intent as training_step's
            # torch.no_grad() block that populates batch['joints3d']).
            _gt_smpl_joints_for_loss = gt_smpl_joints.detach()

            if self.J_regressor_h36m is not None:
                # --- H36M-14 protocol (paper-comparable) ---
                #
                # Pred SMPL vertices: same barycentric mapping applied to pred
                # MHR vertices (pred['vertices'] is [B, 18439, 3]).
                # We reuse the same tv/w buffers from mhr_head.
                pv = pred['vertices']
                pred_smpl_v = (w[:, 0].unsqueeze(0).unsqueeze(-1) * pv[:, tv[:, 0]] +
                               w[:, 1].unsqueeze(0).unsqueeze(-1) * pv[:, tv[:, 1]] +
                               w[:, 2].unsqueeze(0).unsqueeze(-1) * pv[:, tv[:, 2]])  # [B, 6890, 3]

                # Apply H36M J_regressor [17, 6890] → 17 H36M joints each.
                # ← original: torch.matmul(J_regressor_batch_smpl, cam_vertices)
                #   where J_regressor_batch_smpl = self.J_regressor.unsqueeze(0).expand(B,-1,-1)
                #   MHR: equivalent einsum, no explicit batch expansion needed.
                gt_h36m   = torch.einsum('jv,bvd->bjd', self.J_regressor_h36m, gt_smpl_v)    # [B, 17, 3]
                pred_h36m = torch.einsum('jv,bvd->bjd', self.J_regressor_h36m, pred_smpl_v)  # [B, 17, 3]

                # Pelvis = H36M joint 0 (not included in J14, used only for centering).
                # ← original hmr_trainer.py:162: gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_pelvis   = gt_h36m[:,   [0], :]  # [B, 1, 3]
                pred_pelvis = pred_h36m[:, [0], :]  # [B, 1, 3]

                gt_keypoints_3d   = gt_h36m[:,   H36M_TO_J14, :] - gt_pelvis    # [B, 14, 3]
                pred_keypoints_3d = pred_h36m[:, H36M_TO_J14, :] - pred_pelvis  # [B, 14, 3]

                # Pelvis-center mesh vertices for PVE using the same H36M pelvis,
                # so that vertex error and joint error share a common root frame.
                pred_cam_vertices = pred_cam_vertices - pred_pelvis
                if gt_cam_vertices is not None:
                    gt_cam_vertices = gt_cam_vertices - gt_pelvis
            else:
                # --- SMPL-24 fallback (numbers not comparable to paper) ---
                #
                # Use barycentric SMPL joints directly; pelvis = average of
                # left hip (SMPL joint 1) and right hip (SMPL joint 2).
                pred_keypoints_3d = pred_smpl[:, :24]
                gt_keypoints_3d   = gt_smpl_joints[:, :24]

                gt_pelvis   = (gt_keypoints_3d[:,   [1]] + gt_keypoints_3d[:,   [2]]) / 2.0
                pred_pelvis = (pred_keypoints_3d[:, [1]] + pred_keypoints_3d[:, [2]]) / 2.0
                pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
                pred_cam_vertices = pred_cam_vertices - pred_pelvis
                gt_keypoints_3d   = gt_keypoints_3d - gt_pelvis
                if gt_cam_vertices is not None:
                    gt_cam_vertices = gt_cam_vertices - gt_pelvis
        else:
            # Fallback: map MHR skel_state's 127 joints to SMPL ordering via
            # MHR_TO_SMPL_JOINT_INDICES. Slicing the raw [:24] does NOT work —
            # the first 24 MHR joints are foot/twist procedurals, not body joints.
            idx = torch.as_tensor(
                MHR_TO_SMPL_JOINT_INDICES,
                dtype=torch.long,
                device=pred['joints3d'].device,
            )
            pred_keypoints_3d = pred['joints3d'].index_select(1, idx)
            gt_joints_mhr = batch.get('joints3d_mhr',
                                      batch.get('joints3d', None))
            if gt_joints_mhr is not None and gt_joints_mhr.ndim == 3 \
                    and gt_joints_mhr.shape[1] >= max(MHR_TO_SMPL_JOINT_INDICES) + 1:
                gt_keypoints_3d = gt_joints_mhr.index_select(
                    1, idx.to(gt_joints_mhr.device))
            else:
                # Last resort: original-dataset SMPL joints (already SMPL-ordered).
                gt_keypoints_3d = batch.get('joints',
                                            None)[:, :NUM_MHR_SKELETON_JOINTS]

            # Pelvis-center: SMPL-24 convention (avg of L/R hip at indices 1, 2).
            gt_pelvis   = (gt_keypoints_3d[:,   [1]] + gt_keypoints_3d[:,   [2]]) / 2.0
            pred_pelvis = (pred_keypoints_3d[:, [1]] + pred_keypoints_3d[:, [2]]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            gt_keypoints_3d   = gt_keypoints_3d - gt_pelvis
            if gt_cam_vertices is not None:
                gt_cam_vertices = gt_cam_vertices - gt_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d -
                             gt_keypoints_3d)**2).sum(dim=-1)).cpu().numpy()
        if gt_cam_vertices is not None:
            error_verts = torch.sqrt(
                ((pred_cam_vertices -
                  gt_cam_vertices)**2).sum(dim=-1)).cpu().numpy()
        else:
            error_verts = np.zeros((batch_size, pred_cam_vertices.shape[1]))

        # Reconstruction error (PA-MPJPE)
        r_error, _ = reconstruction_error(pred_keypoints_3d.cpu().numpy(),
                                          gt_keypoints_3d.cpu().numpy(),
                                          reduction=None)
        val_mpjpe = error.mean(-1)
        val_pampjpe = r_error.mean(-1)
        val_pve = error_verts.mean(-1)

        # --- Validation loss breakdown ---
        # MPJPE/PA-MPJPE/PVE tell you *how wrong* the predictions are, but not
        # *which loss component* is responsible.  Running the loss function on
        # the validation batch surfaces this:
        #   val/loss_keypoints_3d stuck while val/loss_keypoints drops  →
        #     the barycentric path is broken or the 3D GT is wrong
        #   val/loss_shape high while val/loss_keypoints_3d low  →
        #     pose is roughly right but vertex positions deviate (e.g. scale
        #     in wrong units, or MHR identity_coeffs not matching GT shape)
        #   val/loss_regr_pose not decreasing  →
        #     GT lbs_model_params are missing or the pose branch is saturated
        #
        # batch['joints3d'] must be set for the primary keypoint-3d path; we
        # use _gt_smpl_joints_for_loss captured from the barycentric block
        # above.  If that is None (MHR-skel fallback was taken), the loss
        # function will try the MHR-skel fallback internally.
        #
        # Wrapped in try/except so a monitoring failure never crashes validation.
        try:
            if _gt_smpl_joints_for_loss is not None:
                batch['joints3d'] = _gt_smpl_joints_for_loss
            _val_loss, _val_loss_dict = self.loss_fn(pred=pred, gt=batch)
            for _k, _v in _val_loss_dict.items():
                self.log(f'val/{_k.split("/")[-1]}', _v.detach(),
                         on_step=False, on_epoch=True,
                         logger=True, sync_dist=True)
        except Exception as _e:
            logger.warning(
                f'val loss breakdown failed at batch {batch_nb}: {_e}')

        loss_dict = {}

        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset
            ds_idx = val_dataset_names.index(ds.dataset)
            idxs = np.where(dataset_index == ds_idx)
            loss_dict[ds_name + '_mpjpe'] = list(val_mpjpe[idxs])
            loss_dict[ds_name + '_pampjpe'] = list(val_pampjpe[idxs])
            loss_dict[ds_name + '_pve'] = list(val_pve[idxs])

        # Accumulate for on_validation_epoch_end (PL v2.0+ no longer passes outputs)
        self._val_step_outputs.append(loss_dict)
        return loss_dict

    def on_validation_epoch_end(self):
        # MAPPED FROM: HMRTrainer.validation_epoch_end (hmr_trainer.py:201)
        #
        # PL v2.0 change: original received `outputs` as arg, now reads self._val_step_outputs
        # (populated by validation_step's accumulation at line 303)
        """Log validation metrics at end of epoch (PL v2.0+: no outputs arg)."""
        outputs = self._val_step_outputs
        self._val_step_outputs = []  # clear for next epoch

        logger.info(f'***** Epoch {self.current_epoch} *****')
        val_log = {}

        if len(self.val_ds) > 1:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000 * np.hstack(
                    np.array([x[ds_name + '_mpjpe'] for x in outputs])).mean()
                pampjpe = 1000 * np.hstack(
                    np.array([x[ds_name + '_pampjpe'] for x in outputs])).mean()
                pve = 1000 * np.hstack(
                    np.array([x[ds_name + '_pve'] for x in outputs])).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_PVE: ' + str(pve))
                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_pve'] = pve
        else:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000 * np.hstack(
                    np.array([x[ds_name + '_mpjpe'] for x in outputs])).mean()
                pampjpe = 1000 * np.hstack(
                    np.array([x[ds_name + '_pampjpe']
                              for x in outputs])).mean()
                pve = 1000 * np.hstack(
                    np.array([x[ds_name + '_pve'] for x in outputs])).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_PVE: ' + str(pve))

                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_pve'] = pve

        self.log('val_loss',
                 val_log[self.val_ds[0].dataset + '_val_pampjpe'],
                 logger=True,
                 sync_dist=True)
        self.log('val_loss_mpjpe',
                 val_log[self.val_ds[0].dataset + '_val_mpjpe'],
                 logger=True,
                 sync_dist=True)
        for k, v in val_log.items():
            self.log(k, v, logger=True, sync_dist=True)

    def test_step(self, batch, batch_nb, dataloader_nb=0):
        # MAPPED FROM: HMRTrainer.test_step (hmr_trainer.py:241)
        return self.validation_step(batch, batch_nb, dataloader_nb)

    def on_test_epoch_end(self):
        # MAPPED FROM: HMRTrainer.test_epoch_end (hmr_trainer.py:244)
        # PL v2.0: test_epoch_end(outputs) → on_test_epoch_end (no outputs arg)
        return self.on_validation_epoch_end()

    def _load_pretrained_model(self):
        """Load compatible weights from a pretrained checkpoint.

        The checkpoint stores keys from the original HMR model (e.g.
        ``model.backbone.conv1.weight``).  The MHR trainer is a
        LightningModule whose ``self.state_dict()`` keys are prefixed with
        ``model.`` (the nested HMR submodule), so we must align the two
        key namespaces before matching.

        Config key: ``TRAINING.PRETRAINED_LIT`` — path to a .ckpt file.
        Set to ``null`` (default) to skip.
        """
        pretrained_path = self.hparams.TRAINING.PRETRAINED_LIT
        if not pretrained_path:
            logger.info('PRETRAINED_LIT is not set — skipping pretrained weight loading.')
            return

        logger.info('=' * 60)
        logger.info('PRETRAINED WEIGHT LOADING')
        logger.info(f'  Checkpoint: {pretrained_path}')
        if not os.path.isfile(pretrained_path):
            logger.warning(f'  File not found — skipping.')
            logger.info('=' * 60)
            return

        logger.info('  Loading checkpoint...')
        ckpt = torch.load(pretrained_path, weights_only=False)
        raw_sd = ckpt.get('state_dict', ckpt)
        if not isinstance(raw_sd, dict):
            logger.warning('  Checkpoint has no "state_dict" key — skipping.')
            logger.info('=' * 60)
            return

        ckpt_size_mb = sum(v.numel() * v.element_size() for v in raw_sd.values()) / 1024 / 1024
        logger.info(f'  Checkpoint loaded: {len(raw_sd)} params, {ckpt_size_mb:.1f} MB')

        # --- MHR trainer state dict (LightningModule) -----------------------
        # self.state_dict() keys look like:
        #   model.backbone.conv1.weight
        #   model.head.fc1.weight
        #   mhr_head.mhr_model.xxx
        #   loss_fn.xxx
        # We only want to load into the nested ``self.model`` (MHRHMR).
        full_sd = self.state_dict()
        model_sd = self.model.state_dict()  # keys: backbone.conv1.weight, etc.

        # Build a lookup: bare key -> full key so we can write back
        bare_to_full = {}
        for fk, fv in full_sd.items():
            if fk.startswith('model.'):
                bare = fk[len('model.'):]
                bare_to_full[bare] = fk
        logger.info(f'  MHR model has {len(model_sd)} parameters')

        # --- align checkpoint keys to bare names ----------------------------
        # The checkpoint may store keys as:
        #   model.backbone.conv1.weight  → strip to backbone.conv1.weight
        # or bare:
        #   backbone.conv1.weight
        ckpt_bare = {}
        stripped = 0
        for k, v in raw_sd.items():
            # Strip 'model.' prefix if present
            if k.startswith('model.'):
                bare = k[len('model.'):]
                stripped += 1
            else:
                bare = k
            ckpt_bare[bare] = v
        logger.info(f'  Stripped "model." prefix from {stripped} checkpoint keys')

        # --- filter out smpl/smplx keys (not in MHR model) ------------------
        total_in_ckpt = len(ckpt_bare)
        filtered = {}
        removed_smpl = 0
        removed_no_match = 0
        for bare_key, v in ckpt_bare.items():
            if 'smpl' in bare_key or 'smplx' in bare_key:
                removed_smpl += 1
                continue
            if bare_key not in model_sd:
                removed_no_match += 1
                continue
            filtered[bare_key] = v
        logger.info(f'  Filtered out {removed_smpl} smpl/smplx keys')
        logger.info(f'  Filtered out {removed_no_match} keys with no matching model parameter')
        logger.info(f'  {len(filtered)} / {total_in_ckpt} keys from checkpoint match model')

        # --- skip shape mismatches, copy into full_sd -----------------------
        loaded = 0
        skipped = 0
        mismatched = []
        for bare_key, v in filtered.items():
            full_key = bare_to_full.get(bare_key)
            if full_key is None:
                skipped += 1
                continue
            if v.shape == full_sd[full_key].shape:
                full_sd[full_key] = v
                loaded += 1
            else:
                skipped += 1
                mismatched.append((bare_key, tuple(v.shape), tuple(full_sd[full_key].shape)))

        for key, ckpt_shape, model_shape in mismatched:
            logger.warning(f'  Size mismatch "{key}": ckpt {ckpt_shape} vs model {model_shape} — skipping')

        # Write back into the LightningModule
        self.load_state_dict(full_sd, strict=False)

        logger.info(f'  Loaded: {loaded}/{len(model_sd)} keys ({loaded/len(model_sd)*100:.1f}%)')
        logger.info(f'  Skipped: {skipped} shape mismatches')
        logger.info(f'  Coverage: {loaded}/{len(model_sd)} params initialized from checkpoint')

        # --- freeze pretrained layers (optional) ----------------------------
        if self.hparams.TRAINING.FREEZE_PRETRAINED:
            self._freeze_pretrained()

        logger.info('=' * 60)

    def _freeze_pretrained(self):
        """Freeze pretrained backbone, decoders, and attention.

        Keeps the MHRRegressor (self.model.head) and MHRHead (self.model.mhr_head)
        trainable. The regressor is the adaptation layer — it maps pretrained
        features to MHR output parameters. Freezing it would make training
        impossible because no parameters could learn.

        Frozen: backbone, depth_decoder, segmentation_decoder, attention
        Trainable: head (MHRRegressor), mhr_head (MHRHead + TorchScript MHR model)
        """
        # Only these pretrained components are frozen — 'head.' (MHRRegressor) is
        # intentionally excluded because it is the adaptation layer that maps
        # pretrained visual features to MHR parameters. The original SMPLXCamHead
        # was a thin wrapper around SMPLX (no trainable params), but MHRRegressor
        # is a learned MLP that replaces it entirely.
        pretrained_prefixes = ('backbone.', 'depth_decoder.',
                               'segmentation_decoder.', 'attention.')

        frozen_count = 0
        total_count = 0
        for name, param in self.named_parameters():
            total_count += 1
            if name.startswith('model.'):
                # Strip 'model.' prefix (LightningModule names self.model.* as
                # 'model.xxx' in named_parameters()). MHRHMR internals like
                # self.backbone become 'model.backbone.conv1.weight'.
                bare = name[len('model.'):]
                is_pretrained = any(bare.startswith(prefix)
                                    for prefix in pretrained_prefixes)
                if is_pretrained:
                    param.requires_grad = False
                    frozen_count += 1
                    continue
            # mhr_head contains only buffers (faces, _smpl_tri_vids, etc.) and
            # the TorchScript MHR model — no nn.Parameter to freeze.
            # The regressor (model.head.*) is intentionally left trainable.

        logger.info(f'  Frozen {frozen_count}/{total_count} parameters (pretrained layers)')
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f'  Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)')

    def configure_optimizers(self):
        # MAPPED FROM: HMRTrainer.configure_optimizers (hmr_trainer.py:247)
        """Configure optimizer with separate LR groups for pretrained vs MHR layers.

        Pretrained layers (backbone, decoders, attention) get a lower learning
        rate (base_lr / 10) since they were initialized from the D-PoSE checkpoint.
        MHR-specific layers (head / MHRRegressor, mhr_head) get the full base_lr
        since they are trained from scratch.

        After UNFREEZE_EPOCH all layers share the same LR.

        Note: 'head.' (MHRRegressor) is NOT in pretrained_prefixes — it is the
        adaptation layer mapping pretrained features to MHR parameters and must
        always be trainable. This matches _freeze_pretrained() which also excludes
        'head.' from freezing.
        """
        base_lr = self.hparams.OPTIMIZER.LR
        wd = self.hparams.OPTIMIZER.WD
        frozen_lr = base_lr / 10.0

        # Same set as _freeze_pretrained() — 'head.' excluded because MHRRegressor
        # is the adaptation layer, not a pretrained component we want to protect.
        pretrained_prefixes = ('backbone.', 'depth_decoder.',
                               'segmentation_decoder.', 'attention.')

        trainable_params = []
        pretrained_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            bare = name[len('model.'):] if name.startswith('model.') else name
            is_pretrained = any(bare.startswith(p) for p in pretrained_prefixes)
            if is_pretrained and self.current_epoch < self.hparams.TRAINING.UNFREEZE_EPOCH:
                pretrained_params.append(param)
            else:
                trainable_params.append(param)

        if not trainable_params:
            # This should never happen if _freeze_pretrained() excludes 'head.',
            # but guard against it to give a clear error instead of "empty param list".
            raise RuntimeError(
                "configure_optimizers: zero trainable parameters! "
                "Check _freeze_pretrained() — MHRRegressor (model.head.*) "
                "must NOT be frozen, and mhr_head TorchScript params are buffers "
                "(not nn.Parameter).")

        # If all params are trainable (no freezing or after unfreeze), use single group
        if not pretrained_params:
            return torch.optim.Adam(trainable_params, lr=base_lr, weight_decay=wd)

        groups = [
            {'params': trainable_params, 'lr': base_lr},
            {'params': pretrained_params, 'lr': frozen_lr},
        ]
        n_train = sum(p.numel() for g in groups for p in g['params']
                       if g.get('lr') == base_lr)
        n_pre = sum(p.numel() for g in groups for p in g['params']
                     if g.get('lr') == frozen_lr)
        logger.info(f'  Optimizer: {n_train:,} params @ lr={base_lr:.1e}, '
                     f'{n_pre:,} params @ lr={frozen_lr:.1e}')
        return torch.optim.Adam(groups, weight_decay=wd)

    def train_dataset(self):
        # MAPPED FROM: HMRTrainer.train_dataset (hmr_trainer.py:255)
        """Create training dataset."""
        options = self.hparams.DATASET
        dataset_names = options.DATASETS_AND_RATIOS.split('_')
        dataset_list = [DatasetHMR(options, ds) for ds in dataset_names]
        train_ds = ConcatDataset(dataset_list)
        return train_ds

    def train_dataloader(self):
        # MAPPED FROM: HMRTrainer.train_dataloader (hmr_trainer.py:264)
        #
        # New: multiprocessing_context='spawn' added (original had none)
        #   Original workers may have run SMPL→MHR conversion (see dataset_wrapper.py)
        #   which uses SMPL model instances — CUDA can't be re-initialized in forked
        #   processes, so 'spawn' is required.
        """Create training dataloader."""
        self.train_ds = self.train_dataset()
        img_dataloader = DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
            drop_last=True,
            # 'spawn' avoids "Cannot re-initialize CUDA in forked subprocess"
            # when on-the-fly SMPL→MHR conversion runs in workers
            multiprocessing_context='spawn'
            if self.hparams.DATASET.NUM_WORKERS > 0 else None,
            persistent_workers=self.hparams.DATASET.NUM_WORKERS > 0,
        )
        return img_dataloader

    def val_dataset(self):
        # MAPPED FROM: HMRTrainer.val_dataset (hmr_trainer.py:277)
        """Create validation dataset."""
        datasets = self.hparams.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_datasets = []
        for dataset_name in datasets:
            val_datasets.append(
                DatasetHMR(
                    options=self.hparams.DATASET,
                    dataset=dataset_name,
                    is_train=False,
                ))
        return val_datasets

    def val_dataloader(self):
        # MAPPED FROM: HMRTrainer.val_dataloader (hmr_trainer.py:291)
        # Same 'spawn' addition as train_dataloader.
        """Create validation dataloader."""
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds,
                    batch_size=self.hparams.DATASET.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.hparams.DATASET.NUM_WORKERS,
                    drop_last=True,
                    multiprocessing_context='spawn'
                    if self.hparams.DATASET.NUM_WORKERS > 0 else None,
                    persistent_workers=self.hparams.DATASET.NUM_WORKERS > 0,
                ))
        return dataloaders

    def test_dataloader(self):
        # MAPPED FROM: HMRTrainer.test_dataloader (hmr_trainer.py:305)
        return self.val_dataloader()
