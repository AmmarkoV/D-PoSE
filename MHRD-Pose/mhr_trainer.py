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
        if self.current_epoch == self.hparams.TRAINING.UNFREEZE_EPOCH:
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

        # Compute GT SMPL joints from GT MHR vertices (same mapping as pred path)
        # This gives differentiable 3D joint supervision since pred['joints3d_smpl']
        # is differentiable and gt is a constant produced here under no_grad.
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

        # Log losses
        self.log('train_loss', loss, logger=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(k, v, logger=True, sync_dist=True)

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

        if batch_nb % 200 == 0:
            loss_summary = {
                k.split('/')[-1]: f'{v.item():.4f}'
                for k, v in loss_dict.items()
            }
            logger.info(
                f'Epoch {self.current_epoch} batch {batch_nb}: {loss_summary}')

        return {'loss': loss}

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

        # Use SMPL joints derived from MHR vertices — same metric as training.
        # Compute GT SMPL joints from GT MHR vertices with the same mhr2smpl mapping.
        pred_smpl = pred.get('joints3d_smpl')
        mhr_head = getattr(self.model, 'mhr_head', None)
        gt_verts_val = batch.get('vertices')
        if pred_smpl is not None and mhr_head is not None and gt_verts_val is not None:
            tv = mhr_head._smpl_tri_vids
            w = mhr_head._smpl_baryc
            gv0 = gt_verts_val[:, tv[:, 0]]
            gv1 = gt_verts_val[:, tv[:, 1]]
            gv2 = gt_verts_val[:, tv[:, 2]]
            gt_smpl_joints = torch.einsum(
                'jv,bvd->bjd',
                mhr_head._smpl_J_reg,
                (w[:, 0].unsqueeze(0).unsqueeze(-1) * gv0 +
                 w[:, 1].unsqueeze(0).unsqueeze(-1) * gv1 +
                 w[:, 2].unsqueeze(0).unsqueeze(-1) * gv2),
            )
            pred_keypoints_3d = pred_smpl[:, :24]
            gt_keypoints_3d = gt_smpl_joints[:, :24]
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

        # Pelvis-center
        gt_pelvis = (gt_keypoints_3d[:, [1]] + gt_keypoints_3d[:, [2]]) / 2.0
        pred_pelvis = (pred_keypoints_3d[:, [1]] +
                       pred_keypoints_3d[:, [2]]) / 2.0
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        pred_cam_vertices = pred_cam_vertices - pred_pelvis
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
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
                    np.array([
                        val[ds_name + '_mpjpe'] for x in outputs for val in x
                    ])).mean()
                pampjpe = 1000 * np.hstack(
                    np.array([
                        val[ds_name + '_pampjpe'] for x in outputs for val in x
                    ])).mean()
                pve = 1000 * np.hstack(
                    np.array(
                        [val[ds_name + '_pve'] for x in outputs
                         for val in x])).mean()

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

        Only the MHR-specific head layers (mhr_head) are kept trainable.
        The pretrained parts (backbone, depth_decoder, segmentation_decoder,
        attention, original head) were loaded from the D-PoSE checkpoint
        and do not need gradients during early training.
        """
        frozen_count = 0
        total_count = 0
        for name, param in self.named_parameters():
            total_count += 1
            if name.startswith('model.model.'):
                # Nested inside self.model (MHRHMR) — freeze if pretrained
                bare = name[len('model.model.'):]
                is_pretrained = any(bare.startswith(prefix) for prefix in
                                    ('backbone.', 'depth_decoder.',
                                     'segmentation_decoder.', 'attention.',
                                     'head.'))
                if is_pretrained:
                    param.requires_grad = False
                    frozen_count += 1
                    continue
            # Also freeze mhr_head sub-parts that come from the loaded
            # TorchScript model (character_torch, face_expressions_model,
            # pose_correctives_model) — these are non-trainable buffers.
            # The trainable parts inside mhr_head are the MHR-specific
            # linear layers that we want to train.

        logger.info(f'  Frozen {frozen_count}/{total_count} parameters (pretrained layers)')
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f'  Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)')

    def configure_optimizers(self):
        # MAPPED FROM: HMRTrainer.configure_optimizers (hmr_trainer.py:247)
        """Configure optimizer with separate LR groups for pretrained vs MHR layers.

        Pretrained layers (backbone, decoders, attention, original head) get
        a lower learning rate (base_lr / 10) since they were initialized from
        the D-PoSE checkpoint.  MHR-specific layers (mhr_head) get the full
        base_lr since they are trained from scratch.

        After UNFREEZE_EPOCH all layers share the same LR.
        """
        base_lr = self.hparams.OPTIMIZER.LR
        wd = self.hparams.OPTIMIZER.WD
        frozen_lr = base_lr / 10.0

        pretrained_prefixes = ('backbone.', 'depth_decoder.',
                               'segmentation_decoder.', 'attention.', 'head.')

        trainable_params = []
        pretrained_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            bare = name[len('model.model.'):] if name.startswith('model.model.') else name
            is_pretrained = any(bare.startswith(p) for p in pretrained_prefixes)
            if is_pretrained and self.current_epoch < self.hparams.TRAINING.UNFREEZE_EPOCH:
                pretrained_params.append(param)
            else:
                trainable_params.append(param)

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
                ))
        return dataloaders

    def test_dataloader(self):
        # MAPPED FROM: HMRTrainer.test_dataloader (hmr_trainer.py:305)
        return self.val_dataloader()
