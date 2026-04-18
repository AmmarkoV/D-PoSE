"""
MHR Trainer Module for D-Pose MHR port.

This module provides a PyTorch Lightning module for training the MHR-based HMR model.
It replaces HMRTrainer and is completely independent of SMPL dependencies.
"""

import os
import torch
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from mhr_constants import NUM_MHR_SKELETON_JOINTS
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
        super(MHRTrainer, self).__init__()

        self.hparams.update(hparams)

        # Import here to avoid circular imports
        from mhr_hmr import MHRHMR
        from mhr_losses import MHRLoss
        from config import MHR_MODEL_PT

        # Load MHR model
        self.mhr_model = torch.load(MHR_MODEL_PT, map_location='cpu', weights_only=False)
        self.add_module('mhr_model', self.mhr_model)

        # Initialize MHR-based HMR model
        self.model = MHRHMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
            mhr_model=self.mhr_model,
        )

        # MHR loss function
        self.loss_fn = MHRLoss(hparams=self.hparams)

        # Dataset
        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()
        self.val_ds = self.val_dataset()

        self.save_itr = 0
        self._val_step_outputs = []  # accumulated by validation_step; read by on_validation_epoch_end

        # Renderer for visualization
        # mhr_model is a TorchScript RecursiveScriptModule; .character is not
        # accessible via __getattr__. Load faces from the portable dump instead.
        import sys as _sys
        _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _proj_root not in _sys.path:
            _sys.path.insert(0, _proj_root)
        from load_mhr_portable import load_portable
        _portable = load_portable(os.path.join(_proj_root, 'mhr_portable_dump', 'mhr_dump_lod1.pt'))
        _faces = _portable['interesting']['character.mesh.faces']
        if isinstance(_faces, torch.Tensor):
            self.faces = _faces.cpu().numpy()
        else:
            self.faces = np.array(_faces)

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )

    def forward(self, x, bbox_center, bbox_scale, img_w, img_h, fl=None):
        """Forward pass through MHR-based HMR model."""
        return self.model(x, bbox_center=bbox_center, bbox_scale=bbox_scale,
                         img_w=img_w, img_h=img_h, fl=fl)

    def training_step(self, batch, batch_nb, dataloader_nb=0):
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
        pred, depth, _, _, part_segms = self(
            images, bbox_center=bbox_center, bbox_scale=bbox_scale,
            img_w=img_w, img_h=img_h, fl=fl
        )
        pred['depth'] = depth
        pred['part_segms'] = part_segms

        # Compute GT SMPL joints from GT MHR vertices (same mapping as pred path)
        # This gives differentiable 3D joint supervision since pred['joints3d_smpl']
        # is differentiable and gt is a constant produced here under no_grad.
        gt_verts = batch.get('vertices')
        mhr_head = getattr(self.model, 'mhr_head', None)
        if gt_verts is not None and mhr_head is not None and 'joints3d_smpl' in pred:
            with torch.no_grad():
                tv = mhr_head._smpl_tri_vids   # [6890, 3]
                w = mhr_head._smpl_baryc        # [6890, 3]
                gv0 = gt_verts[:, tv[:, 0]]
                gv1 = gt_verts[:, tv[:, 1]]
                gv2 = gt_verts[:, tv[:, 2]]
                gt_smpl_v = (w[:, 0].unsqueeze(0).unsqueeze(-1) * gv0
                           + w[:, 1].unsqueeze(0).unsqueeze(-1) * gv1
                           + w[:, 2].unsqueeze(0).unsqueeze(-1) * gv2)
                batch['joints3d'] = torch.einsum(
                    'jv,bvd->bjd', mhr_head._smpl_J_reg, gt_smpl_v
                )  # [B, 24, 3]

        # Compute loss
        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)

        # Log losses
        self.log('train_loss', loss, logger=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(k, v, logger=True, sync_dist=True)

        if batch_nb == 0 and self.current_epoch == 0:
            for key in ('joints3d', 'joints3d_smpl', 'vertices'):
                t = pred.get(key)
                if t is not None:
                    logger.info(f'{key}: requires_grad={t.requires_grad} grad_fn={t.grad_fn}')

        if batch_nb % 200 == 0:
            loss_summary = {k.split('/')[-1]: f'{v.item():.4f}' for k, v in loss_dict.items()}
            logger.info(f'Epoch {self.current_epoch} batch {batch_nb}: {loss_summary}')

        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb=0, vis=False, save=True, mesh_save_dir=None):
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
        pred, depth, _, _, part_segms = self(
            images, bbox_center=bbox_center, bbox_scale=bbox_scale,
            img_w=img_w, img_h=img_h
        )
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
                (w[:, 0].unsqueeze(0).unsqueeze(-1) * gv0
                 + w[:, 1].unsqueeze(0).unsqueeze(-1) * gv1
                 + w[:, 2].unsqueeze(0).unsqueeze(-1) * gv2),
            )
            pred_keypoints_3d = pred_smpl[:, :24]
            gt_keypoints_3d = gt_smpl_joints[:, :24]
        else:
            gt_keypoints_3d = batch.get('joints3d_mhr', batch.get('joints3d', None))
            if gt_keypoints_3d is not None and (
                gt_keypoints_3d.ndim != 3 or gt_keypoints_3d.shape[-1] < 3
            ):
                logger.warning(
                    f"joints3d has unexpected shape {gt_keypoints_3d.shape}; skipping 3D eval"
                )
                gt_keypoints_3d = None
            if gt_keypoints_3d is not None:
                gt_keypoints_3d = gt_keypoints_3d[:, :NUM_MHR_SKELETON_JOINTS]
            else:
                gt_keypoints_3d = batch.get('joints', None)[:, :NUM_MHR_SKELETON_JOINTS]
            pred_keypoints_3d = pred['joints3d'][:, :NUM_MHR_SKELETON_JOINTS]

        # Pelvis-center
        gt_pelvis = (gt_keypoints_3d[:, [1]] + gt_keypoints_3d[:, [2]]) / 2.0
        pred_pelvis = (pred_keypoints_3d[:, [1]] + pred_keypoints_3d[:, [2]]) / 2.0
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        pred_cam_vertices = pred_cam_vertices - pred_pelvis
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
        if gt_cam_vertices is not None:
            gt_cam_vertices = gt_cam_vertices - gt_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
        if gt_cam_vertices is not None:
            error_verts = torch.sqrt(((pred_cam_vertices - gt_cam_vertices) ** 2).sum(dim=-1)).cpu().numpy()
        else:
            error_verts = np.zeros((batch_size, pred_cam_vertices.shape[1]))

        # Reconstruction error (PA-MPJPE)
        r_error, _ = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None
        )
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
        """Log validation metrics at end of epoch (PL v2.0+: no outputs arg)."""
        outputs = self._val_step_outputs
        self._val_step_outputs = []  # clear for next epoch

        logger.info(f'***** Epoch {self.current_epoch} *****')
        val_log = {}

        if len(self.val_ds) > 1:
            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                mpjpe = 1000 * np.hstack(np.array([val[ds_name + '_mpjpe'] for x in outputs for val in x])).mean()
                pampjpe = 1000 * np.hstack(np.array([val[ds_name + '_pampjpe'] for x in outputs for val in x])).mean()
                pve = 1000 * np.hstack(np.array([val[ds_name + '_pve'] for x in outputs for val in x])).mean()

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
                mpjpe = 1000 * np.hstack(np.array([x[ds_name + '_mpjpe'] for x in outputs])).mean()
                pampjpe = 1000 * np.hstack(np.array([x[ds_name + '_pampjpe'] for x in outputs])).mean()
                pve = 1000 * np.hstack(np.array([x[ds_name + '_pve'] for x in outputs])).mean()

                if self.trainer.is_global_zero:
                    logger.info(ds_name + '_MPJPE: ' + str(mpjpe))
                    logger.info(ds_name + '_PA-MPJPE: ' + str(pampjpe))
                    logger.info(ds_name + '_PVE: ' + str(pve))

                    val_log[ds_name + '_val_mpjpe'] = mpjpe
                    val_log[ds_name + '_val_pampjpe'] = pampjpe
                    val_log[ds_name + '_val_pve'] = pve

        self.log('val_loss', val_log[self.val_ds[0].dataset + '_val_pampjpe'], logger=True, sync_dist=True)
        self.log('val_loss_mpjpe', val_log[self.val_ds[0].dataset + '_val_mpjpe'], logger=True, sync_dist=True)
        for k, v in val_log.items():
            self.log(k, v, logger=True, sync_dist=True)

    def test_step(self, batch, batch_nb, dataloader_nb=0):
        return self.validation_step(batch, batch_nb, dataloader_nb)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        """Configure optimizer for training."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD
        )

    def train_dataset(self):
        """Create training dataset."""
        options = self.hparams.DATASET
        dataset_names = options.DATASETS_AND_RATIOS.split('_')
        dataset_list = [DatasetHMR(options, ds) for ds in dataset_names]
        train_ds = ConcatDataset(dataset_list)
        return train_ds

    def train_dataloader(self):
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
            multiprocessing_context='spawn' if self.hparams.DATASET.NUM_WORKERS > 0 else None,
        )
        return img_dataloader

    def val_dataset(self):
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
                )
            )
        return val_datasets

    def val_dataloader(self):
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
                    multiprocessing_context='spawn' if self.hparams.DATASET.NUM_WORKERS > 0 else None,
                )
            )
        return dataloaders

    def test_dataloader(self):
        return self.val_dataloader()
