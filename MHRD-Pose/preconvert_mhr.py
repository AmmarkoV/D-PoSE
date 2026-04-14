"""
MHR Cache Pre-generation Script.

Pre-converts SMPL ground truth from training/validation datasets to MHR
parameters and saves them to .npz cache files. After this runs, DatasetHMR
will load from cache on every __getitem__ call instead of running the
200-iteration SMPL→MHR optimisation per sample.

Usage (from project root, with venv active):
    python MHRD-Pose/preconvert_mhr.py --cfg MHRD-Pose/config_mhr.yaml
    python MHRD-Pose/preconvert_mhr.py --cfg MHRD-Pose/config_mhr.yaml \
        --datasets agora-bfh agora-body --batch_size 64

Arguments:
    --cfg         Path to config yaml (required for dataset options)
    --datasets    Dataset names to convert (default: reads from config
                  DATASETS_AND_RATIOS + VAL_DS)
    --batch_size  Samples per conversion batch (default: 64)
    --cache_dir   Output directory for .npz files (default: data/mhr_cache)
    --resume      Skip datasets whose cache already exists
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — mirrors train_mhr.py
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'MHR'))

from train.utils.train_utils import update_hparams


def preconvert(dataset_name, options, cache_dir, batch_size, resume):
    """Convert a single dataset to MHR cache.

    Processes samples in batches and writes an .npz file under cache_dir.
    """
    cache_path = Path(cache_dir) / f"{dataset_name}_mhr_params.npz"

    if resume and cache_path.exists():
        logger.info(f"[{dataset_name}] Cache already exists at {cache_path} — skipping")
        return

    logger.info(f"[{dataset_name}] Starting conversion → {cache_path}")

    # Import inside function so path setup is complete
    from train.dataset.dataset import DatasetHMR as OriginalDatasetHMR
    from dataset_wrapper import DatasetHMR

    # Build dataset — use the wrapper so we can call _init_converter
    ds = DatasetHMR(options, dataset_name, is_train=True)
    ds._init_converter()
    ds._converter_initialized = True

    n = len(ds)
    logger.info(f"[{dataset_name}] {n} samples to convert (batch_size={batch_size})")

    all_identity = []
    all_expr = []
    all_pose = []
    all_verts = []
    all_joints = []

    # Suppress inner per-iteration tqdm bars while outer bar runs
    os.environ['TQDM_DISABLE'] = '1'

    try:
        with tqdm(total=n, desc=dataset_name, unit="sample") as pbar:
            i = 0
            while i < n:
                end = min(i + batch_size, n)
                batch_poses = []
                batch_betas = []
                batch_transl = []

                for j in range(i, end):
                    # Use parent __getitem__ to avoid triggering on-the-fly conversion
                    item = OriginalDatasetHMR.__getitem__(ds, j)
                    batch_poses.append(item['pose'])
                    batch_betas.append(item['betas'])
                    batch_transl.append(item.get('transl', torch.zeros(3)))

                smpl_data = {
                    'pose':   torch.stack(batch_poses),    # [B, 165]
                    'betas':  torch.stack(batch_betas),    # [B, 10]
                    'transl': torch.stack(batch_transl),   # [B, 3]
                }

                mhr_data = ds._convert_smpl_to_mhr(smpl_data)

                all_identity.append(mhr_data['identity_coeffs'].cpu().numpy())
                all_expr.append(mhr_data['face_expr_coeffs'].cpu().numpy())
                all_pose.append(mhr_data['lbs_model_params'].cpu().numpy())
                all_verts.append(mhr_data['vertices'].cpu().numpy())
                all_joints.append(mhr_data['joints3d'].cpu().numpy())

                pbar.update(end - i)
                i = end

    finally:
        # Always restore tqdm
        os.environ.pop('TQDM_DISABLE', None)

    # Stack along sample axis
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        identity_coeffs=np.concatenate(all_identity, axis=0),
        face_expr_coeffs=np.concatenate(all_expr, axis=0),
        lbs_model_params=np.concatenate(all_pose, axis=0),
        vertices=np.concatenate(all_verts, axis=0),
        joints3d=np.concatenate(all_joints, axis=0),
    )
    logger.info(f"[{dataset_name}] Saved {n} samples → {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-convert datasets to MHR cache")
    parser.add_argument('--cfg',        type=str, required=True,
                        help='Path to config yaml (e.g. MHRD-Pose/config_mhr.yaml)')
    parser.add_argument('--datasets',   nargs='+', default=None,
                        help='Dataset names to convert. Default: reads from config.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Samples per conversion batch (default: 64)')
    parser.add_argument('--cache_dir',  type=str, default=None,
                        help='Output directory. Default: reads CACHE_DIR from config.')
    parser.add_argument('--resume',     action='store_true',
                        help='Skip datasets whose cache already exists.')
    args = parser.parse_args()

    hparams = update_hparams(args.cfg)

    # Resolve datasets
    if args.datasets:
        datasets = args.datasets
    else:
        train_ds = hparams.DATASET.DATASETS_AND_RATIOS.split('_')
        val_ds   = hparams.DATASET.VAL_DS.split('_')
        # deduplicate while preserving order
        seen = set()
        datasets = []
        for d in train_ds + val_ds:
            if d not in seen:
                datasets.append(d)
                seen.add(d)

    # Resolve cache dir
    cache_dir = args.cache_dir or getattr(
        getattr(hparams, 'MHR', None), 'CACHE_DIR', 'data/mhr_cache'
    )

    logger.info(f"Datasets to convert: {datasets}")
    logger.info(f"Cache directory:     {cache_dir}")
    logger.info(f"Batch size:          {args.batch_size}")

    for ds_name in datasets:
        preconvert(ds_name, hparams.DATASET, cache_dir, args.batch_size, args.resume)

    logger.info("All datasets converted.")


if __name__ == '__main__':
    main()
