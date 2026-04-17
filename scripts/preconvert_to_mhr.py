#!/usr/bin/env python3
"""
Pre-convert SMPL/SMPLX training data to MHR parameters.

This script converts SMPL ground truth data to MHR parameters and caches
them for efficient training. This avoids on-the-fly conversion during training.

Usage:
    python scripts/preconvert_to_mhr.py --dataset agora-bfh --output data/mhr_cache/agora-bfh.npz
    python scripts/preconvert_to_mhr.py --all  # Convert all datasets
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-convert SMPL data to MHR parameters')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to convert (e.g., agora-bfh)')
    parser.add_argument('--all', action='store_true',
                        help='Convert all training datasets')
    parser.add_argument('--output_dir', type=str, default='data/mhr_cache',
                        help='Output directory for cached MHR parameters')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for conversion')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for conversion')
    return parser.parse_args()


def load_mhr_model(device):
    """Load MHR model from checkpoint."""
    mhr_model_path = PROJECT_ROOT / 'mhr_model.pt'
    if not mhr_model_path.exists():
        raise FileNotFoundError(f"MHR model not found at {mhr_model_path}")

    logger.info(f"Loading MHR model from {mhr_model_path}")
    mhr_model = torch.load(str(mhr_model_path), map_location=device)
    return mhr_model


def load_smplx_model(device):
    """Load SMPLX model."""
    import smplx
    from train.core.config import SMPLX_MODEL_DIR

    smplx_path = Path(SMPLX_MODEL_DIR)
    if not smplx_path.exists():
        raise FileNotFoundError(f"SMPLX model not found at {SMPLX_MODEL_DIR}")

    logger.info(f"Loading SMPLX model from {SMPLX_MODEL_DIR}")
    smplx_model = smplx.SMPLX(
        SMPLX_MODEL_DIR,
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    return smplx_model


def create_converter(mhr_model, smplx_model, device, batch_size):
    """Create SMPL to MHR converter."""
    # conversion.py uses local-relative imports so its directory must be on sys.path
    import sys as _sys
    _conv_dir = str(PROJECT_ROOT.parent / "MHR" / "tools" / "mhr_smpl_conversion")
    if _conv_dir not in _sys.path:
        _sys.path.insert(0, _conv_dir)
    from conversion import Conversion

    logger.info("Creating SMPL->MHR converter")
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model,
        method="pytorch",
        batch_size=batch_size
    )
    return converter


def convert_dataset(dataset_name, converter, batch_size, device):
    """Convert a single dataset to MHR parameters."""
    from train.dataset.dataset import DatasetHMR
    from train.core.config import update_hparams
    from train.core.config import DATASET_FOLDERS, DATASET_FILES

    logger.info(f"Converting dataset: {dataset_name}")

    # Load config
    config_path = PROJECT_ROOT / 'configs' / 'dpose_conf.yaml'
    hparams = update_hparams(str(config_path))
    options = hparams.DATASET

    # Create dataset
    dataset = DatasetHMR(options, dataset_name, is_train=True)

    # Get data file
    data_file = DATASET_FILES[True][dataset_name]
    data = np.load(data_file, allow_pickle=True)

    # Get SMPL parameters
    if 'pose_cam' in data.files:
        pose_cam = data['pose_cam'][:, :66].astype(np.float32)  # SMPLX pose
    else:
        pose_cam = data['body_pose']

    if 'shape' in data.files:
        betas = data['shape'].astype(np.float32)
    else:
        betas = data['betas']

    # Ensure correct shapes
    if betas.shape[-1] == 10:
        betas = np.hstack((betas, np.zeros((betas.shape[0], 1))))

    num_samples = len(pose_cam)

    # Initialize storage
    all_identity = []
    all_expr = []
    all_pose = []
    all_verts = []
    all_joints = []

    # Convert in batches
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Converting {dataset_name}"):
        end_idx = min(i + batch_size, num_samples)
        batch_size_actual = end_idx - i

        # Get batch data
        batch_pose = torch.from_numpy(pose_cam[i:end_idx]).to(device)
        batch_betas = torch.from_numpy(betas[i:end_idx]).to(device)
        batch_transl = torch.zeros(batch_size_actual, 3).to(device)

        # Generate SMPLX mesh
        with torch.no_grad():
            smplx_output = converter.smpl_model(
                betas=batch_betas,
                body_pose=batch_pose[:, 3:66],
                global_orient=batch_pose[:, :3],
                transl=batch_transl,
            )
            smpl_vertices = smplx_output.vertices

        # Convert to MHR — request vertices via return_mhr_vertices (numpy array)
        # result.result_meshes is a list of trimesh objects with no skel_state;
        # run a second forward pass to get skel_state for joint positions.
        result = converter.convert_smpl2mhr(
            smpl_vertices=smpl_vertices,
            single_identity=False,
            is_tracking=False,
            return_mhr_parameters=True,
            return_mhr_vertices=True,
        )

        mhr_params = result.result_parameters

        # Extract parameters
        identity = mhr_params.get('identity_coeffs', torch.zeros(batch_size_actual, 45))
        expr = mhr_params.get('face_expr_coeffs', torch.zeros(batch_size_actual, 72))
        pose = mhr_params.get('lbs_model_params', torch.zeros(batch_size_actual, 144))

        # Vertices from result_vertices (numpy [N, V, 3] in cm)
        verts = torch.from_numpy(result.result_vertices).float() * 0.01  # cm to m

        # Joint positions: run MHR forward to get skel_state
        with torch.no_grad():
            _, skel_state = converter._mhr_model(
                identity_coeffs=identity.float().to(device),
                model_parameters=pose.float().to(device),
                face_expr_coeffs=expr.float().to(device),
                apply_correctives=False,
            )
        # Extract translations from skel_state (handles 8-elem and 16-elem formats)
        skel = skel_state.cpu().float()
        if skel.ndim == 4 and skel.shape[-2:] == (4, 4):
            joints = skel[:, :, :3, 3] * 0.01
        elif skel.ndim == 3 and skel.shape[-1] >= 16:
            joints = skel[:, :, 12:15] * 0.01
        elif skel.ndim == 3 and skel.shape[-1] >= 3:
            joints = skel[:, :, :3] * 0.01
        else:
            joints = torch.zeros(batch_size_actual, skel.shape[1] if skel.ndim >= 2 else 24, 3)

        # Store
        all_identity.append(identity.cpu().numpy())
        all_expr.append(expr.cpu().numpy())
        all_pose.append(pose.cpu().numpy())
        all_verts.append(verts.cpu().numpy())
        all_joints.append(joints.cpu().numpy())

    # Stack results
    cache_data = {
        'identity_coeffs': np.concatenate(all_identity, axis=0),
        'face_expr_coeffs': np.concatenate(all_expr, axis=0),
        'lbs_model_params': np.concatenate(all_pose, axis=0),
        'vertices': np.concatenate(all_verts, axis=0),
        'joints3d': np.concatenate(all_joints, axis=0),
    }

    logger.info(f"Conversion complete. Shapes: { {k: v.shape for k, v in cache_data.items()} }")

    return cache_data


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    mhr_model = load_mhr_model(device)
    smplx_model = load_smplx_model(device)
    converter = create_converter(mhr_model, smplx_model, device, args.batch_size)

    # Get list of datasets to convert
    from train.core.config import DATASET_FILES
    training_datasets = list(DATASET_FILES[True].keys())

    if args.dataset:
        datasets_to_convert = [args.dataset]
    elif args.all:
        datasets_to_convert = training_datasets
    else:
        logger.error("Please specify --dataset or --all")
        return

    # Convert each dataset
    for dataset_name in datasets_to_convert:
        if dataset_name not in training_datasets:
            logger.warning(f"Dataset {dataset_name} not found in training datasets, skipping")
            continue

        output_file = output_dir / f"{dataset_name}_mhr.npz"

        if output_file.exists():
            logger.info(f"Cache already exists at {output_file}, skipping")
            continue

        try:
            cache_data = convert_dataset(dataset_name, converter, args.batch_size, device)

            # Save cache
            np.savez(str(output_file), **cache_data)
            logger.success(f"Saved MHR cache to {output_file}")

        except Exception as e:
            logger.error(f"Failed to convert {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
