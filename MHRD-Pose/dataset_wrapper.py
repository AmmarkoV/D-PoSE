"""
Dataset Wrapper for SMPL to MHR Conversion.

This module wraps the existing DatasetHMR and converts SMPL ground truth
to MHR parameters using the Conversion class from MHR tools.

The wrapper performs one-time conversion of SMPL vertices to MHR parameters,
then caches the results for efficient training.
"""

import os
import torch
import numpy as np
from pathlib import Path
from loguru import logger

# Import original dataset
from train.dataset.dataset import DatasetHMR as OriginalDatasetHMR


class DatasetHMR(OriginalDatasetHMR):
    """Dataset wrapper that converts SMPL ground truth to MHR parameters.

    This class extends the original DatasetHMR and adds SMPL→MHR conversion:
    1. Loads SMPL ground truth from original dataset
    2. Converts SMPL vertices to MHR parameters using Conversion class
    3. Returns batch with MHR parameters instead of SMPL parameters

    The conversion is performed once at initialization and cached for efficiency.

    MHR Parameters Returned:
    - identity_coeffs: [B, 45] identity blendshape coefficients
    - face_expr_coeffs: [B, 72] face expression coefficients
    - lbs_model_params: [B, 144] LBS model parameters
    - vertices: [B, V, 3] MHR mesh vertices (in meters)
    - joints3d: [B, J, 3] MHR skeleton joints (in meters)

    Args:
        options: Dataset options
        dataset: Dataset name
        is_train: Whether this is training dataset
    """

    def __init__(self, options, dataset, is_train=True):
        super().__init__(options, dataset, is_train=is_train)

        self.dataset_name = dataset
        self.is_train = is_train

        # MHR conversion cache directory
        self.mhr_cache_dir = Path('data/mhr_cache')
        self.mhr_cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file for this dataset
        self.cache_file = self.mhr_cache_dir / f"{dataset}_mhr_params.npz"

        # Try to load cached MHR parameters
        self.mhr_params_cached = self._try_load_cache()

        # Converter is initialized lazily in each worker process (on first __getitem__
        # call) so that the dataset object remains pickle-clean for spawn workers.
        self._converter_initialized = False

        if self.mhr_params_cached is None:
            logger.info(f"No cache found for {dataset}, will convert on-the-fly")
        else:
            logger.info(f"Loaded cached MHR parameters for {dataset}")

    def _try_load_cache(self):
        """Try to load cached MHR parameters."""
        if self.cache_file.exists():
            try:
                logger.info(f"Loading MHR cache from {self.cache_file}")
                cache_data = np.load(self.cache_file, allow_pickle=True)
                # Eagerly materialise all arrays into a plain dict so the
                # dataset object stays pickle-clean for spawn DataLoader workers.
                # NpzFile holds an open BufferedReader which cannot be pickled.
                return {k: cache_data[k] for k in cache_data.files}
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None

    def _init_converter(self):
        """Initialize SMPL→MHR converter.

        This sets up the Conversion class for on-the-fly conversion.
        Called when cache is not available.
        """
        try:
            import smplx
            from mhr.mhr import MHR
            # conversion.py uses local-relative imports so its directory must be on path
            import sys as _sys, os as _os
            _proj_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            _conv_dir = _os.path.join(_proj_root, "MHR", "tools", "mhr_smpl_conversion")
            if _conv_dir not in _sys.path:
                _sys.path.insert(0, _conv_dir)
            from conversion import Conversion

            # Use CPU for the converter — it runs in DataLoader workers and
            # should not compete with the training model for GPU memory.
            self.device = torch.device('cpu')

            # Load SMPLX model
            from train.core.config import SMPLX_MODEL_DIR
            self.smplx_model = smplx.SMPLX(
                SMPLX_MODEL_DIR,
                gender='neutral',
                use_pca=False,
                flat_hand_mean=True,
                dtype=torch.float32,
            ).cpu()  # explicit .cpu() — must not go to CUDA, runs in DataLoader workers

            # Load MHR model
            from config import MHR_MODEL_PT
            self.mhr_model = torch.load(MHR_MODEL_PT, map_location=self.device, weights_only=False)

            # Create converter
            self.converter = Conversion(
                mhr_model=self.mhr_model,
                smpl_model=self.smplx_model,
                method="pytorch",
                batch_size=64
            )

            logger.info("SMPL→MHR converter initialized")

        except ImportError as e:
            logger.error(f"Failed to import conversion modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize converter: {e}")
            raise

    def _convert_smpl_to_mhr(self, smpl_data):
        """Convert SMPL data to MHR parameters.

        Args:
            smpl_data: Dictionary with SMPL ground truth
                - pose: [N, 165] SMPLX pose (batched over all samples)
                - betas: [N, 10] SMPLX betas
                - transl: [N, 3] SMPLX translation

        Returns:
            Dictionary with MHR parameters:
                - identity_coeffs: [N, 45]
                - face_expr_coeffs: [N, 72]
                - lbs_model_params: [N, 144]
                - vertices: [N, V, 3]
                - joints3d: [N, J, 3]
        """
        batch_size = smpl_data['pose'].shape[0]

        # Conversion.__init__ may have moved smplx_model to GPU internally;
        # pin it back to CPU before every forward pass.
        self.smplx_model = self.smplx_model.cpu()

        # Generate SMPLX mesh — converter is always on CPU
        smplx_output = self.smplx_model(
            betas=smpl_data['betas'].cpu(),
            body_pose=smpl_data['pose'][:, 3:66].cpu(),  # 21 joints × 3
            global_orient=smpl_data['pose'][:, :3].cpu(),
            transl=smpl_data['transl'].cpu() if smpl_data['transl'] is not None else None,
        )

        smpl_vertices = smplx_output.vertices  # [N, 10475, 3]

        # Convert to MHR
        result = self.converter.convert_smpl2mhr(
            smpl_vertices=smpl_vertices,
            single_identity=False,
            is_tracking=False,
            return_mhr_parameters=True,
            return_mhr_meshes=True
        )

        mhr_params = result.result_parameters
        mhr_meshes = result.result_meshes

        # Extract MHR parameters
        # result_parameters structure:
        # - identity_coeffs: [N, 45]
        # - lbs_model_params: [N, 144]
        # - face_expr_coeffs: [N, 72]

        identity_coeffs = mhr_params.get('identity_coeffs',
                                         torch.zeros(batch_size, 45))
        face_expr_coeffs = mhr_params.get('face_expr_coeffs',
                                          torch.zeros(batch_size, 72))
        lbs_model_params = mhr_params.get('lbs_model_params',
                                          torch.zeros(batch_size, 144))

        # Extract vertices and joints from mesh
        vertices = mhr_meshes['vertices']  # [N, V, 3] in cm
        skel_state = mhr_meshes.get('skel_state', None)

        # Convert cm to m
        vertices = vertices * 0.01

        # Extract joints from skeleton state
        if skel_state is not None:
            joints3d = skel_state[:, :, 12:15] * 0.01  # [N, J, 3]
        else:
            # Fallback: use vertex-based approximation
            joints3d = torch.zeros(batch_size, 24, 3)

        return {
            'identity_coeffs': identity_coeffs,
            'face_expr_coeffs': face_expr_coeffs,
            'lbs_model_params': lbs_model_params,
            'vertices': vertices,
            'joints3d': joints3d,
        }

    def __getitem__(self, index):
        """Get item with MHR parameters.

        Args:
            index: Sample index

        Returns:
            Dictionary with:
                - img: Image tensor
                - All original SMPL fields
                - MHR fields: identity_coeffs, face_expr_coeffs, lbs_model_params
        """
        # Get original SMPL data
        item = super().__getitem__(index)

        # Get SMPL parameters for this sample
        smpl_pose = item['pose']  # [165]
        smpl_betas = item['betas']  # [10]
        smpl_transl = item.get('transl', torch.zeros(3))  # [3]

        # Check cache
        if self.mhr_params_cached is not None:
            # Load from cache
            try:
                mhr_identity = self.mhr_params_cached['identity_coeffs'][index:index+1]
                mhr_expr = self.mhr_params_cached['face_expr_coeffs'][index:index+1]
                mhr_pose = self.mhr_params_cached['lbs_model_params'][index:index+1]
                mhr_verts = self.mhr_params_cached['vertices'][index:index+1]
                mhr_joints = self.mhr_params_cached['joints3d'][index:index+1]
            except KeyError:
                # Cache missing some fields, fall back to on-the-fly
                mhr_data = self._convert_single_sample(smpl_pose, smpl_betas, smpl_transl)
                mhr_identity = mhr_data['identity_coeffs']
                mhr_expr = mhr_data['face_expr_coeffs']
                mhr_pose = mhr_data['lbs_model_params']
                mhr_verts = mhr_data['vertices']
                mhr_joints = mhr_data['joints3d']
        else:
            # On-the-fly conversion
            mhr_data = self._convert_single_sample(smpl_pose, smpl_betas, smpl_transl)
            mhr_identity = mhr_data['identity_coeffs']
            mhr_expr = mhr_data['face_expr_coeffs']
            mhr_pose = mhr_data['lbs_model_params']
            mhr_verts = mhr_data['vertices']
            mhr_joints = mhr_data['joints3d']

        # Add MHR fields to item
        item['identity_coeffs'] = mhr_identity.squeeze(0)
        item['face_expr_coeffs'] = mhr_expr.squeeze(0)
        item['lbs_model_params'] = mhr_pose.squeeze(0)
        item['vertices_mhr'] = mhr_verts.squeeze(0)
        item['joints3d_mhr'] = mhr_joints.squeeze(0)

        return item

    def _convert_single_sample(self, pose, betas, transl):
        """Convert single SMPL sample to MHR."""
        # Lazy-initialize the converter in whichever process first needs it.
        # This keeps the dataset pickle-clean for spawn-based DataLoader workers.
        if not self._converter_initialized:
            self._init_converter()
            self._converter_initialized = True

        # Unsqueeze to batch
        smpl_data = {
            'pose': pose.unsqueeze(0),
            'betas': betas.unsqueeze(0),
            'transl': transl.unsqueeze(0),
        }

        mhr_data = self._convert_smpl_to_mhr(smpl_data)

        return {
            'identity_coeffs': mhr_data['identity_coeffs'][0],
            'face_expr_coeffs': mhr_data['face_expr_coeffs'][0],
            'lbs_model_params': mhr_data['lbs_model_params'][0],
            'vertices': mhr_data['vertices'][0],
            'joints3d': mhr_data['joints3d'][0],
        }


def preconvert_dataset(options, dataset_name, output_path):
    """Pre-convert entire dataset to MHR parameters.

    This function converts all samples in a dataset to MHR parameters
    and saves them to a cache file for efficient training.

    Args:
        options: Dataset options
        dataset_name: Name of dataset to convert
        output_path: Path to save converted data
    """
    logger.info(f"Pre-converting {dataset_name} to MHR parameters...")

    # Create dataset
    ds = DatasetHMR(options, dataset_name, is_train=True)

    # Initialize converter
    ds._init_converter()

    # Convert all samples
    all_identity = []
    all_expr = []
    all_pose = []
    all_verts = []
    all_joints = []

    from tqdm import tqdm
    for i in tqdm(range(len(ds)), desc=f"Converting {dataset_name}"):
        # Get SMPL data
        item = super(DatasetHMR, ds).__getitem__(i)

        smpl_data = {
            'pose': item['pose'].unsqueeze(0),
            'betas': item['betas'].unsqueeze(0),
            'transl': item.get('transl', torch.zeros(3)).unsqueeze(0),
        }

        # Convert
        mhr_data = ds._convert_smpl_to_mhr(smpl_data)

        all_identity.append(mhr_data['identity_coeffs'].cpu().numpy())
        all_expr.append(mhr_data['face_expr_coeffs'].cpu().numpy())
        all_pose.append(mhr_data['lbs_model_params'].cpu().numpy())
        all_verts.append(mhr_data['vertices'].cpu().numpy())
        all_joints.append(mhr_data['joints3d'].cpu().numpy())

    # Stack and save
    cache_data = {
        'identity_coeffs': np.stack(all_identity),
        'face_expr_coeffs': np.stack(all_expr),
        'lbs_model_params': np.stack(all_pose),
        'vertices': np.stack(all_verts),
        'joints3d': np.stack(all_joints),
    }

    np.savez(output_path, **cache_data)
    logger.info(f"Saved MHR cache to {output_path}")

    return cache_data
