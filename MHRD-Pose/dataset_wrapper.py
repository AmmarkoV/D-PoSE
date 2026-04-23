"""
Dataset Wrapper for SMPL to MHR Conversion.

MAPPED FROM: train/dataset/dataset.py (original SMPL/SMPLX DatasetHMR)

This module wraps the existing DatasetHMR and converts SMPL ground truth
to MHR parameters using the Conversion class from MHR tools.

The wrapper performs one-time conversion of SMPL vertices to MHR parameters,
then caches the results for efficient training.

File correspondence:
  dataset_wrapper.py:DatasetHMR       ←  train/dataset/dataset.py:DatasetHMR  (extends it)
  dataset_wrapper.py:_extract_joints_from_skel_state  — new utility (no original)
  dataset_wrapper.py:preconvert_dataset  — new standalone (no original)

Key differences from original DatasetHMR:
  - __init__ calls super().__init__() then adds MHR cache/conversion machinery
  - __getitem__ calls super().__getitem__() to get original SMPL data, then adds MHR fields
  - New methods: _try_load_cache, _init_converter, _convert_smpl_to_mhr, _convert_single_sample
  - Original DatasetHMR.__init__ (L56-191) handles pose_cam, betas, gtkps, SMPL/SMPLX model loading
    — most of that is untouched but NOT re-implemented here (inherited from super)
  - Original __getitem__ (L406-584) returns SMPL fields (pose, betas, vertices, joints)
    — wrapper adds identity_coeffs, face_expr_coeffs, lbs_model_params, overwrites vertices/joints3d
  - Original get_vertices (L287-392) runs SMPL/SMPLX forward for GT mesh generation
    — replaced by _convert_smpl_to_mhr which runs SMPLX → Conversion → MHR pipeline
"""

import os
import torch
import numpy as np
from pathlib import Path
from loguru import logger

# Import original dataset
from train.dataset.dataset import DatasetHMR as OriginalDatasetHMR
from train.core.config import DATASET_FILES

_SKEL_STATE_FORMAT_LOGGED = False


def _extract_joints_from_skel_state(skel_state: torch.Tensor) -> torch.Tensor:
    # NEW FUNCTION — no original in train/dataset/dataset.py
    # This is a MHR-specific utility that didn't exist in the original codebase.
    # The original dataset used SMPL joint regressor (J_regressor @ vertices) to
    # extract joints. Here we extract them from MHR's skel_state tensor instead.
    """Extract joint world-space translations from MHR skeleton state.

    MHR.from_files() may return skel_state in one of several formats:
      - [N, J, 8]  — 8-element: [tx, ty, tz, qw, qx, qy, qz, 1], cm
      - [N, J, 16] — column-major 4×4 flattened, translation at indices 12:15, cm
      - [N, J, 4, 4] — 4×4 homogeneous matrices, translation at [:, :, :3, 3], cm

    Returns [N, J, 3] joint translations in metres.
    """
    global _SKEL_STATE_FORMAT_LOGGED
    if not _SKEL_STATE_FORMAT_LOGGED:
        logger.info(
            f"skel_state shape={skel_state.shape} dtype={skel_state.dtype}")
        _SKEL_STATE_FORMAT_LOGGED = True

    if skel_state.ndim == 4 and skel_state.shape[-2] == 4 and skel_state.shape[
            -1] == 4:
        # [N, J, 4, 4] — homogeneous matrix; translation is last column first 3 rows
        return skel_state[:, :, :3, 3].float() * 0.01
    elif skel_state.ndim == 3:
        last = skel_state.shape[-1]
        if last >= 16:
            # Column-major 4×4 flattened: translation at indices 12, 13, 14
            return skel_state[:, :, 12:15].float() * 0.01
        elif last >= 3:
            # 8-element compact: [tx, ty, tz, ...]
            return skel_state[:, :, :3].float() * 0.01
        else:
            logger.warning(
                f"skel_state last dim too small ({last}); falling back to zero joints"
            )
            return torch.zeros(skel_state.shape[0],
                               skel_state.shape[1],
                               3,
                               dtype=torch.float32,
                               device=skel_state.device)
    else:
        logger.warning(
            f"Unknown skel_state format {skel_state.shape}; falling back to zero joints"
        )
        return torch.zeros(skel_state.shape[0],
                           127,
                           3,
                           dtype=torch.float32,
                           device=skel_state.device)


class DatasetHMR(OriginalDatasetHMR):
    # MAPPED FROM: train/dataset/dataset.py:DatasetHMR (L56)
    """Dataset wrapper that converts SMPL ground truth to MHR parameters.

    This class extends the original DatasetHMR and adds SMPL→MHR conversion:
    1. Loads SMPL ground truth from original dataset
    2. Converts SMPL vertices to MHR parameters using Conversion class
    3. Returns batch with MHR parameters instead of SMPL parameters

    The conversion is performed once at initialization and cached for efficiency.

    MHR Parameters Returned:
    - identity_coeffs: [B, 45] identity blendshape coefficients
    - face_expr_coeffs: [B, 72] face expression coefficients
    - lbs_model_params: [B, 204] LBS model parameters
    - vertices: [B, V, 3] MHR mesh vertices (in meters)
    - joints3d: [B, J, 3] MHR skeleton joints (in meters)

    Inherited from Original DatasetHMR (train/dataset/dataset.py:56-191):
    - pose_cam, betas, scale, center, imgname, gender, keypoints
    - SMPL/SMPLX model instances (smpl_male, smpl_female, smplx_male, smplx_female, etc.)
    - J_regressor, smplx2smpl, smplx_part_segm
    - get_vertices() method (L287-392 in original) — used for GT mesh generation
    - rgb_processing() method (L202-236) — image crop + normalization
    - j2d_processing() method (L396-404) — 2D keypoint transform
    - __len__() method (L586-591) — dataset length

    New/overridden methods:
    - __init__: calls super().__init__() then adds MHR cache/conversion
    - __getitem__: calls super().__getitem__() then adds MHR fields
    - _try_load_cache: new — loads .npz cache file
    - _init_converter: new — initializes SMPLX + MHR + Conversion objects
    - _convert_smpl_to_mhr: new — core SMPLX→MHR conversion pipeline
    - _convert_single_sample: new — per-sample conversion wrapper
    """

    def __init__(self, options, dataset, is_train=True):
        # MAPPED FROM: Original DatasetHMR.__init__ (train/dataset/dataset.py:58)
        # super().__init__() runs the original __init__ which:
        #   - loads self.data from .npz file (DATASET_FILES[is_train][dataset])
        #   - extracts pose_cam, betas, scale, center, imgname
        #   - loads SMPL/SMPLX model instances (smpl_male, smplx_female, etc.)
        #   - sets up gender, keypoints, J_regressor, smplx2smpl
        #   - sets up depth/mask dirs if USE_DEPTH
        #   - sets up augmentation (albumentations)
        # All of that is inherited and untouched — we just add MHR on top.
        super().__init__(options, dataset, is_train=is_train)

        self.dataset_name = dataset
        self.is_train = is_train
        # Stored so __setstate__ can reopen self.data after spawn-pickling.
        self._data_path = DATASET_FILES[is_train][dataset]

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
            logger.info(
                f"No cache found for {dataset}, will convert on-the-fly")
        else:
            logger.info(f"Loaded cached MHR parameters for {dataset}")

    def __getstate__(self):
        # NpzFile holds an open BufferedReader that cannot be pickled by spawn.
        # Drop it; __setstate__ reopens it in the worker.
        state = self.__dict__.copy()
        state['data'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = np.load(self._data_path, allow_pickle=True)

    def _try_load_cache(self):
        # NEW METHOD — no original in train/dataset/dataset.py
        # The original dataset had no caching — every __getitem__ ran SMPLX forward
        # via get_vertices() (original L287-392) to generate GT vertices on-the-fly.
        # This method loads pre-converted MHR parameters from a .npz cache file,
        # avoiding repeated SMPLX→MHR conversion at runtime.
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
        # NEW METHOD — no original in train/dataset/dataset.py
        # The original dataset loaded SMPL/SMPLX models in __init__ (L170-183)
        # and used them directly in get_vertices() for GT mesh generation.
        # Here we load SMPLX + MHR + Conversion objects specifically for
        # the SMPL→MHR conversion pipeline.
        """Initialize SMPL→MHR converter.

        This sets up the Conversion class for on-the-fly conversion.
        Called when cache is not available.
        """
        try:
            import smplx
            import sys
            import os
            sys.path.append('MHR/')
            cwd = os.getcwd()
            print("Working Directory is ",
                  cwd)  # Working Directory is  /home/user/workspace
            from mhr.mhr import MHR

            # conversion.py uses local-relative imports so its directory must be on path
            import sys as _sys, os as _os
            _proj_root = _os.path.dirname(
                _os.path.dirname(_os.path.abspath(__file__)))
            _conv_dir = _os.path.join(_proj_root, "MHR", "tools",
                                      "mhr_smpl_conversion")
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
            ).cpu(
            )  # explicit .cpu() — must not go to CUDA, runs in DataLoader workers

            # Create MHR Python instance — Conversion needs .character attribute
            # access which TorchScript RecursiveScriptModule does not expose.
            # MHR.from_files() returns a proper Python nn.Module with .character.
            from config import MHR_MODEL_DIR, MHR_LOD
            import os as _os2
            from pathlib import Path as _Path
            _proj_root2 = _os2.path.dirname(
                _os2.path.dirname(_os2.path.abspath(__file__)))
            _assets_path = _Path(_proj_root2) / MHR_MODEL_DIR
            self.mhr_instance = MHR.from_files(
                folder=_assets_path,
                device=self.device,
                lod=MHR_LOD,
            )

            # Create converter
            self.converter = Conversion(mhr_model=self.mhr_instance,
                                        smpl_model=self.smplx_model,
                                        method="pytorch",
                                        batch_size=64)

            logger.info("SMPL→MHR converter initialized")

        except ImportError as e:
            logger.error(f"Failed to import conversion modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize converter: {e}")
            raise

    def _convert_smpl_to_mhr(self, smpl_data):
        # NEW METHOD — replaces original DatasetHMR.get_vertices() (train/dataset/dataset.py:L287-392)
        #
        # Original get_vertices() ran SMPL/SMPLX forward and returned (vertices, joints)
        # with dataset-specific branches (3dpw/emdb → SMPL, rich → SMPLX→SMPL, h36m → zeros,
        # agora → SMPLX, etc.). All used gender-specific model instances loaded in __init__.
        #
        # This method replaces that pipeline:
        #   1. Runs SMPLX forward (similar to original's SMPLX path) to get SMPLX vertices
        #   2. Runs Conversion.convert_smpl2mhr() to convert SMPLX→MHR (new — no original)
        #   3. Runs MHR forward to extract skel_state → joints (new — no original)
        #   4. Returns MHR params: identity_coeffs, face_expr_coeffs, lbs_model_params, vertices, joints3d
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

        # Generate SMPLX mesh — converter is always on CPU.
        # All optional pose/expression parameters must be passed explicitly for
        # batch sizes > 1 because smplx registers defaults as batch-1 buffers
        # that do not broadcast automatically.
        # Keep camera-space global_orient from pose_cam (first 3 values).
        # pose_cam encodes the body's orientation relative to the camera
        # (OpenCV Y-down convention), which is learnable from the image.
        # Zero only transl — absolute 3D placement is unlearnable from a crop
        # and is handled separately by the camera head (scale, tx, ty).
        # Using zero global_orient would put the body in Y-up canonical space,
        # breaking the 2D reprojection loss (v = fy*py/tz+cy projects head
        # above the pelvis only when py < 0, i.e. Y-down) and flipping the
        # rendered mesh upside-down.
        smplx_output = self.smplx_model(
            betas=smpl_data['betas'].cpu()
            [:, :10],  # dataset pads to 11; smplx expects 10
            body_pose=smpl_data['pose'][:, 3:66].cpu(),  # 21 joints × 3
            global_orient=smpl_data['pose']
            [:, :3].cpu(),  # camera-space body orientation
            transl=torch.zeros(batch_size, 3),
            jaw_pose=torch.zeros(batch_size, 3),
            leye_pose=torch.zeros(batch_size, 3),
            reye_pose=torch.zeros(batch_size, 3),
            left_hand_pose=torch.zeros(batch_size, 45),
            right_hand_pose=torch.zeros(batch_size, 45),
            expression=torch.zeros(batch_size, 10),
        )

        smpl_vertices = smplx_output.vertices  # [N, 10475, 3]

        # Convert to MHR.
        # torch.enable_grad() is required because pytorch_fitting.py runs a
        # gradient-based optimisation internally (loss.backward()).  PL wraps
        # the validation/eval loop in torch.no_grad(), which disables the grad
        # tape and makes loss.backward() raise "does not require grad".
        # torch.enable_grad() is required: pytorch_fitting.py runs gradient-based
        # optimisation internally; PL wraps eval loops in torch.no_grad() which
        # kills the grad tape and makes loss.backward() raise "does not require grad".
        with torch.enable_grad():
            result = self.converter.convert_smpl2mhr(
                smpl_vertices=smpl_vertices,
                single_identity=False,
                is_tracking=False,
                return_mhr_parameters=True,
                return_mhr_vertices=True,  # numpy [N, V, 3] in cm
                return_mhr_meshes=False,
            )

        mhr_params = result.result_parameters  # dict of tensors
        mhr_verts_np = result.result_vertices  # numpy [N, V, 3] in cm

        # Extract parameter tensors
        identity_coeffs = mhr_params.get('identity_coeffs',
                                         torch.zeros(batch_size, 45))
        face_expr_coeffs = mhr_params.get('face_expr_coeffs',
                                          torch.zeros(batch_size, 72))
        lbs_model_params = mhr_params.get('lbs_model_params',
                                          torch.zeros(batch_size, 204))

        # Detach params — they were produced inside an enable_grad context but we
        # only need the values for caching / GT supervision, not for backprop here.
        identity_coeffs = identity_coeffs.detach()
        face_expr_coeffs = face_expr_coeffs.detach()
        lbs_model_params = lbs_model_params.detach()

        # Convert vertices from cm → m
        vertices = torch.from_numpy(mhr_verts_np).float() * 0.01  # [N, V, 3]

        # Obtain joint positions via MHR forward → skel_state.
        # The skel_state format depends on whether MHR.from_files() returns
        # 8-element [tx, ty, tz, qw, qx, qy, qz, 1] or 16-element column-major
        # 4×4 transforms. We detect the format at runtime.
        with torch.no_grad():
            # apply_correctives=True to match mhr_head.py's training-time forward.
            # Using False here would produce joints from a slightly different
            # mesh than what the network is regressing against.
            _, skel_state = self.mhr_instance(
                identity_coeffs=identity_coeffs.float(),
                model_parameters=lbs_model_params.float(),
                face_expr_coeffs=face_expr_coeffs.float(),
                apply_correctives=True,
            )
        joints3d = _extract_joints_from_skel_state(skel_state)

        return {
            'identity_coeffs': identity_coeffs,
            'face_expr_coeffs': face_expr_coeffs,
            'lbs_model_params': lbs_model_params,
            'vertices': vertices,
            'joints3d': joints3d,
        }

    def __getitem__(self, index):
        # MAPPED FROM: Original DatasetHMR.__getitem__ (train/dataset/dataset.py:L406-584)
        #
        # Original __getitem__ returned: img, pose, betas, vertices, joints, keypoints,
        #   scale, center, imgname, gender, depth, faces, etc.
        #   GT vertices/joints were generated via get_vertices() (L287-392) which ran
        #   SMPL/SMPLX forward with gender-specific models.
        #
        # This version:
        #   1. Calls super().__getitem__(index) to get all original SMPL fields
        #   2. Loads or converts MHR parameters (cache or on-the-fly)
        #   3. Adds MHR fields: identity_coeffs, face_expr_coeffs, lbs_model_params
        #   4. Overwrites 'vertices' (MHR verts vs SMPL verts — different vertex counts)
        #   5. Adds 'joints3d_mhr' (separate key from inherited 'joints' which is SMPL)
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
                mhr_identity = self.mhr_params_cached['identity_coeffs'][
                    index:index + 1]
                mhr_expr = self.mhr_params_cached['face_expr_coeffs'][
                    index:index + 1]
                mhr_pose = self.mhr_params_cached['lbs_model_params'][
                    index:index + 1]
                mhr_verts = self.mhr_params_cached['vertices'][index:index + 1]
                mhr_joints = self.mhr_params_cached['joints3d'][index:index +
                                                                1]
            except KeyError:
                # Cache missing some fields, fall back to on-the-fly
                mhr_data = self._convert_single_sample(smpl_pose, smpl_betas,
                                                       smpl_transl)
                mhr_identity = mhr_data['identity_coeffs']
                mhr_expr = mhr_data['face_expr_coeffs']
                mhr_pose = mhr_data['lbs_model_params']
                mhr_verts = mhr_data['vertices']
                mhr_joints = mhr_data['joints3d']
        else:
            # On-the-fly conversion
            mhr_data = self._convert_single_sample(smpl_pose, smpl_betas,
                                                   smpl_transl)
            mhr_identity = mhr_data['identity_coeffs']
            mhr_expr = mhr_data['face_expr_coeffs']
            mhr_pose = mhr_data['lbs_model_params']
            mhr_verts = mhr_data['vertices']
            mhr_joints = mhr_data['joints3d']

        # Add MHR fields to item.
        # 'vertices' overwrites the SMPL vertices from the original dataset
        # (18439 MHR verts vs ~6890 SMPL verts — must not mix them in the loss).
        # 'joints3d_mhr' uses a separate key because the cached joints3d may have
        # been generated with the old (buggy) skel_state extraction; once the cache
        # is regenerated after this fix the key can be renamed to 'joints3d'.
        item['identity_coeffs'] = mhr_identity.squeeze(0)
        item['face_expr_coeffs'] = mhr_expr.squeeze(0)
        item['lbs_model_params'] = mhr_pose.squeeze(0)
        item['vertices'] = mhr_verts.squeeze(
            0)  # [V_mhr, 3] — overrides SMPL verts
        item['joints3d_mhr'] = mhr_joints.squeeze(
            0)  # cached value may still be wrong

        return item

    def _convert_single_sample(self, pose, betas, transl):
        # NEW METHOD — no original in train/dataset/dataset.py
        # Convenience wrapper that unsqueezes a single sample into batch format
        # and calls _convert_smpl_to_mhr. The original dataset had no equivalent
        # — all GT generation happened inside get_vertices() which operated on
        # a single sample directly via SMPL/SMPLX forward pass.
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
    # NEW FUNCTION — no original in train/dataset/dataset.py
    # Convenience function for one-time full-dataset SMPL→MHR conversion.
    # The original dataset generated GT vertices on-the-fly in every __getitem__
    # via get_vertices() (original L287-392). This function pre-computes all
    # MHR parameters and saves them as .npz cache, so runtime __getitem__
    # can load from cache instead of running SMPLX + Conversion every time.
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
