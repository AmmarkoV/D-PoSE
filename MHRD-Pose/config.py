"""
MHR Training Configuration for D-Pose MHR port.

This module provides configuration for MHR-based training.
"""

import os
import numpy as np

# =============================================================================
# Model Paths
# =============================================================================

# MHR model paths (relative to project root)
MHR_MODEL_DIR = 'assets'
MHR_MODEL_PT = os.path.join(MHR_MODEL_DIR, 'mhr_model.pt')
MHR_FBX_PATH = os.path.join(MHR_MODEL_DIR, 'lod1.fbx')
MHR_BLENDSHAPES_PATH = os.path.join(MHR_MODEL_DIR, 'corrective_blendshapes_lod1.npz')
MHR_CORRECTIVE_ACTIVATION_PATH = os.path.join(MHR_MODEL_DIR, 'corrective_activation.npz')

# MHR metadata for portable loading
MHR_CONVERSION_META_PATH = 'mhr_portable_dump/mhr_conversion_meta_lod1.pt'

# MHR Level of Detail
MHR_LOD = 1

# =============================================================================
# MHR Parameter Dimensions (Verified)
# =============================================================================

NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72
NUM_LBS_MODEL_PARAMS = 204
NUM_TOTAL_PARAMS = 321
NUM_MHR_SKELETON_JOINTS = 24

# =============================================================================
# Default MHR Parameters (shapes only - create tensors as needed)
# =============================================================================

# Shape constants for creating default tensors
DEFAULT_IDENTITY_SHAPE = NUM_IDENTITY_BLENDSHAPES
DEFAULT_EXPR_SHAPE = NUM_FACE_EXPRESSION_BLENDSHAPES
DEFAULT_POSE_SHAPE = NUM_LBS_MODEL_PARAMS
DEFAULT_CAM_SHAPE = 3  # [scale, tx, ty]

# =============================================================================
# Training Defaults
# =============================================================================

# Default loss weights for MHR
DEFAULT_LOSS_WEIGHTS = {
    'identity': 1.0,
    'expression': 0.5,
    'pose': 1.0,
    'camera': 1.0,
    'vertices': 1.0,
    'joints_3d': 5.0,
    'joints_2d': 10.0,
}

# Default training hyperparameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_IMG_RES = 224
DEFAULT_FOCAL_LENGTH = 5000
DEFAULT_LR = 5e-5
DEFAULT_WD = 0.0
DEFAULT_MAX_EPOCHS = 100

# =============================================================================
# Unit Conversion
# =============================================================================

# MHR uses centimeters; convert to meters for consistency with SMPL
CM_TO_M = 0.01
M_TO_CM = 100.0
