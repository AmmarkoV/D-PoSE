"""
MHR-specific constants for D-Pose MHR port.

This module defines all MHR-specific constants that replace SMPL constants
in the training pipeline.
"""

import numpy as np

# =============================================================================
# MHR Model Constants (Verified from MHR Character)
# =============================================================================

# Number of identity blendshapes (body shape parameters)
NUM_IDENTITY_BLENDSHAPES = 45

# Number of face expression blendshapes
NUM_FACE_EXPRESSION_BLENDSHAPES = 72

# LBS (Linear Blend Skinning) model parameters
# Structure: 144 parameters for rigid transforms and joint poses
# This is the exact value from MHR character.parameter_transform.pose_parameters
# The first 144 params are pose-related, followed by 45 identity + 72 expression blendshapes
NUM_LBS_MODEL_PARAMS = 144

# Total parameter transform size: 144 + 45 + 72 = 261
NUM_TOTAL_PARAMS = 261

# Number of skeleton joints in MHR (144 / 6 = 24 joints with 6D params each)
NUM_MHR_SKELETON_JOINTS = 24

# MHR coordinate system uses centimeters (vs SMPL's meters)
CM_TO_M = 0.01
M_TO_CM = 100.0

# =============================================================================
# MHR Joint/Body Part Constants
# =============================================================================

# MHR joint names following pymomentum convention
# These correspond to the Character's joint hierarchy
MHR_JOINT_NAMES = [
    # Root and spine
    'root',
    'hip',
    'spine',
    'chest',
    'neck',
    'head',
    # Left leg
    'leftUpLeg',
    'leftLeg',
    'leftFoot',
    'leftToeBase',
    # Right leg
    'rightUpLeg',
    'rightLeg',
    'rightFoot',
    'rightToeBase',
    # Left arm
    'leftShoulder',
    'leftUpperArm',
    'leftLowerArm',
    'leftHand',
    # Right arm
    'rightShoulder',
    'rightUpperArm',
    'rightLowerArm',
    'rightHand',
]

NUM_MHR_JOINTS = len(MHR_JOINT_NAMES)

# =============================================================================
# File Paths (relative to project root)
# =============================================================================

# MHR model file paths
MHR_MODEL_PT = 'mhr_model.pt'  # TorchScript MHR model
MHR_FACES_DUMP = 'mhr_portable_dump/mhr_conversion_meta_lod1.pt'
MHR_LOD = 1  # Level of Detail for MHR mesh

# =============================================================================
# Network Output Dimensions
# =============================================================================

# Total MHR parameters predicted by network
TOTAL_MHR_PARAMS = (
    NUM_IDENTITY_BLENDSHAPES +
    NUM_FACE_EXPRESSION_BLENDSHAPES +
    NUM_LBS_MODEL_PARAMS +
    3  # Camera parameters
)

# =============================================================================
# Loss Function Defaults
# =============================================================================

# Default loss weights for MHR training
LOSS_WEIGHTS_MHR = {
    'identity': 1.0,      # Identity coefficient loss weight
    'expression': 0.5,    # Face expression loss weight
    'pose': 1.0,          # LBS model params (pose) loss weight
    'camera': 1.0,        # Camera parameters loss weight
    'vertices': 1.0,      # Per-vertex reconstruction loss
    'joints_3d': 1.0,     # 3D joint loss
    'joints_2d': 1.0,     # 2D projected joint loss
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_mhr_param_names():
    """Return dictionary mapping parameter names to their dimensions."""
    return {
        'identity_coeffs': NUM_IDENTITY_BLENDSHAPES,
        'face_expr_coeffs': NUM_FACE_EXPRESSION_BLENDSHAPES,
        'lbs_model_params': NUM_LBS_MODEL_PARAMS,
        'cam': 3,
    }

def get_total_mhr_output_dim():
    """Return total dimension of MHR network output."""
    return TOTAL_MHR_PARAMS
