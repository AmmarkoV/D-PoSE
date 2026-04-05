"""
MHRD-Pose: MHR-based D-Pose Training Pipeline

This package contains the MHR-based training pipeline for D-Pose,
with all SMPL dependencies removed.

Modules:
    - mhr_constants: MHR-specific constants
    - config: MHR configuration
    - mhr_hmr: MHR-based HMR model
    - mhr_regressor: MHR parameter regressor
    - mhr_head: MHR forward pass head
    - mhr_losses: MHR loss functions
    - mhr_trainer: PyTorch Lightning trainer
    - dataset_wrapper: SMPL to MHR data converter
    - train_mhr: Training entry point
"""

__version__ = '1.0.0'
__author__ = 'D-Pose MHR Port'

# Export main components
from .mhr_constants import (
    NUM_IDENTITY_BLENDSHAPES,
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_LBS_MODEL_PARAMS,
    NUM_MHR_SKELETON_JOINTS,
    CM_TO_M,
    M_TO_CM,
)

__all__ = [
    'NUM_IDENTITY_BLENDSHAPES',
    'NUM_FACE_EXPRESSION_BLENDSHAPES',
    'NUM_LBS_MODEL_PARAMS',
    'NUM_MHR_SKELETON_JOINTS',
    'CM_TO_M',
    'M_TO_CM',
]