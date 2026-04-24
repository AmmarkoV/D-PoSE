"""
MHR Training Configuration for D-Pose MHR port.

This module provides configuration for MHR-based training.
"""

import os
import numpy as np
from loguru import logger as _logger


def check_file(path, description, context=None, fatal=False):
    """Check that a file exists; emit verbose path diagnostics when it doesn't.

    Designed for Docker / volume-mount deployments where a file is present on
    the host but the container resolves to a different (wrong) absolute path
    due to CWD, __file__, or proj_root being derived incorrectly.

    When the file is missing the message shows:
      - The absolute path that was attempted
      - Current working directory (reveals CWD-relative resolution)
      - Any caller-supplied context variables (e.g. __file__, _proj_root)
      - A listing of the parent directory so "right directory, wrong filename"
        and "wrong directory entirely" are immediately distinguishable
      - Docker volume-mount reminder with the exact -v flag used

    Args:
        path:        Path to check (absolute or relative to CWD).
        description: Human-readable name of this file (used in error text).
        context:     Optional {label: value} dict printed below the path line.
                     Useful for passing __file__ and the derived proj_root so
                     the reader can see exactly how the path was constructed.
        fatal:       If True raise FileNotFoundError; if False log a warning
                     and return False so callers can decide whether to abort.

    Returns:
        True if the file exists, False otherwise.
    """
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        return True

    cwd = os.getcwd()
    lines = [
        f"MISSING FILE ── {description}",
        f"  path attempted : {abs_path}",
        f"  cwd            : {cwd}",
    ]
    if context:
        for k, v in context.items():
            lines.append(f"  {k:<16} : {v}")

    # Docker hint: the most common cause of "file exists on host, not in
    # container" is a wrong path prefix.  The deploy script mounts the repo
    # root as /home/user/workspace; any path that starts elsewhere will fail.
    lines.append(
        "  Docker mount   : host $REPOSITORY → /home/user/workspace"
        "  (docker/build_and_deploy.sh  -v $mount_pth:/home/user/workspace)"
    )

    # List the parent directory so the reader can immediately see whether
    # the problem is "wrong directory" or "right directory, wrong filename".
    parent = os.path.dirname(abs_path)
    if os.path.isdir(parent):
        contents = sorted(os.listdir(parent))
        basename = os.path.basename(abs_path)
        lines.append(
            f"  parent dir ({parent}) — {len(contents)} item(s):")
        for name in contents[:15]:
            marker = "  ← EXPECTED NAME" if name == basename else ""
            lines.append(f"    {name}{marker}")
        if len(contents) > 15:
            lines.append(f"    … {len(contents) - 15} more")
    else:
        lines.append(f"  parent dir does not exist: {parent}")
        # Walk up to the first existing ancestor so the reader can see where
        # the tree diverges from what is actually on disk.
        ancestor, depth = os.path.dirname(parent), 0
        while ancestor and depth < 5:
            if os.path.isdir(ancestor):
                children = sorted(os.listdir(ancestor))[:8]
                lines.append(
                    f"  first existing ancestor: {ancestor}  →  {children}")
                break
            ancestor, depth = os.path.dirname(ancestor), depth + 1

    full_msg = "\n".join(lines)
    if fatal:
        raise FileNotFoundError(full_msg)
    _logger.warning(full_msg)
    return False

# =============================================================================
# Model Paths
# =============================================================================

# MHR model paths (relative to project root)
MHR_MODEL_DIR = 'assets'
MHR_MODEL_PT = os.path.join(MHR_MODEL_DIR, 'mhr_model.pt')
MHR_FBX_PATH = os.path.join(MHR_MODEL_DIR, 'lod1.fbx')
MHR_BLENDSHAPES_PATH = os.path.join(MHR_MODEL_DIR,
                                    'corrective_blendshapes_lod1.npz')
MHR_CORRECTIVE_ACTIVATION_PATH = os.path.join(MHR_MODEL_DIR,
                                              'corrective_activation.npz')

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
