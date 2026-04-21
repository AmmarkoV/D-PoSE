"""
MHR-specific constants for D-Pose MHR port.

MAPPED FROM: train/core/constants.py (SMPL/SMPLX constants)
             train/core/config.py (loss weight defaults)

This module defines all MHR-specific constants that replace SMPL constants
in the training pipeline.

File correspondence:
  MHR constants              ←  train/core/constants.py SMPL constants
  NUM_IDENTITY_BLENDSHAPES=45  ←  NUM_BETAS (SMPLX has 10/11 betas)
  NUM_FACE_EXPRESSION_BLENDSHAPES=72  ←  (new — SMPLX has 10 expression dims, not 72)
  NUM_LBS_MODEL_PARAMS=204     ←  NUM_JOINTS_SMPLX*6=132 (6D pose per joint)
  NUM_MHR_SKELETON_JOINTS=24   ←  NUM_JOINTS_SMPL=24  (same count, different joints)
  NUM_REGRESSOR_FEATURE_JOINTS=22  ←  NUM_JOINTS_SMPLX=22  (same count)
  LOSS_WEIGHTS_MHR dict        ←  hparams.MODEL.LOSS_WEIGHT, BETA_LOSS_WEIGHT, etc.
"""

import numpy as np

# =============================================================================
# MHR Model Constants (Verified from MHR Character)
# =============================================================================

# Number of identity blendshapes (body shape parameters)
# ← original: shape/betas = 10 or 11 dims (NUM_BETAS / hparams.MODEL.BETA_LOSS_WEIGHT)
NUM_IDENTITY_BLENDSHAPES = 45

# Number of face expression blendshapes
# ← original: SMPLX expression = 10 dims (not directly exposed as a loss weight,
#   SMPLXCamHead didn't use expression — it was in the model but not predicted)
NUM_FACE_EXPRESSION_BLENDSHAPES = 72

# LBS (Linear Blend Skinning) model parameters
# Structure: [tx(3), ry(3), rz(3), pose_angles(varying), scale(varying)] = 204 total
# Verified by running mhr_model.pt (TorchScript) forward: only size 204 succeeds.
# The MHR parameter_transform internally pads by identity(45) → 204+45=249 total.
# ← original: pose = NUM_JOINTS_SMPLX*6 = 22*6 = 132 (6D rotation per joint)
#   plus 3 for global_orient = 135, but Regressor outputs 6D per joint (22*6=132)
#   plus global_orient = 135 total pose dims
NUM_LBS_MODEL_PARAMS = 204

# Total parameter transform size: 204 + 45 + 72 = 321
NUM_TOTAL_PARAMS = 321

# Total number of joints in the MHR skeleton (full skel_state).
NUM_MHR_JOINTS_TOTAL = 127

# Number of body joints used for loss/eval (matches SMPL's 24-joint body).
# Do NOT use this to index into the raw 127-joint MHR skel_state directly —
# the first N MHR joints are foot/twist procedurals, not body joints. Use
# MHR_TO_SMPL_JOINT_INDICES for that.
# ← original: NUM_JOINTS_SMPL = 24  (same count, same anatomical ordering)
NUM_MHR_SKELETON_JOINTS = 24

# Architectural feature-tensor joint dim used by the regressor.
# This is NOT a semantic body-joint count — it's the J axis of the [B, C, J]
# pose-feature tensor produced by the upstream HRNet + attention head, which
# was pretrained with a 22-joint convention (SMPL body minus 2 hands). Changing
# this would break loading of the COCO-pretrained HRNet weights.
# ← original: NUM_JOINTS_SMPLX = 22  (same number, coincidental — SMPLX body joints)
NUM_REGRESSOR_FEATURE_JOINTS = 22

# MHR coordinate system uses centimeters (vs SMPL's meters)
CM_TO_M = 0.01
M_TO_CM = 100.0

# =============================================================================
# MHR Joint/Body Part Constants
# =============================================================================

# Real MHR skeleton joint names (127 joints total) — extracted from
# mhr_portable_dump/mhr_summary_lod1.json (character_torch.skeleton.joint_names).
# The order matches skel_state's joint axis exactly.
MHR_JOINT_NAMES = [
    'body_world',
    'root',
    'l_upleg',
    'l_lowleg',
    'l_foot',
    'l_talocrural',
    'l_subtalar',
    'l_transversetarsal',
    'l_ball',
    'l_lowleg_twist1_proc',
    'l_lowleg_twist2_proc',
    'l_lowleg_twist3_proc',
    'l_lowleg_twist4_proc',
    'l_upleg_twist0_proc',
    'l_upleg_twist1_proc',
    'l_upleg_twist2_proc',
    'l_upleg_twist3_proc',
    'l_upleg_twist4_proc',
    'r_upleg',
    'r_lowleg',
    'r_foot',
    'r_talocrural',
    'r_subtalar',
    'r_transversetarsal',
    'r_ball',
    'r_lowleg_twist1_proc',
    'r_lowleg_twist2_proc',
    'r_lowleg_twist3_proc',
    'r_lowleg_twist4_proc',
    'r_upleg_twist0_proc',
    'r_upleg_twist1_proc',
    'r_upleg_twist2_proc',
    'r_upleg_twist3_proc',
    'r_upleg_twist4_proc',
    'c_spine0',
    'c_spine1',
    'c_spine2',
    'c_spine3',
    'r_clavicle',
    'r_uparm',
    'r_lowarm',
    'r_wrist_twist',
    'r_wrist',
    'r_pinky0',
    'r_pinky1',
    'r_pinky2',
    'r_pinky3',
    'r_pinky_null',
    'r_ring1',
    'r_ring2',
    'r_ring3',
    'r_ring_null',
    'r_middle1',
    'r_middle2',
    'r_middle3',
    'r_middle_null',
    'r_index1',
    'r_index2',
    'r_index3',
    'r_index_null',
    'r_thumb0',
    'r_thumb1',
    'r_thumb2',
    'r_thumb3',
    'r_thumb_null',
    'r_lowarm_twist1_proc',
    'r_lowarm_twist2_proc',
    'r_lowarm_twist3_proc',
    'r_lowarm_twist4_proc',
    'r_uparm_twist0_proc',
    'r_uparm_twist1_proc',
    'r_uparm_twist2_proc',
    'r_uparm_twist3_proc',
    'r_uparm_twist4_proc',
    'l_clavicle',
    'l_uparm',
    'l_lowarm',
    'l_wrist_twist',
    'l_wrist',
    'l_pinky0',
    'l_pinky1',
    'l_pinky2',
    'l_pinky3',
    'l_pinky_null',
    'l_ring1',
    'l_ring2',
    'l_ring3',
    'l_ring_null',
    'l_middle1',
    'l_middle2',
    'l_middle3',
    'l_middle_null',
    'l_index1',
    'l_index2',
    'l_index3',
    'l_index_null',
    'l_thumb0',
    'l_thumb1',
    'l_thumb2',
    'l_thumb3',
    'l_thumb_null',
    'l_lowarm_twist1_proc',
    'l_lowarm_twist2_proc',
    'l_lowarm_twist3_proc',
    'l_lowarm_twist4_proc',
    'l_uparm_twist0_proc',
    'l_uparm_twist1_proc',
    'l_uparm_twist2_proc',
    'l_uparm_twist3_proc',
    'l_uparm_twist4_proc',
    'c_neck',
    'c_neck_twist1_proc',
    'c_neck_twist0_proc',
    'c_head',
    'c_jaw',
    'c_teeth',
    'c_jaw_null',
    'c_tongue0',
    'c_tongue1',
    'c_tongue2',
    'c_tongue3',
    'c_tongue4',
    'r_eye',
    'r_eye_null',
    'l_eye',
    'l_eye_null',
    'c_head_null',
]
assert len(MHR_JOINT_NAMES) == NUM_MHR_JOINTS_TOTAL, \
    f"MHR_JOINT_NAMES has {len(MHR_JOINT_NAMES)} names, expected {NUM_MHR_JOINTS_TOTAL}"

NUM_MHR_JOINTS = len(MHR_JOINT_NAMES)

# Mapping: SMPL joint index → MHR joint index.
# SMPL has 24 joints in the standard ordering:
#   0 pelvis, 1 L_hip, 2 R_hip, 3 spine1, 4 L_knee, 5 R_knee,
#   6 spine2, 7 L_ankle, 8 R_ankle, 9 spine3, 10 L_foot, 11 R_foot,
#   12 neck, 13 L_collar, 14 R_collar, 15 head, 16 L_shoulder, 17 R_shoulder,
#   18 L_elbow, 19 R_elbow, 20 L_wrist, 21 R_wrist, 22 L_hand, 23 R_hand
# Use `skel_state[:, MHR_TO_SMPL_JOINT_INDICES, :3]` to get SMPL-ordered joints
# from the raw 127-joint MHR skel_state. Approximate anatomy mapping only —
# fingers/twist bones intentionally left out of SMPL side.
MHR_TO_SMPL_JOINT_INDICES = [
    1,  # 0  pelvis     ← MHR 'root'
    2,  # 1  L_hip       ← 'l_upleg'
    18,  # 2  R_hip       ← 'r_upleg'
    34,  # 3  spine1      ← 'c_spine0'
    3,  # 4  L_knee      ← 'l_lowleg'
    19,  # 5  R_knee      ← 'r_lowleg'
    35,  # 6  spine2      ← 'c_spine1'
    4,  # 7  L_ankle     ← 'l_foot'
    20,  # 8  R_ankle     ← 'r_foot'
    36,  # 9  spine3      ← 'c_spine2'
    8,  # 10 L_foot(toe) ← 'l_ball'
    24,  # 11 R_foot(toe) ← 'r_ball'
    110,  # 12 neck        ← 'c_neck'
    74,  # 13 L_collar    ← 'l_clavicle'
    38,  # 14 R_collar    ← 'r_clavicle'
    113,  # 15 head        ← 'c_head'
    75,  # 16 L_shoulder  ← 'l_uparm'
    39,  # 17 R_shoulder  ← 'r_uparm'
    76,  # 18 L_elbow     ← 'l_lowarm'
    40,  # 19 R_elbow     ← 'r_lowarm'
    78,  # 20 L_wrist     ← 'l_wrist'
    42,  # 21 R_wrist     ← 'r_wrist'
    88,  # 22 L_hand      ← 'l_middle1'
    52,  # 23 R_hand      ← 'r_middle1'
]
assert len(MHR_TO_SMPL_JOINT_INDICES) == NUM_MHR_SKELETON_JOINTS, \
    f"mapping has {len(MHR_TO_SMPL_JOINT_INDICES)}, expected {NUM_MHR_SKELETON_JOINTS}"

# =============================================================================
# File Paths (relative to project root)
# =============================================================================
# ← original: SMPL_MODEL_DIR, SMPLX_MODEL_DIR, SMPL_MEAN_PARAMS, SMPLX2SMPL
#   in train/core/config.py — replaced by MHR-specific paths
# MHR model file paths
MHR_MODEL_PT = 'mhr_model.pt'  # TorchScript MHR model
MHR_FACES_DUMP = 'mhr_portable_dump/mhr_conversion_meta_lod1.pt'
MHR_LOD = 1  # Level of Detail for MHR mesh

# =============================================================================
# Network Output Dimensions
# =============================================================================

# Total MHR parameters predicted by network
TOTAL_MHR_PARAMS = (
    NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES +
    NUM_LBS_MODEL_PARAMS + 3  # Camera parameters
)

# =============================================================================
# Loss Function Defaults
# =============================================================================
# ← original loss weights from train/core/config.py:
#   LOSS_WEIGHT = 60.        → self.loss_weight
#   SHAPE_LOSS_WEIGHT = 0.   → replaced by IDENTITY_LOSS_WEIGHT + EXPR_LOSS_WEIGHT
#   POSE_LOSS_WEIGHT = 1.    → self.pose_loss_weight (same)
#   JOINT_LOSS_WEIGHT = 5.   → self.joint_loss_weight (same)
#   KEYPOINT_LOSS_WEIGHT = 10. → self.keypoint_loss_weight_2d (same)
#   BETA_LOSS_WEIGHT = 0.001 → replaced by IDENTITY_LOSS_WEIGHT (1.0) + EXPR_LOSS_WEIGHT (0.5)
#   ADVERSARIAL_LOSS_WEIGHT = 0.1 → not used in MHRLoss
LOSS_WEIGHTS_MHR = {
    'identity': 1.0,  # Identity coefficient loss weight
    'expression': 0.5,  # Face expression loss weight
    'pose': 1.0,  # LBS model params (pose) loss weight
    'camera': 1.0,  # Camera parameters loss weight
    'vertices': 1.0,  # Per-vertex reconstruction loss
    'joints_3d': 1.0,  # 3D joint loss
    'joints_2d': 1.0,  # 2D projected joint loss
}

# =============================================================================
# Utility Functions
# =============================================================================


def get_mhr_param_names():
    # ← original: train/core/constants.py had no equivalent — these are new MHR-specific names
    """Return dictionary mapping parameter names to their dimensions."""
    return {
        'identity_coeffs': NUM_IDENTITY_BLENDSHAPES,
        'face_expr_coeffs': NUM_FACE_EXPRESSION_BLENDSHAPES,
        'lbs_model_params': NUM_LBS_MODEL_PARAMS,
        'cam': 3,
    }


def get_total_mhr_output_dim():
    # ← original: no equivalent — original HMR output dim was pose(135) + shape(11) + cam(3) = 149
    #   MHR output dim: identity(45) + expr(72) + lbs(204) + cam(3) = 324
    """Return total dimension of MHR network output."""
    return TOTAL_MHR_PARAMS
