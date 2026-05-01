#!/usr/bin/env python3
"""
test_demo_comparison.py — Automated regression test for MHRD-Pose demo.

Compares MHRD-Pose/demo_webcam.py output against demo_webcam.py (original
SMPL+MHR optimization baseline) and flags the known failure modes:

  T1  not_upside_down    Head Y < pelvis Y in camera frame (Y-down)
  T2  faces_camera       Chest normal has negative Z (pointing toward camera)
  T3  joint_range        All joints within 2 m of pelvis (physically plausible)
  T4  arm_responsive     Elbow/wrist pelvis-relative std > 2 cm across frames
  T5  rotation_tracks    Shoulder azimuth std > 5° across frames
  T6  vs_reference       Mean centered MPJPE < 200 mm vs original demo
                         (only runs when --ref-ckpt is supplied)
  T7  elbow_angle_varies Elbow bend-angle std > 10° across frames
  T8  pred_pose_varies   Raw pred_pose mean-std > 0.01 across frames
  T9  vertices_upside    Top vertices near head joint, not feet
  T10 joints2d_near_bbox  Model's 2D projected joints land near detection bbox
                          (catches cam_t errors)
  T11 side_by_side_diag  Prints cam_t and elbow angles side-by-side for
                          MHRD vs reference models (diagnostic, not pass/fail)
  T12 elbow_not_tpose    Mean elbow angle < 170° (catches frozen T-pose arms)
  T13 depth_consistent   cam_t_z CV < 35% across frames (scale regressor quality)
  T14 pelvis_on_person   Pelvis projected via cam_t lands within 35% of bbox
                         (definitive mesh-displacement test)
  T15 scale_plausible    Back-computed scale s = 2*fl/(bbox_h*cam_t_z) in [0.3,8]
  T16 cache_plausible    (--cache-dir) MHR forward on cached params: body heights
                         in [0.8, 2.5] m  → preconvert stage quality
  T17 cache_diversity    (--cache-dir) lbs_model_params variance > 0.05
                         → cached poses are not collapsed to T-pose
  T18 cache_orient_varies (--cache-dir) root rotation std > 0.3
                         → global orientation is actually being predicted

Coordinate convention assumed for joints3d_smpl:
  Camera frame — X right, Y down, Z forward (into scene), units metres.
  A right-side-up person: head_Y < pelvis_Y.
  A forward-facing person: chest normal Z < 0.

Usage (run from D-PoSE root):
    python MHRD-Pose/test_demo_comparison.py \\
        --input crazydance.mp4 \\
        --ckpt  MHRD-Pose/test.ckpt [--frames 20]

    # With visual dumps (headless-safe — writes JPEGs, no display window):
    python MHRD-Pose/test_demo_comparison.py \\
        --input crazydance.mp4 \\
        --ckpt  MHRD-Pose/test.ckpt \\
        --out-dir debug_diag/

    # With cache checks (T16/T17/T18 — run after preconvert_mhr.sh):
    python MHRD-Pose/test_demo_comparison.py \\
        --input crazydance.mp4 \\
        --ckpt  MHRD-Pose/test.ckpt \\
        --cache-dir data/mhr_cache

    # Full run — all checks including reference model (T6):
    python MHRD-Pose/test_demo_comparison.py \\
        --input crazydance.mp4 \\
        --ckpt  MHRD-Pose/test.ckpt \\
        --ref-ckpt data/ckpt/paper_arxiv.ckpt \\
        --cache-dir data/mhr_cache \\
        --out-dir debug_diag/
"""

import os
import sys
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS_DIR)

# ─── SMPL-24 joint indices (only 22 body joints used here) ───────────────────
PELVIS     = 0
L_HIP      = 1
R_HIP      = 2
SPINE1     = 3
L_KNEE     = 4
R_KNEE     = 5
SPINE2     = 6
L_ANKLE    = 7
R_ANKLE    = 8
SPINE3     = 9
NECK       = 12
HEAD       = 15
L_SHOULDER = 16
R_SHOULDER = 17
L_ELBOW    = 18
R_ELBOW    = 19
L_WRIST    = 20
R_WRIST    = 21

SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
]


# ─── Check result ─────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name:   str
    passed: bool
    detail: str

    def __str__(self) -> str:
        mark = '✓ PASS' if self.passed else '✗ FAIL'
        return f'  {mark}  {self.name:30s}  {self.detail}'


# ─── Individual checks ────────────────────────────────────────────────────────

def check_not_upside_down(joints: np.ndarray) -> CheckResult:
    """T1: In Y-down camera frame, head Y must be less than pelvis Y."""
    head_y   = joints[:, HEAD,   1]
    pelvis_y = joints[:, PELVIS, 1]
    frac_ok  = float(np.mean(head_y < pelvis_y))
    return CheckResult(
        name='T1_not_upside_down',
        passed=frac_ok >= 0.80,
        detail=f'{frac_ok*100:.1f}% of frames have head above pelvis (need ≥80%)',
    )


def check_faces_camera(joints: np.ndarray) -> CheckResult:
    """T2: Chest normal should have negative Z (facing toward camera).

    chest_normal = cross(neck - pelvis, right_shoulder - left_shoulder).
    In Y-down camera frame, a forward-facing person gives chest_normal.z < 0.
    """
    spine  = joints[:, NECK,       :] - joints[:, PELVIS,     :]  # [N, 3]
    sh_dir = joints[:, R_SHOULDER, :] - joints[:, L_SHOULDER, :]  # [N, 3]
    chest_n = np.cross(spine, sh_dir)                              # [N, 3]
    frac_ok = float(np.mean(chest_n[:, 2] < 0))
    return CheckResult(
        name='T2_faces_camera',
        passed=frac_ok >= 0.70,
        detail=f'{frac_ok*100:.1f}% of frames face the camera (need ≥70%)',
    )


def check_joint_range(joints: np.ndarray) -> CheckResult:
    """T3: All joints must be within 2 m of the pelvis (physically plausible)."""
    pelvis = joints[:, PELVIS:PELVIS+1, :]          # [N, 1, 3]
    dists  = np.linalg.norm(joints - pelvis, axis=-1)  # [N, J]
    frac_ok = float(np.mean(dists < 2.0))
    return CheckResult(
        name='T3_joint_range',
        passed=frac_ok >= 0.95,
        detail=f'{frac_ok*100:.1f}% of joint-frames within 2 m of pelvis (need ≥95%)',
    )


def check_arm_responsive(joints: np.ndarray) -> CheckResult:
    """T4: Arm joints must vary across frames (std > 2 cm pelvis-relative).

    Detects the failure mode where arms are frozen regardless of input pose.
    """
    pelvis   = joints[:, PELVIS:PELVIS+1, :]                          # [N, 1, 3]
    arm_idx  = [L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]
    arm_rel  = joints[:, arm_idx, :] - pelvis                         # [N, 4, 3]
    mean_std = float(arm_rel.std(axis=0).mean())                      # scalar (m)
    return CheckResult(
        name='T4_arm_responsive',
        passed=mean_std > 0.02,
        detail=f'Arm joint std = {mean_std*100:.1f} cm pelvis-relative (need >2 cm)',
    )


def check_rotation_tracks(joints: np.ndarray) -> CheckResult:
    """T5: Shoulder azimuth must vary across frames (std > 5°).

    Detects the failure mode where global rotation is not predicted.
    """
    l_sh   = joints[:, L_SHOULDER, :]
    r_sh   = joints[:, R_SHOULDER, :]
    sh_xz  = (r_sh - l_sh)[:, [0, 2]]                        # [N, 2] XZ plane
    norms  = np.linalg.norm(sh_xz, axis=-1, keepdims=True) + 1e-8
    azimuth   = np.arctan2(sh_xz[:, 1] / norms[:, 0],
                           sh_xz[:, 0] / norms[:, 0])        # [N]
    angle_std = float(azimuth.std())
    return CheckResult(
        name='T5_rotation_tracks',
        passed=angle_std > np.deg2rad(5.0),
        detail=f'Shoulder azimuth std = {np.rad2deg(angle_std):.1f}° (need >5°)',
    )


def check_elbow_angle_varies(joints: np.ndarray) -> CheckResult:
    """T7: Local elbow bend angle must vary across frames (std > 10°).

    Unlike T4, this is invariant to global rotation and translation —
    it measures whether the network predicts different LOCAL arm poses.
    A frozen arm (e.g. always straight or always at rest) will fail.
    """
    results = []
    for side in [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)]:
        sh, el, wr = side
        upper = joints[:, el, :] - joints[:, sh, :]   # [N, 3] upper-arm vector
        lower = joints[:, wr, :] - joints[:, el, :]   # [N, 3] forearm vector
        dot  = np.sum(upper * lower, axis=-1)
        norm = (np.linalg.norm(upper, axis=-1) * np.linalg.norm(lower, axis=-1) + 1e-8)
        angles = np.arccos(np.clip(dot / norm, -1.0, 1.0))   # [N] radians
        results.append(float(angles.std()))
    mean_std_deg = float(np.mean(results)) * 180.0 / np.pi
    return CheckResult(
        name='T7_elbow_angle_varies',
        passed=mean_std_deg > 10.0,
        detail=f'Elbow bend-angle std = {mean_std_deg:.1f}° (need >10°; '
               f'tests LOCAL pose, not global motion)',
    )


def check_pred_pose_varies(pred_poses: Optional[np.ndarray]) -> CheckResult:
    """T8: Raw predicted pose parameters must vary across frames.

    pred_pose is [N, D] (LBS model params).  If it's constant or near-constant
    across all frames the network is ignoring the image content entirely.
    """
    if pred_poses is None or len(pred_poses) < 2:
        return CheckResult('T8_pred_pose_varies', False,
                           'pred_pose not available — check model output keys')
    mean_std = float(pred_poses.std(axis=0).mean())
    return CheckResult(
        name='T8_pred_pose_varies',
        passed=mean_std > 0.01,
        detail=f'pred_pose mean-std = {mean_std:.4f} across frames (need >0.01)',
    )


def check_vertices_orientation(joints: np.ndarray,
                               vertices: np.ndarray) -> CheckResult:
    """T9: Top-1% of vertices by Y must be closer to the head joint than to the feet.

    joints3d_smpl and the rendered vertices can live in different coordinate
    frames.  This catches the case where joints appear correct but the mesh
    rendered from vertices is upside down (the bug shown in the screenshot).

    Uses the same Y-convention as joints3d_smpl — so the check is self-consistent
    regardless of whether that convention is Y-up or Y-down.
    """
    ok_count = 0
    for j, v in zip(joints, vertices):           # per frame
        n_top  = max(1, len(v) // 100)
        # "top" = smallest Y (matches joints3d_smpl convention)
        top_idx     = np.argpartition(v[:, 1], n_top)[:n_top]
        top_centroid = v[top_idx].mean(axis=0)

        dist_head  = np.linalg.norm(top_centroid - j[HEAD])
        dist_l_ank = np.linalg.norm(top_centroid - j[L_ANKLE])
        dist_r_ank = np.linalg.norm(top_centroid - j[R_ANKLE])
        dist_feet  = min(dist_l_ank, dist_r_ank)

        if dist_head < dist_feet:
            ok_count += 1

    frac_ok = ok_count / max(len(joints), 1)
    return CheckResult(
        name='T9_vertices_not_upside_down',
        passed=frac_ok >= 0.80,
        detail=f'{frac_ok*100:.1f}% of frames: top vertices are near head joint '
               f'(not feet) — catches Y-flip in vertices vs joints (need ≥80%)',
    )


def check_joints2d_near_bbox(joints2d: np.ndarray,
                             dets: np.ndarray) -> CheckResult:
    """T10: Projected 2D joints should land near the detected bounding box.

    Catches cam_t errors — if the camera translation is wrong, the 3D joints
    will project to the wrong 2D location in the image.

    joints2d: [N, 22, 2] pixel coords (x, y)
    dets:     [N, 5] — cx, cy, scale_px, _, score
    """
    if joints2d is None or len(joints2d) == 0:
        return CheckResult('T10_joints2d_near_bbox', False,
                           'joints2d not available from model output')

    if len(dets) != len(joints2d):
        return CheckResult('T10_joints2d_near_bbox', False,
                           f'det/bbox count mismatch ({len(dets)} vs {len(joints2d)})')

    ok_count = 0
    for frame_j2d, det in zip(joints2d, dets):
        cx, cy, scale_px = det[0], det[1], det[2]
        # BBox half-dimensions in pixels, with 50% margin for pose extremes
        hw = scale_px * 0.75
        hh = scale_px * 0.55
        x_min, x_max = cx - hw, cx + hw
        y_min, y_max = cy - hh, cy + hh

        # Check what fraction of joints land inside the padded bbox
        inside_x = (frame_j2d[:, 0] >= x_min) & (frame_j2d[:, 0] <= x_max)
        inside_y = (frame_j2d[:, 1] >= y_min) & (frame_j2d[:, 1] <= y_max)
        frac_inside = float((inside_x & inside_y).mean())
        if frac_inside >= 0.6:
            ok_count += 1

    frac_ok = ok_count / max(len(joints2d), 1)
    return CheckResult(
        name='T10_joints2d_near_bbox',
        passed=frac_ok >= 0.80,
        detail=f'{frac_ok*100:.1f}% of frames have ≥60% joints inside padded bbox (need ≥80%)',
    )


def check_elbow_not_tpose(joints: np.ndarray) -> CheckResult:
    """T12: Mean elbow bend angle should be far from T-pose (~180°).

    Catches the failure mode where arms are frozen in a T-pose (straight arms)
    regardless of the actual pose in the video.

    A T-pose has elbows straight (~180° bend). A natural walking/running pose
    has bent elbows (typically 90°-150°).
    """
    for side in [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)]:
        sh, el, wr = side
        upper = joints[:, el, :] - joints[:, sh, :]   # [N, 3]
        lower = joints[:, wr, :] - joints[:, el, :]   # [N, 3]
        dot  = np.sum(upper * lower, axis=-1)
        norm = (np.linalg.norm(upper, axis=-1) *
                np.linalg.norm(lower, axis=-1) + 1e-8)
        angles = np.arccos(np.clip(dot / norm, -1.0, 1.0))   # [N] radians
        mean_deg = float(np.mean(angles)) * 180.0 / np.pi
        if mean_deg < 170:
            continue
    mean_elbow = 0.0
    for side in [(L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)]:
        sh, el, wr = side
        upper = joints[:, el, :] - joints[:, sh, :]
        lower = joints[:, wr, :] - joints[:, el, :]
        dot  = np.sum(upper * lower, axis=-1)
        norm = (np.linalg.norm(upper, axis=-1) *
                np.linalg.norm(lower, axis=-1) + 1e-8)
        angles = np.arccos(np.clip(dot / norm, -1.0, 1.0))
        mean_elbow += float(np.mean(angles)) * 180.0 / np.pi
    mean_elbow /= 2.0

    return CheckResult(
        name='T12_elbow_not_tpose',
        passed=mean_elbow < 170.0,
        detail=f'Mean elbow angle = {mean_elbow:.1f}° (need <170°; ~180° → frozen T-pose)',
    )


def check_depth_consistent(cam_t: Optional[np.ndarray]) -> CheckResult:
    """T13: cam_t_z (depth) coefficient of variation must be < 35%.

    A well-trained model predicts consistent depth across frames for a static
    scene.  If cam_t_z jumps from 3 m to 14 m the scale parameter is not
    being predicted correctly — this makes the mesh appear at wildly different
    sizes frame-to-frame.

    CV = std / mean.  Threshold 0.35 is generous; original D-PoSE is ~0.10.
    """
    if cam_t is None or len(cam_t) < 3:
        return CheckResult('T13_depth_consistent', False,
                           'pred_cam_t not available — cannot assess depth consistency')
    z = cam_t[:, 2]
    mean_z = float(z.mean())
    std_z  = float(z.std())
    cv     = std_z / (abs(mean_z) + 1e-8)
    return CheckResult(
        name='T13_depth_consistent',
        passed=cv < 0.35,
        detail=(f'cam_t_z  mean={mean_z:.2f} m  std={std_z:.2f} m  CV={cv:.2f}'
                f'  (need CV<0.35; large CV → model predicts inconsistent scale)'),
    )


def check_pelvis_projects_to_person(joints: np.ndarray, cam_t: np.ndarray,
                                    dets: np.ndarray,
                                    frame_w: int, frame_h: int) -> CheckResult:
    """T14: Pelvis projected through cam_t must land near the detection bbox centre.

    This is the definitive 'mesh is on the person' check. If it fails the
    camera translation is wrong regardless of how the renderer uses it.
    Failure here while T10 passes means the 2D joints are internally consistent
    (cam_t + projection agree) but the absolute value of cam_t is wrong.

    joints: [N, J, 3] body-local joints in metres (joints3d_smpl convention)
    cam_t:  [N, 3] camera translation (metres)
    dets:   [N, 5]  cx_px, cy_px, scale_px, _, score
    """
    fl = (frame_w**2 + frame_h**2) ** 0.5
    ok = 0
    errors_u, errors_v = [], []
    for j, ct, det in zip(joints, cam_t, dets):
        pelvis = j[PELVIS]
        depth  = pelvis[2] + ct[2]
        if depth <= 0:
            continue
        u = frame_w / 2 + fl * (pelvis[0] + ct[0]) / depth
        v = frame_h / 2 + fl * (pelvis[1] + ct[1]) / depth
        bbox_px = det[2]
        eu, ev = abs(u - det[0]), abs(v - det[1])
        errors_u.append(eu)
        errors_v.append(ev)
        if eu < 0.35 * bbox_px and ev < 0.35 * bbox_px:
            ok += 1
    n = max(len(joints), 1)
    frac_ok = ok / n
    mean_eu = float(np.mean(errors_u)) if errors_u else 0.0
    mean_ev = float(np.mean(errors_v)) if errors_v else 0.0
    return CheckResult(
        name='T14_pelvis_on_person',
        passed=frac_ok >= 0.70,
        detail=(f'{frac_ok*100:.1f}% frames: pelvis projects within 35% of bbox '
                f'(mean err u={mean_eu:.0f}px v={mean_ev:.0f}px; need ≥70%)'),
    )


def check_scale_plausibility(cam_t: np.ndarray, dets: np.ndarray,
                              frame_w: int, frame_h: int) -> CheckResult:
    """T15: Back-computed scale s = 2*fl/(bbox_h * cam_t_z) must be in [0.3, 8].

    s is the raw scale output from the camera head.  Values outside [0.3, 8]
    are physically impossible for a person-sized detection and indicate the
    regressor is not learning the scale at all.

    Complements T13 (depth CV): T13 catches frame-to-frame inconsistency,
    T15 catches systematic bias (e.g. always predicting scale → 0).
    """
    fl = (frame_w**2 + frame_h**2) ** 0.5
    scales = []
    for ct, det in zip(cam_t, dets):
        tz, bbox_h = ct[2], det[2]
        if tz > 0 and bbox_h > 0:
            scales.append(2 * fl / (bbox_h * tz))
    if not scales:
        return CheckResult('T15_scale_plausible', False, 'no valid frames to compute scale')
    sc = np.array(scales)
    frac_ok = float(np.mean((sc > 0.3) & (sc < 8.0)))
    return CheckResult(
        name='T15_scale_plausible',
        passed=frac_ok >= 0.80,
        detail=(f'back-computed s: mean={sc.mean():.2f} std={sc.std():.2f} '
                f'range=[{sc.min():.2f},{sc.max():.2f}]; '
                f'{frac_ok*100:.1f}% in [0.3,8] (need ≥80%)'),
    )


def run_cache_checks(cache_dir: str, cfg, device: torch.device,
                     n_samples: int = 50) -> List[CheckResult]:
    """Stage-1 pipeline checks: T16/T17/T18 on the pre-converted MHR cache.

    T16 — body heights from MHR forward pass are in [0.8, 2.5] m
    T17 — lbs_model_params have sufficient variance (not collapsed)
    T18 — root rotation varies across samples (global orientation predicted)
    """
    import glob
    npz_files = sorted(glob.glob(os.path.join(cache_dir, '*_mhr_params.npz')))
    if not npz_files:
        msg = f'No *_mhr_params.npz found in {cache_dir}'
        return [CheckResult(n, False, msg)
                for n in ('T16_cache_plausible', 'T17_cache_diversity',
                          'T18_cache_global_orient_varies')]

    npz_path = npz_files[0]
    print(f'  Cache file: {os.path.basename(npz_path)}')
    data    = np.load(npz_path)
    identity   = data['identity_coeffs']    # [N, 45]
    lbs_params = data['lbs_model_params']   # [N, 204]
    face_expr  = data.get('face_expr_coeffs', None)
    n_total    = len(identity)
    print(f'  Cache samples: {n_total}')

    # T17: diversity — no MHR model needed
    lbs_std = float(lbs_params.std(axis=0).mean())
    t17 = CheckResult(
        name='T17_cache_diversity',
        passed=lbs_std > 0.05,
        detail=(f'lbs_model_params mean-std={lbs_std:.4f} over {n_total} samples '
                f'(need >0.05; low → collapsed to single pose)'),
    )

    # T18: root rotation variance
    root_rot_std = float(lbs_params[:, :3].std(axis=0).mean())
    t18 = CheckResult(
        name='T18_cache_global_orient_varies',
        passed=root_rot_std > 0.3,
        detail=(f'root rotation (lbs[:3]) std={root_rot_std:.3f} '
                f'(need >0.3; low → global orientation frozen — camera space wrong)'),
    )

    # T16: body height — needs MHR forward pass
    mhr_model_path = os.path.join(_ROOT, cfg.MHR.MODEL_PT)
    try:
        mhr_model = torch.load(mhr_model_path, map_location='cpu', weights_only=False)
        mhr_model.eval().to(device)
    except Exception as e:
        t16 = CheckResult('T16_cache_plausible', False, f'Cannot load MHR model: {e}')
        return [t16, t17, t18]

    indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    heights = []
    with torch.no_grad():
        for i in indices:
            id_t  = torch.from_numpy(identity[i:i+1]).float().to(device)
            lbs_t = torch.from_numpy(lbs_params[i:i+1]).float().to(device)
            expr_t = (torch.from_numpy(face_expr[i:i+1]).float().to(device)
                      if face_expr is not None
                      else torch.zeros(1, 72, device=device))
            try:
                verts_cm, _ = mhr_model(identity_coeffs=id_t,
                                        model_parameters=lbs_t,
                                        face_expr_coeffs=expr_t,
                                        apply_correctives=True)
                verts_m = verts_cm[0].cpu().numpy() * 0.01
                heights.append(verts_m[:, 1].max() - verts_m[:, 1].min())
            except Exception:
                pass

    if not heights:
        t16 = CheckResult('T16_cache_plausible', False, 'MHR forward failed on all samples')
    else:
        h = np.array(heights)
        frac_ok = float(np.mean((h > 0.8) & (h < 2.5)))
        t16 = CheckResult(
            name='T16_cache_plausible',
            passed=frac_ok >= 0.80,
            detail=(f'body height: mean={h.mean():.2f}m std={h.std():.2f}m '
                    f'range=[{h.min():.2f},{h.max():.2f}]; '
                    f'{frac_ok*100:.1f}% in [0.8,2.5]m (need ≥80%)'),
        )
    return [t16, t17, t18]


def dump_frame_diagnostics(out_dir: str, frames: list, joints: np.ndarray,
                           cam_t: np.ndarray, dets: np.ndarray,
                           j2d: Optional[np.ndarray] = None) -> None:
    """Save per-frame debug images with bbox, joints2d, and projected pelvis.

    All output is written to files — no display window is opened.
    Headless-safe (only uses cv2.imwrite, no cv2.imshow).

    Annotations:
      Green rectangle  — detection bbox
      Blue dots        — predicted joints2d_smpl (if available)
      Magenta dot      — projected pelvis via cam_t
      Green ring       — bbox centre
      Yellow line      — error vector from bbox centre to projected pelvis
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W = frames[0].shape[:2]
    fl   = (W**2 + H**2) ** 0.5

    saved = 0
    for i, (frame, j, ct, det) in enumerate(zip(frames, joints, cam_t, dets)):
        vis = frame.copy()   # RGB uint8

        cx_px   = float(det[0])
        cy_px   = float(det[1])
        bbox_px = float(det[2])

        # Detection bbox (green)
        half = bbox_px / 2
        cv2.rectangle(vis,
                      (int(cx_px - half), int(cy_px - half)),
                      (int(cx_px + half), int(cy_px + half)),
                      (0, 255, 0), 2)

        # Predicted joints2d (blue dots)
        if j2d is not None and i < len(j2d):
            for jx, jy in j2d[i]:
                cv2.circle(vis, (int(jx), int(jy)), 3, (0, 100, 255), -1)

        # Projected pelvis via cam_t (magenta) and error line (yellow)
        pelvis = j[PELVIS]
        depth  = pelvis[2] + ct[2]
        if depth > 0:
            u = W / 2 + fl * (pelvis[0] + ct[0]) / depth
            v = H / 2 + fl * (pelvis[1] + ct[1]) / depth
            cv2.circle(vis, (int(cx_px), int(cy_px)), 8, (0, 255, 0), 2)
            cv2.circle(vis, (int(u), int(v)), 8, (255, 0, 255), -1)
            cv2.line(vis, (int(cx_px), int(cy_px)), (int(u), int(v)), (255, 255, 0), 2)
            err = ((u - cx_px)**2 + (v - cy_px)**2) ** 0.5
            cv2.putText(vis, f'pelvis err={err:.0f}px',
                        (10, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # cam_t annotation
        cv2.putText(vis,
                    f'cam_t=[{ct[0]:+.2f},{ct[1]:+.2f},{ct[2]:+.2f}] m',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        out_path = os.path.join(out_dir, f'frame_{i:03d}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f'  Saved {saved} diagnostic frames → {out_dir}/')


def print_side_by_side_diag(mhrd_cam_t, mhrd_joints,
                            ref_cam_t=None, ref_joints=None):
    """T11 enhanced: side-by-side cam_t and arm angle diagnostics.

    Prints camera translation and per-frame elbow angles for whichever
    models are available.
    """
    print('\nSide-by-side diagnostics:')
    header = f'{"Frame":>6s}'
    if mhrd_cam_t is not None:
        header += '  ' + 'MHRD cam_t'.center(22) + '  ' + 'MHRD elbow_avg'.center(14)
    if ref_cam_t is not None and ref_joints is not None:
        header += '  ' + 'Ref cam_t'.center(22) + '  ' + 'Ref elbow_avg'.center(14)
    print(header)

    n = 0
    if mhrd_joints is not None:
        n = max(n, len(mhrd_joints))
    if ref_joints is not None:
        n = max(n, len(ref_joints))

    for i in range(n):
        row = f'{i:6d}'
        if mhrd_cam_t is not None and i < len(mhrd_cam_t):
            ct = mhrd_cam_t[i]
            row += f'  [{ct[0]:+7.2f}, {ct[1]:+7.2f}, {ct[2]:+7.2f}]'
            # Compute avg elbow angle for this frame
            angles = []
            for sh, el, wr in [(L_SHOULDER, L_ELBOW, L_WRIST),
                               (R_SHOULDER, R_ELBOW, R_WRIST)]:
                upper = mhrd_joints[i, el] - mhrd_joints[i, sh]
                lower = mhrd_joints[i, wr] - mhrd_joints[i, el]
                dot = np.dot(upper, lower)
                nm = np.linalg.norm(upper) * np.linalg.norm(lower) + 1e-8
                ang = np.arccos(np.clip(dot / nm, -1, 1)) * 180 / np.pi
                angles.append(ang)
            avg_elbow = float(np.mean(angles))
            row += f'  {avg_elbow:11.1f}°'
        else:
            if mhrd_cam_t is not None:
                row += '  ' + ' ' * 22 + '  ' + ' ' * 14
        if ref_cam_t is not None and ref_joints is not None and i < len(ref_cam_t):
            ct = ref_cam_t[i]
            row += f'  [{ct[0]:+7.2f}, {ct[1]:+7.2f}, {ct[2]:+7.2f}]'
            angles = []
            for sh, el, wr in [(L_SHOULDER, L_ELBOW, L_WRIST),
                               (R_SHOULDER, R_ELBOW, R_WRIST)]:
                upper = ref_joints[i, el] - ref_joints[i, sh]
                lower = ref_joints[i, wr] - ref_joints[i, el]
                dot = np.dot(upper, lower)
                nm = np.linalg.norm(upper) * np.linalg.norm(lower) + 1e-8
                ang = np.arccos(np.clip(dot / nm, -1, 1)) * 180 / np.pi
                angles.append(ang)
            avg_elbow = float(np.mean(angles))
            row += f'  {avg_elbow:11.1f}°'
        print(row)


def check_vs_reference(mhrd_joints: np.ndarray,
                       ref_joints:  np.ndarray) -> CheckResult:
    """T6: Pelvis-centred MPJPE < 200 mm vs original SMPL demo output.

    Both arrays: [N_frames, N_joints, 3], metres.
    """
    mhrd_c = mhrd_joints - mhrd_joints[:, PELVIS:PELVIS+1, :]
    ref_c  = ref_joints  - ref_joints[:,  PELVIS:PELVIS+1, :]

    # Align scales (handles unit discrepancy between models)
    mhrd_scale = np.sqrt((mhrd_c ** 2).sum(axis=(1, 2), keepdims=True)) + 1e-8
    ref_scale  = np.sqrt((ref_c  ** 2).sum(axis=(1, 2), keepdims=True)) + 1e-8
    mhrd_scaled = mhrd_c / mhrd_scale * ref_scale

    mpjpe_mm = float(np.linalg.norm(mhrd_scaled - ref_c, axis=-1).mean()) * 1000
    return CheckResult(
        name='T6_vs_reference',
        passed=mpjpe_mm < 200.0,
        detail=f'Centred MPJPE = {mpjpe_mm:.1f} mm (need <200 mm)',
    )


# ─── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, n: int) -> List[np.ndarray]:
    """Return N evenly-spaced RGB frames from a video file."""
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), n, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f'Could not read any frames from {video_path}')
    return frames


# ─── MHRD model inference wrapper ─────────────────────────────────────────────

class MHRDInference:
    """Minimal wrapper around MHRHMR for per-frame inference."""

    def __init__(self, ckpt_path: str, cfg_path: str, device: torch.device):
        from train.core.config import update_hparams
        from train.core import constants
        from torchvision.transforms import Normalize
        from mhr_hmr import MHRHMR

        self.device = device
        self.cfg    = update_hparams(cfg_path)
        self.norm   = Normalize(mean=constants.IMG_NORM_MEAN,
                                std=constants.IMG_NORM_STD)

        mhr_model_path = os.path.join(_ROOT, self.cfg.MHR.MODEL_PT)
        mhr_model = torch.load(mhr_model_path, map_location='cpu',
                               weights_only=False)
        mhr_model.eval()

        self.model = MHRHMR(
            backbone=self.cfg.MODEL.BACKBONE,
            img_res=self.cfg.DATASET.IMG_RES,
            pretrained_ckpt=self.cfg.TRAINING.PRETRAINED_CKPT,
            hparams=self.cfg,
            mhr_model=mhr_model,
        ).to(device)

        if ckpt_path:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd    = state.get('state_dict', state)
            stripped = {k[len('model.'):]: v for k, v in sd.items()
                        if k.startswith('model.')}
            result = self.model.load_state_dict(stripped, strict=False)
            print(f'[MHRDInference] loaded ckpt — '
                  f'missing={len(result.missing_keys)}, '
                  f'unexpected={len(result.unexpected_keys)}')
        else:
            warnings.warn('No --ckpt supplied — running with random weights')
        self.model.eval()

    @torch.no_grad()
    def infer(self, frame_rgb: np.ndarray,
              dets: np.ndarray):
        """Run MHRD inference on a single frame.

        frame_rgb: HxWx3 uint8 RGB
        dets:      [M, 5] — cx, cy, scale_px, unused, score  (MPT format)
        Returns: tuple of (joints3d, pred_pose, vertices, joints2d, pred_cam_t)
                 each [M, ...] or None, or None overall if no detections.
        """
        from train.utils.image_utils import crop

        if len(dets) == 0:
            return None

        H, W    = frame_rgb.shape[:2]
        B       = len(dets)
        img_res = self.cfg.DATASET.IMG_RES

        bbox_center = torch.tensor([[d[0], d[1]] for d in dets],
                                   device=self.device)
        bbox_scale  = torch.tensor([d[2] / 200.0 for d in dets],
                                   device=self.device)
        img_h = torch.full((B,), H, device=self.device, dtype=torch.float)
        img_w = torch.full((B,), W, device=self.device, dtype=torch.float)

        centers_np = bbox_center.cpu().numpy()
        scales_np  = bbox_scale.cpu().numpy()
        crops = np.stack([
            np.transpose(
                crop(frame_rgb, centers_np[i], scales_np[i],
                     [img_res, img_res]).astype('float32'),
                (2, 0, 1)
            ) / 255.0
            for i in range(B)
        ])
        inp = self.norm(torch.from_numpy(crops).to(self.device))

        out, *_ = self.model(inp, bbox_center=bbox_center,
                             bbox_scale=bbox_scale,
                             img_w=img_w, img_h=img_h)

        j = out.get('joints3d_smpl')
        if j is None:
            j = out.get('joints3d')
        if j is None:
            return None

        p   = out.get('pred_pose')       # [B, D] LBS params
        v   = out.get('vertices')        # [B, V, 3] mesh vertices
        j2d = out.get('joints2d_smpl')   # [B, 24, 2] SMPL-ordered pixel coords
        ct  = out.get('pred_cam_t')      # [B, 3] full perspective cam translation

        return (j.detach().cpu().numpy(),
                p.detach().cpu().numpy() if p is not None else None,
                v.detach().cpu().numpy() if v is not None else None,
                j2d.detach().cpu().numpy() if j2d is not None else None,
                ct.detach().cpu().numpy() if ct is not None else None)


# ─── Original (SMPL+MHR optimization) reference wrapper ──────────────────────

class OriginalInference:
    """Wraps the original Tester and extracts SMPL-mapped joints from
    mhr_skel_state for comparison with joints3d_smpl.

    skel_state layout: [B, 127, 16] — column-major 4×4, translation at [12:15],
    in centimetres.  Mapped to SMPL-22 using MHR_TO_SMPL_JOINT_INDICES[:22].
    """

    def __init__(self, ckpt_path: str, device: torch.device):
        import types
        from train.core.tester import Tester
        from mhr_constants import MHR_TO_SMPL_JOINT_INDICES

        self.smpl_idx = MHR_TO_SMPL_JOINT_INDICES[:22]   # first 22 = body

        # Tester.__init__ expects an argparse Namespace with these fields.
        args = types.SimpleNamespace(
            cfg=os.path.join(_ROOT, 'configs', 'dpose_conf.yaml'),
            ckpt=ckpt_path,
            tracker_batch_size=1,
            detector='yolo',
            yolo_img_size=416,
        )
        self.tester = Tester(args)

    def infer(self, frame_rgb: np.ndarray,
              dets: np.ndarray):
        """Returns (joints_m, pred_cam_t) — joints [M, 22, 3] in metres,
        pred_cam_t [M, 3] or None."""
        if len(dets) == 0:
            return None
        result = self.tester.run_on_single_image_tensor(frame_rgb, [dets])
        if result is None:
            return None
        skel = result.get('mhr_skel_state')
        if skel is None:
            return None
        if isinstance(skel, torch.Tensor):
            skel = skel.detach().cpu().numpy()
        # Extract XYZ translation: column 12-14 in column-major 4×4
        trans_cm = skel[:, :, 12:15]          # [B, 127, 3]
        joints_m = trans_cm[:, self.smpl_idx, :] / 100.0  # cm → m, [B, 22, 3]

        ct = result.get('pred_cam_t')
        if ct is not None and isinstance(ct, torch.Tensor):
            ct = ct.detach().cpu().numpy()
        return (joints_m, ct)


# ─── Detection helper ─────────────────────────────────────────────────────────

def _build_mot(device: torch.device):
    """Build an MPT detector, falling back from yolo → maskrcnn.

    yolov3 requires a pre-compiled numpy==1.14.4 wheel which can't be built on
    Python 3.12.  maskrcnn uses torchvision directly and is always available.
    """
    from multi_person_tracker import MPT
    for dtype in ('yolo', 'maskrcnn'):
        try:
            mot = MPT(device=device, batch_size=4, display=False,
                      detector_type=dtype, output_format='dict',
                      yolo_img_size=416)
            print(f'  Detector: {dtype}')
            return mot
        except (ImportError, ModuleNotFoundError):
            print(f'  Detector {dtype} unavailable, trying next …')
    raise RuntimeError('No working MPT detector found (tried yolo, maskrcnn)')


def detect_people(frames: List[np.ndarray],
                  device: torch.device) -> List[np.ndarray]:
    """Run person detection on a list of RGB frames.

    Returns a list of [M, 5] arrays (cx, cy, scale, _, score).
    """
    mot = _build_mot(device)
    detections = []
    for frame in frames:
        t = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
        raw = mot.detector(t.to(device))
        if raw:
            boxes  = torch.cat([p['boxes']  for p in raw], dim=0)
            scores = torch.cat([p['scores'] for p in raw], dim=0)
            mask   = scores > 0.7
            boxes, scores = boxes[mask], scores[mask]
            dets_np = torch.cat([boxes, scores.unsqueeze(1)],
                                dim=1).detach().cpu().numpy()
        else:
            dets_np = np.empty((0, 5))
        prep = mot.prepare_output_detections([dets_np])
        detections.append(prep[0] if prep and len(prep[0]) > 0
                          else np.empty((0, 5)))
    return detections


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_tests(args) -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 1. Extract frames ────────────────────────────────────────────────────
    print(f'\nExtracting {args.frames} frames from {args.input} …')
    frames = extract_frames(args.input, args.frames)
    print(f'  Got {len(frames)} frames  ({frames[0].shape[1]}×{frames[0].shape[0]})')

    # ── 2. Detect people ─────────────────────────────────────────────────────
    print('Running person detection …')
    dets_list = detect_people(frames, device)
    n_frames_with_people = sum(1 for d in dets_list if len(d) > 0)
    print(f'  Frames with detections: {n_frames_with_people}/{len(frames)}')
    if n_frames_with_people < 3:
        print('  ERROR: too few detections — try a different input video')
        return 2

    # ── 3. MHRD inference ────────────────────────────────────────────────────
    print(f'\nLoading MHRD model ({args.ckpt or "random weights"}) …')
    mhrd = MHRDInference(args.ckpt, args.cfg, device)

    mhrd_joints_all   = []
    mhrd_poses_all    = []
    mhrd_verts_all    = []
    mhrd_j2d_all      = []
    mhrd_camt_all     = []
    mhrd_dets_all     = []
    mhrd_frames_all   = []   # raw frames aligned with the above (for visual dump)
    for frame, dets in zip(frames, dets_list):
        result = mhrd.infer(frame, dets)
        if result is None:
            continue
        j, p, v, j2d, ct = result
        if j is not None and len(j) > 0:
            mhrd_joints_all.append(j[0])
            if p is not None:
                mhrd_poses_all.append(p[0])
            if v is not None:
                mhrd_verts_all.append(v[0])
            if j2d is not None:
                mhrd_j2d_all.append(j2d[0])
            if ct is not None:
                mhrd_camt_all.append(ct[0])
            mhrd_dets_all.append(dets[0])
            mhrd_frames_all.append(frame)

    if len(mhrd_joints_all) < 3:
        print('ERROR: MHRD model produced too few joint outputs — aborting')
        return 2

    mhrd_joints  = np.stack(mhrd_joints_all)   # [N, J, 3]
    mhrd_poses   = np.stack(mhrd_poses_all) if mhrd_poses_all else None
    mhrd_verts   = np.stack(mhrd_verts_all)  if mhrd_verts_all  else None
    mhrd_j2d     = np.stack(mhrd_j2d_all)    if mhrd_j2d_all    else None
    mhrd_camt    = np.stack(mhrd_camt_all)   if mhrd_camt_all   else None
    mhrd_dets    = np.stack(mhrd_dets_all)   if mhrd_dets_all   else None
    print(f'  Collected {len(mhrd_joints)} MHRD joint frames, '
          f'shape {mhrd_joints.shape}, '
          f'range [{mhrd_joints.min():.3f}, {mhrd_joints.max():.3f}] m')
    if mhrd_poses is not None:
        print(f'  pred_pose shape {mhrd_poses.shape}, '
              f'mean-std across frames = {mhrd_poses.std(axis=0).mean():.4f}')

    # ── 4. Reference inference (optional) ────────────────────────────────────
    ref_joints = None
    ref_camt   = None
    if args.ref_ckpt:
        print(f'\nLoading original reference model ({args.ref_ckpt}) …')
        try:
            orig = OriginalInference(args.ref_ckpt, device)
            ref_j_all  = []
            ref_ct_all = []
            for frame, dets in zip(frames, dets_list):
                result = orig.infer(frame, dets)
                if result is None:
                    continue
                j, ct = result
                if j is not None and len(j) > 0:
                    ref_j_all.append(j[0])
                    if ct is not None:
                        ref_ct_all.append(ct[0])
            if ref_j_all:
                ref_joints = np.stack(ref_j_all)
                ref_camt   = np.stack(ref_ct_all) if ref_ct_all else None
                print(f'  Collected {len(ref_joints)} reference joint frames')
            else:
                print('  WARNING: reference model produced no output — T6 skipped')
                ref_joints = None
                ref_camt   = None
        except Exception as e:
            print(f'  WARNING: could not load reference model ({e}) — T6 skipped')
            ref_joints = None
            ref_camt   = None

    # ── 5. Run checks ─────────────────────────────────────────────────────────
    print('\n' + '─' * 72)
    print('Test results')
    print('─' * 72)

    results: List[CheckResult] = [
        check_not_upside_down(mhrd_joints),
        check_faces_camera(mhrd_joints),
        check_joint_range(mhrd_joints),
        check_arm_responsive(mhrd_joints),
        check_rotation_tracks(mhrd_joints),
        check_elbow_angle_varies(mhrd_joints),
        check_pred_pose_varies(mhrd_poses),
        check_elbow_not_tpose(mhrd_joints),
    ]
    if mhrd_j2d is not None and mhrd_dets is not None:
        n_j2d = min(len(mhrd_j2d), len(mhrd_dets))
        results.append(check_joints2d_near_bbox(mhrd_j2d[:n_j2d], mhrd_dets[:n_j2d]))
    else:
        results.append(CheckResult('T10_joints2d_near_bbox', False,
                                   'joints2d/dets not available from model output'))
    if mhrd_verts is not None:
        n_v = min(len(mhrd_joints), len(mhrd_verts))
        results.append(check_vertices_orientation(mhrd_joints[:n_v], mhrd_verts[:n_v]))
    else:
        results.append(CheckResult('T9_vertices_not_upside_down', False,
                                   'vertices not in model output — check output keys'))
    results.append(check_depth_consistent(mhrd_camt))
    # T14/T15 — inference-level cam_t quality checks
    if mhrd_camt is not None and mhrd_dets is not None:
        fh, fw = frames[0].shape[:2]
        n_ct = min(len(mhrd_joints), len(mhrd_camt), len(mhrd_dets))
        results.append(check_pelvis_projects_to_person(
            mhrd_joints[:n_ct], mhrd_camt[:n_ct], mhrd_dets[:n_ct], fw, fh))
        results.append(check_scale_plausibility(
            mhrd_camt[:n_ct], mhrd_dets[:n_ct], fw, fh))
    else:
        for name in ('T14_pelvis_on_person', 'T15_scale_plausible'):
            results.append(CheckResult(name, False,
                                       'pred_cam_t or dets not available'))
    if ref_joints is not None:
        n_common = min(len(mhrd_joints), len(ref_joints))
        results.append(check_vs_reference(mhrd_joints[:n_common],
                                          ref_joints[:n_common]))

    for r in results:
        print(r)

    # ── 6. Side-by-side diagnostics (T11 enhanced) ────────────────────────────
    print()
    print_side_by_side_diag(mhrd_camt, mhrd_joints,
                            ref_camt, ref_joints)

    # ── 7. Diagnostic dump ────────────────────────────────────────────────────
    print('\nDiagnostics (first frame, first person):')
    j0 = mhrd_joints[0]
    print(f'  pelvis      : {j0[PELVIS]}')
    print(f'  head        : {j0[HEAD]}')
    print(f'  left_shoulder : {j0[L_SHOULDER]}')
    print(f'  right_shoulder: {j0[R_SHOULDER]}')
    spine  = j0[NECK] - j0[PELVIS]
    sh_dir = j0[R_SHOULDER] - j0[L_SHOULDER]
    chest_n = np.cross(spine, sh_dir)
    chest_n_norm = chest_n / (np.linalg.norm(chest_n) + 1e-8)
    print(f'  chest normal (norm): {chest_n_norm}  '
          f'(z<0 → faces camera, z>0 → faces away)')
    head_above = j0[HEAD, 1] < j0[PELVIS, 1]
    print(f'  head Y < pelvis Y: {head_above}  '
          f'(True → right-side-up, False → upside down)')

    # ── 8. Cache checks (T16/T17/T18) ────────────────────────────────────────
    if args.cache_dir:
        print('\n' + '─' * 72)
        print('Cache checks (preconvert stage)')
        print('─' * 72)
        from train.core.config import update_hparams
        cfg_for_cache = update_hparams(args.cfg)
        cache_results = run_cache_checks(args.cache_dir, cfg_for_cache, device)
        for r in cache_results:
            print(r)
        results.extend(cache_results)

    # ── 9. Visual frame dump (headless-safe) ─────────────────────────────────
    if args.out_dir and mhrd_camt is not None and mhrd_dets is not None:
        print(f'\nSaving diagnostic frames → {args.out_dir}/')
        n_dump = min(len(mhrd_frames_all), len(mhrd_joints),
                     len(mhrd_camt), len(mhrd_dets))
        dump_frame_diagnostics(
            args.out_dir,
            mhrd_frames_all[:n_dump],
            mhrd_joints[:n_dump],
            mhrd_camt[:n_dump],
            mhrd_dets[:n_dump],
            j2d=mhrd_j2d[:n_dump] if mhrd_j2d is not None else None,
        )

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print('\n' + '─' * 72)
    n_pass = sum(r.passed for r in results)
    n_total = len(results)
    print(f'Result: {n_pass}/{n_total} checks passed')

    failed = [r for r in results if not r.passed]
    if failed:
        print('\nFailed checks:')
        for r in failed:
            print(f'  • {r.name}: {r.detail}')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare MHRD-Pose demo against original demo on a test video')
    parser.add_argument('--input', required=True,
                        help='Path to test video (mp4/avi/mov) or image folder')
    parser.add_argument('--ckpt', default='',
                        help='MHRD-Pose checkpoint (.ckpt). '
                             'Omit to test with random weights.')
    parser.add_argument('--cfg',
                        default=os.path.join(_THIS_DIR, 'config_mhr.yaml'),
                        help='MHRD config yaml (default: MHRD-Pose/config_mhr.yaml)')
    parser.add_argument('--ref-ckpt', default='',
                        help='Original demo checkpoint for T6 reference comparison')
    parser.add_argument('--frames', type=int, default=20,
                        help='Number of frames to sample from the video (default: 20)')
    parser.add_argument('--out-dir', default=None, dest='out_dir',
                        help='Directory for per-frame diagnostic images (headless-safe). '
                             'Omit to skip image output.')
    parser.add_argument('--cache-dir', default=None, dest='cache_dir',
                        help='Path to preconvert MHR cache directory (runs T16/T17/T18). '
                             'E.g. data/mhr_cache')
    args = parser.parse_args()

    sys.exit(run_tests(args))
