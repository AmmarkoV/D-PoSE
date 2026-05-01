"""
dual_demo_webcam.py

Runs D-PoSE (with MHR conversion) and MHRD-Pose on each frame side-by-side.
Displays:
  - Rendered mesh from each algorithm
  - Person bounding boxes
  - Skeleton joints projected onto the image
  - Raw MHR pose vectors overlaid as text (and printed to console)

Imports shared utilities directly from the two demo_webcam.py scripts.
"""

import os
import sys
import argparse
import contextlib

import torch
import numpy as np
import cv2
from loguru import logger

# ── Path setup ───────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MHRD_DIR = os.path.join(_THIS_DIR, 'MHRD-Pose')
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _MHRD_DIR)
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ── Import from both demo_webcam.py files (no re-implementation) ─────────────
import importlib.util as _ilu


def _load_demo(path, name):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dpose_dw = _load_demo(os.path.join(_THIS_DIR, 'demo_webcam.py'), 'dpose_demo_webcam')
_mhrd_dw  = _load_demo(os.path.join(_MHRD_DIR,  'demo_webcam.py'), 'mhrd_demo_webcam')

# Shared utilities — identical in both scripts, use D-PoSE version
getCaptureDeviceFromPath = _dpose_dw.getCaptureDeviceFromPath
scale_and_embed_frame    = _dpose_dw.scale_and_embed_frame

# Algorithm-specific output encoders
encode_dpose_output = _dpose_dw.encode_mhr_output_to_dict   # D-PoSE mhr_output
encode_mhrd_output  = _mhrd_dw.encode_mhr_output_to_dict    # MHRD-Pose mhr_output

# Tester classes
from train.core.tester import Tester as DPoseTester          # noqa: E402
MHRDTester = _mhrd_dw.MHRTester

# MHR skeleton meta
from mhr_constants import MHR_TO_SMPL_JOINT_INDICES, MHR_JOINT_NAMES  # noqa: E402

# Person detector (shared between both algorithms)
from multi_person_tracker import MPT  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
SMPL_JOINT_NAMES = [
    'pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee',
    'spine2', 'L_ankle', 'R_ankle', 'spine3', 'L_foot', 'R_foot',
    'neck', 'L_collar', 'R_collar', 'head',
    'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist', 'L_hand', 'R_hand',
]

# Bone pairs (SMPL joint indices)
SMPL_SKELETON = [
    (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (9, 14), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21),
    (20, 22), (21, 23),
]

IMG_W, IMG_H = 1280, 720
# focal length matching renderer_pyrd and MHRD-Pose head convention
FOCAL_LENGTH = float(np.sqrt(IMG_W ** 2 + IMG_H ** 2))   # ≈ 1468.6


# ── imshow capture context manager ───────────────────────────────────────────
@contextlib.contextmanager
def capture_imshow():
    """Intercept cv2.imshow and store the last frame per window name."""
    captured = {}
    orig = cv2.imshow

    def _mock(name, img):
        captured[name] = img.copy()

    cv2.imshow = _mock
    try:
        yield captured
    finally:
        cv2.imshow = orig


# ── Visualization helpers ─────────────────────────────────────────────────────
def draw_bboxes(frame_bgr, detections, color=(0, 255, 0)):
    """Overlay bounding boxes. Each detection: [cx, cy, bbox_height_px, ...]."""
    out = frame_bgr.copy()
    for det in detections:
        cx, cy, h = float(det[0]), float(det[1]), float(det[2])
        w = h
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (int(cx), int(cy)), 5, color, -1)
        cv2.putText(out, f'h={h:.0f}px', (x1, y1 - 6),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1, cv2.LINE_AA)
    return out


def draw_joints_2d(frame_bgr, joints2d, label_joints=True,
                   joint_color=(0, 255, 255), bone_color=(255, 128, 0)):
    """
    Draw 2D skeleton (bones + joints) on frame_bgr.
    joints2d: [J, 2] pixel coords (x, y), J <= 24
    """
    out = frame_bgr.copy()
    J = len(joints2d)

    for (j1, j2) in SMPL_SKELETON:
        if j1 >= J or j2 >= J:
            continue
        p1 = (int(joints2d[j1, 0]), int(joints2d[j1, 1]))
        p2 = (int(joints2d[j2, 0]), int(joints2d[j2, 1]))
        in_bounds = lambda p: 0 <= p[0] < frame_bgr.shape[1] and 0 <= p[1] < frame_bgr.shape[0]
        if in_bounds(p1) and in_bounds(p2):
            cv2.line(out, p1, p2, bone_color, 2, cv2.LINE_AA)

    for j in range(J):
        xi, yi = int(joints2d[j, 0]), int(joints2d[j, 1])
        if not (0 <= xi < frame_bgr.shape[1] and 0 <= yi < frame_bgr.shape[0]):
            continue
        cv2.circle(out, (xi, yi), 5, joint_color, -1, cv2.LINE_AA)
        if label_joints and j < len(SMPL_JOINT_NAMES):
            cv2.putText(out, SMPL_JOINT_NAMES[j], (xi + 6, yi - 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, joint_color, 1, cv2.LINE_AA)
    return out


def project_yup_joints(joints_m, fl=FOCAL_LENGTH, cx=IMG_W / 2, cy=IMG_H / 2):
    """
    Project Y-up camera-space 3D joints [J, 3] (metres) to image pixels [J, 2].
    Convention: +X right, +Y up, +Z into scene.
    Image formula: u = cx + fl*x/z,  v = cy - fl*y/z  (negate Y for Y-up→screen)
    """
    z = joints_m[:, 2].copy()
    z = np.where(np.abs(z) < 0.05, np.sign(z) * 0.05 + 0.1, z)
    u = cx + fl * joints_m[:, 0] / z
    v = cy - fl * joints_m[:, 1] / z
    return np.stack([u, v], axis=1)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def overlay_joints_from_dpose(frame_bgr, dpose_output):
    """Draw D-PoSE joints via the shared extract_joints2d_dpose path."""
    j2d = extract_joints2d_dpose(dpose_output)
    if j2d is None:
        return frame_bgr
    return draw_joints_2d(frame_bgr, j2d,
                          joint_color=(0, 255, 200), bone_color=(0, 180, 255))


def overlay_joints_from_mhrd(frame_bgr, mhrd_output):
    """Draw MHRD-Pose joints via the shared extract_joints2d_mhrd path."""
    j2d = extract_joints2d_mhrd(mhrd_output)
    if j2d is None:
        return frame_bgr
    return draw_joints_2d(frame_bgr, j2d,
                          joint_color=(0, 255, 255), bone_color=(255, 128, 0))


# ── Per-frame diagnostics ─────────────────────────────────────────────────────
SMPL_JOINT_NAMES_SHORT = [
    'pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee',
    'spine2', 'L_ankle', 'R_ankle', 'spine3', 'L_foot', 'R_foot',
    'neck', 'L_collar', 'R_collar', 'head',
    'L_sho', 'R_sho', 'L_elb', 'R_elb', 'L_wri', 'R_wri', 'L_hand', 'R_hand',
]

def compute_elbow_angle_joints(j3d):
    """
    Compute elbow bend angles from 3D joints [J, 3].
    Returns (left_elbow_deg, right_elbow_deg) or (nan, nan).
    SMPL indices: L_SHO=16, L_ELB=18, L_WRIST=20; R_SHO=17, R_ELB=19, R_WRIST=21.
    """
    if j3d.shape[0] < 22:
        return float('nan'), float('nan')
    def angle(a, b, c):
        """Angle at joint b formed by a-b-c."""
        v1 = a - b
        v2 = c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-4 or n2 < 1e-4:
            return float('nan')
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        return float(np.degrees(np.arccos(cos_a)))
    l_elbow = angle(j3d[16], j3d[18], j3d[20])
    r_elbow = angle(j3d[17], j3d[19], j3d[21])
    return l_elbow, r_elbow


def extract_smpl_joints3d_dpose(dpose_output):
    """Extract 24 SMPL joints in metres from D-PoSE mhr_skel_state."""
    if dpose_output is None:
        return None
    skel = dpose_output.get('mhr_skel_state')
    if skel is None:
        return None
    skel = _to_numpy(skel)
    if skel.ndim == 2:
        skel = skel[np.newaxis]
    # MHR joint positions are in cm, Y-up
    joints_cm = skel[0][MHR_TO_SMPL_JOINT_INDICES, :3]
    return joints_cm * 0.01  # metres


def extract_smpl_joints3d_mhrd(mhrd_output):
    """Extract 24 SMPL joints in metres from MHRD-Pose output."""
    if mhrd_output is None:
        return None
    j3d = mhrd_output.get('joints3d_smpl')
    if j3d is not None:
        j3d = _to_numpy(j3d)
        if j3d.ndim == 2:
            j3d = j3d[np.newaxis]
        # MHRD-Pose joints3d_smpl are already in metres, Y-down camera frame.
        # Add cam_t to get world position
        camt = mhrd_output.get('pred_cam_t')
        if camt is not None:
            camt = _to_numpy(camt)
            if camt.ndim == 1:
                camt = camt[np.newaxis]
            return (j3d[0] + camt[0]).astype(np.float32)
        return j3d[0].astype(np.float32)
    # Fallback: joints3d + pred_cam_t
    j3d = mhrd_output.get('joints3d')
    camt = mhrd_output.get('pred_cam_t')
    if j3d is not None and camt is not None:
        j3d = _to_numpy(j3d)
        camt = _to_numpy(camt)
        if j3d.ndim == 2:
            j3d = j3d[np.newaxis]
        if camt.ndim == 1:
            camt = camt[np.newaxis]
        return (j3d[0] + camt[0]).astype(np.float32)
    return None


def extract_joints2d_dpose(dpose_output):
    """Project D-PoSE SMPL joints to 2D pixel coords."""
    j3d = extract_smpl_joints3d_dpose(dpose_output)
    if j3d is None:
        return None
    return project_yup_joints(j3d)


def extract_joints2d_mhrd(mhrd_output):
    """Extract 2D SMPL joints from MHRD-Pose output."""
    if mhrd_output is None:
        return None
    j2d = mhrd_output.get('joints2d_smpl')
    if j2d is not None:
        j2d = _to_numpy(j2d)
        if j2d.ndim == 2:
            j2d = j2d[np.newaxis]
        return j2d[0]
    # Fallback: project from joints3d_smpl + pred_cam_t (Y-down)
    j3d_raw = mhrd_output.get('joints3d_smpl')
    camt = mhrd_output.get('pred_cam_t')
    if j3d_raw is not None and camt is not None:
        j3d_raw = _to_numpy(j3d_raw)
        camt = _to_numpy(camt)
        if j3d_raw.ndim == 2:
            j3d_raw = j3d_raw[np.newaxis]
        if camt.ndim == 1:
            camt = camt[np.newaxis]
        xyz = j3d_raw[0] + camt[0]
        z = np.where(np.abs(xyz[:, 2]) < 0.05, 0.05, xyz[:, 2])
        u = IMG_W / 2 + FOCAL_LENGTH * xyz[:, 0] / z
        v = IMG_H / 2 + FOCAL_LENGTH * xyz[:, 1] / z
        return np.stack([u, v], axis=1)
    return None


def print_frame_diag(fn, raw_dets, dpose_output, mhrd_output, frame_bgr):
    """Print per-frame diagnostics comparing D-PoSE and MHRD-Pose."""
    h, w = frame_bgr.shape[:2]
    print(f'\n{"="*70}')
    print(f'  Frame {fn}')
    print(f'{"="*70}')

    # ── Detection info ──
    if raw_dets is not None and len(raw_dets) > 0:
        for i, det in enumerate(raw_dets):
            cx, cy, scale = det[0], det[1], det[2]
            score = det[4] if len(det) > 4 else '?'
            try:
                score = float(score)
                score_s = f'{score:.2f}'
            except (ValueError, TypeError):
                score_s = str(score)
            print(f'  Det#{i}: center=({cx:.0f},{cy:.0f})  height={scale:.0f}px  score={score_s}')
            print(f'         bbox: x=[{cx-scale/2:.0f},{cx+scale/2:.0f}]  y=[{cy-scale/2:.0f},{cy+scale/2:.0f}]  img=({w}x{h})')
    else:
        print('  No detections')
        return

    # ── Camera translation ──
    # D-PoSE uses pred_cam_t from the HMR model (before MHR conversion)
    dpose_camt = None
    mhrd_camt = None

    # D-PoSE: the mhr_output doesn't contain pred_cam_t directly.
    # We need to look at the hmr_output that was used to compute the vertices.
    # Unfortunately it's not in the returned dict. Instead, infer from root position.
    dpose_j3d = extract_smpl_joints3d_dpose(dpose_output)
    mhrd_j3d = extract_smpl_joints3d_mhrd(mhrd_output)

    if dpose_j3d is not None:
        dpose_root = dpose_j3d[0]  # pelvis
        print(f'  D-PoSE  pelvis_3d:  [{dpose_root[0]:+.3f}, {dpose_root[1]:+.3f}, {dpose_root[2]:+.3f}] m (Y-up)')
    if mhrd_j3d is not None:
        mhrd_root = mhrd_j3d[0]
        print(f'  MHRD    pelvis_3d:  [{mhrd_root[0]:+.3f}, {mhrd_root[1]:+.3f}, {mhrd_root[2]:+.3f}] m (Y-down-cam)')

    if dpose_j3d is not None and mhrd_j3d is not None:
        root_diff = np.linalg.norm(dpose_root - mhrd_root)
        print(f'  Root diff: {root_diff:.3f} m')

    # ── Elbow angles ──
    if dpose_j3d is not None and dpose_j3d.shape[0] >= 22:
        l_e, r_e = compute_elbow_angle_joints(dpose_j3d)
        print(f'  D-PoSE  elbows: L={l_e:.1f}°  R={r_e:.1f}°  mean={(l_e+r_e)/2:.1f}°')
    if mhrd_j3d is not None and mhrd_j3d.shape[0] >= 22:
        # MHRD joints are Y-down — flip Y for angle computation (angles are invariant to uniform flip)
        l_e, r_e = compute_elbow_angle_joints(mhrd_j3d)
        print(f'  MHRD    elbows: L={l_e:.1f}°  R={r_e:.1f}°  mean={(l_e+r_e)/2:.1f}°')

    # ── 2D joint comparison ──
    dpose_j2d = extract_joints2d_dpose(dpose_output)
    mhrd_j2d = extract_joints2d_mhrd(mhrd_output)

    if dpose_j2d is not None and mhrd_j2d is not None:
        min_j = min(dpose_j2d.shape[0], mhrd_j2d.shape[0])
        per_joint_diffs = []
        for j in range(min(min_j, 22)):
            diff = np.linalg.norm(dpose_j2d[j] - mhrd_j2d[j])
            name = SMPL_JOINT_NAMES_SHORT[j] if j < len(SMPL_JOINT_NAMES_SHORT) else str(j)
            per_joint_diffs.append((j, name, diff))

        mean_diff = np.mean([d[2] for d in per_joint_diffs])
        max_diff_j = max(per_joint_diffs, key=lambda x: x[2])
        print(f'  2D joint diff (px): mean={mean_diff:.1f}  max={max_diff_j[2]:.1f} @ {max_diff_j[1]}')

        # Show joints with biggest discrepancies
        per_joint_diffs.sort(key=lambda x: x[2], reverse=True)
        top5 = per_joint_diffs[:5]
        parts = ', '.join(f'{j[1]}={j[2]:.0f}px' for j in top5)
        print(f'    worst: {parts}')

        # ── Joints vs bbox center ──
        if raw_dets is not None and len(raw_dets) > 0:
            bcx, bcy = raw_dets[0][0], raw_dets[0][1]
            dpose_j2d_mean_x = np.mean(dpose_j2d[:22, 0])
            dpose_j2d_mean_y = np.mean(dpose_j2d[:22, 1])
            mhrd_j2d_mean_x = np.mean(mhrd_j2d[:22, 0])
            mhrd_j2d_mean_y = np.mean(mhrd_j2d[:22, 1])
            print(f'  Bbox center: ({bcx:.0f}, {bcy:.0f})')
            print(f'  D-PoSE  mean joint2d: ({dpose_j2d_mean_x:.0f}, {dpose_j2d_mean_y:.0f})  '
                  f'dist_to_bbox={np.hypot(dpose_j2d_mean_x-bcx, dpose_j2d_mean_y-bcy):.0f}px')
            print(f'  MHRD    mean joint2d: ({mhrd_j2d_mean_x:.0f}, {mhrd_j2d_mean_y:.0f})  '
                  f'dist_to_bbox={np.hypot(mhrd_j2d_mean_x-bcx, mhrd_j2d_mean_y-bcy):.0f}px')

    # ── Pose vector comparison ──
    dpose_vec = None
    mhrd_vec = None
    if dpose_output is not None:
        p = dpose_output.get('mhr_parameters')
        if p is not None:
            raw = p.get('lbs_model_params')
            if raw is not None:
                dpose_vec = _to_numpy(raw).flatten()
    if mhrd_output is not None:
        raw = mhrd_output.get('pred_pose')
        if raw is not None:
            mhrd_vec = _to_numpy(raw).flatten()

    if dpose_vec is not None and mhrd_vec is not None:
        min_len = min(len(dpose_vec), len(mhrd_vec))
        diff = np.abs(dpose_vec[:min_len] - mhrd_vec[:min_len])
        print(f'  Pose vec: dims D={len(dpose_vec)} M={len(mhrd_vec)}  '
              f'mean_abs_diff={np.mean(diff):.4f}  max_diff={np.max(diff):.4f} @ idx {np.argmax(diff)}')


# ── Pose-vector text panel ────────────────────────────────────────────────────
def make_vector_panel(title, vec, width, height=230, vals_per_row=6, max_rows=10):
    """
    Dark panel (height × width BGR) with the pose vector printed row by row.
    Shows up to vals_per_row * max_rows values.
    """
    panel = np.full((height, width, 3), 25, dtype=np.uint8)

    cv2.putText(panel, title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 60), 1, cv2.LINE_AA)

    if vec is None:
        cv2.putText(panel, 'N/A', (8, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (120, 120, 120), 1)
        return panel

    vals = np.array(vec).flatten()
    y = 46
    for row_i in range(max_rows):
        chunk = vals[row_i * vals_per_row:(row_i + 1) * vals_per_row]
        if len(chunk) == 0:
            break
        # index prefix + values
        idx0 = row_i * vals_per_row
        prefix = f'[{idx0:3d}]'
        nums   = '  '.join(f'{v:+.4f}' for v in chunk)
        line   = f'{prefix}  {nums}'
        cv2.putText(panel, line, (8, y),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (170, 210, 170), 1, cv2.LINE_AA)
        y += 18
        if y > height - 8:
            break
    # dim count
    cv2.putText(panel, f'total dims: {len(vals)}', (8, height - 6),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (100, 160, 100), 1, cv2.LINE_AA)
    return panel


# ── Main display builder ──────────────────────────────────────────────────────
def build_display(dpose_bgr, mhrd_bgr, raw_dets, dpose_output, mhrd_output,
                  frame_number=0):
    """
    Compose the full display frame:
      top half (scaled to 640×360 each, side-by-side = 1280×360):
        [D-PoSE rendered + bboxes + joints] | [MHRD-Pose rendered + bboxes + joints]
      bottom half (1280×230 each side):
        [D-PoSE lbs_model_params text]      | [MHRD-Pose pred_pose text]
    """
    # ── draw bboxes ──────────────────────────────────────────────────────────
    if raw_dets is not None and len(raw_dets) > 0:
        dpose_bgr = draw_bboxes(dpose_bgr, raw_dets, color=(60, 255, 60))
        mhrd_bgr  = draw_bboxes(mhrd_bgr,  raw_dets, color=(60, 200, 255))

    # ── draw skeleton joints ─────────────────────────────────────────────────
    dpose_bgr = overlay_joints_from_dpose(dpose_bgr, dpose_output)
    mhrd_bgr  = overlay_joints_from_mhrd(mhrd_bgr,  mhrd_output)

    # ── algorithm labels ─────────────────────────────────────────────────────
    cv2.putText(dpose_bgr, f'D-PoSE (MHR conv)  #{frame_number}', (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 255, 60), 2, cv2.LINE_AA)
    cv2.putText(mhrd_bgr,  f'MHRD-Pose          #{frame_number}', (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 200, 255), 2, cv2.LINE_AA)

    # ── scale each frame to 640×360 for side-by-side ─────────────────────────
    dpose_small = cv2.resize(dpose_bgr, (640, 360), interpolation=cv2.INTER_LINEAR)
    mhrd_small  = cv2.resize(mhrd_bgr,  (640, 360), interpolation=cv2.INTER_LINEAR)
    top = np.concatenate([dpose_small, mhrd_small], axis=1)    # (360, 1280, 3)

    # ── extract pose vectors ──────────────────────────────────────────────────
    dpose_vec = None
    if dpose_output is not None:
        p = dpose_output.get('mhr_parameters')
        if p is not None:
            raw = p.get('lbs_model_params')
            if raw is not None:
                dpose_vec = _to_numpy(raw).flatten()

    mhrd_vec = None
    if mhrd_output is not None:
        raw = mhrd_output.get('pred_pose')
        if raw is not None:
            mhrd_vec = _to_numpy(raw).flatten()

    # ── vector text panels ────────────────────────────────────────────────────
    panel_l = make_vector_panel(
        'D-PoSE  lbs_model_params', dpose_vec, width=640, height=230)
    panel_r = make_vector_panel(
        'MHRD-Pose  pred_pose',     mhrd_vec,  width=640, height=230)
    bottom = np.concatenate([panel_l, panel_r], axis=1)        # (230, 1280, 3)

    # ── console print (first 10 values each) ─────────────────────────────────
    if dpose_vec is not None:
        print(f'[D-PoSE]    lbs[:10]  {dpose_vec[:10]}')
    if mhrd_vec is not None:
        print(f'[MHRD-Pose] pose[:10] {mhrd_vec[:10]}')

    return np.concatenate([top, bottom], axis=0)               # (590, 1280, 3)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'dual_demo.log'), level='INFO')
    logger.info(f'Dual demo args: {args}')

    torch.set_float32_matmul_precision('medium')

    # ── D-PoSE tester ────────────────────────────────────────────────────────
    logger.info('Loading D-PoSE tester …')
    dpose_args = argparse.Namespace(
        cfg=args.dpose_cfg,
        ckpt=args.dpose_ckpt,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        tracker_batch_size=1,
        display=True,
        detector='yolo',
        yolo_img_size=416,
        eval_dataset=None,
        dataframe_path='data/ssp_3d_test.npz',
        data_split='test',
        save=False,
        input=args.input,
    )
    dpose_tester = DPoseTester(dpose_args)

    # ── MHRD-Pose tester ─────────────────────────────────────────────────────
    logger.info('Loading MHRD-Pose tester …')
    mhrd_args = argparse.Namespace(
        cfg=args.mhrd_cfg,
        ckpt=args.mhrd_ckpt,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        tracker_batch_size=1,
        display=True,
        detector='yolo',
        yolo_img_size=416,
        eval_dataset=None,
        dataframe_path='data/ssp_3d_test.npz',
        data_split='test',
        save=False,
        input=args.input,
    )
    mhrd_tester = MHRDTester(mhrd_args)

    # ── Shared detector ───────────────────────────────────────────────────────
    mot = MPT(
        device=torch.device('cuda'),
        batch_size=4,
        display=False,
        detector_type='yolo',
        output_format='dict',
        yolo_img_size=416,
    )

    videoWidth, videoHeight, videoFramerate = IMG_W, IMG_H, 30
    cap = getCaptureDeviceFromPath(args.input, videoWidth, videoHeight, videoFramerate)


    frame_number = 0

    while True:
        frame_number += 1
        try:
            ret, raw = cap.read()
            frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f'Frame read error: {e}')
            break

        frame, _scale, _px, _py = scale_and_embed_frame(frame)

        # ── Detect persons (shared) ──────────────────────────────────────────
        input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            raw_detections = mot.detector(input_tensor.cuda())

        if raw_detections:
            boxes  = torch.cat([p['boxes']  for p in raw_detections], dim=0)
            scores = torch.cat([p['scores'] for p in raw_detections], dim=0)
            mask   = scores > 0.7
            dets   = torch.cat([boxes[mask], scores[mask].unsqueeze(1)],
                               dim=1).cpu().numpy()
        else:
            dets = np.empty((0, 5))

        detections  = [dets]
        detection   = mot.prepare_output_detections(detections)
        raw_dets    = detection[0]            # list of [cx, cy, scale, ...]

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if len(raw_dets) == 0:
            # No persons — compose a display that matches the normal (1280×590) size
            display = build_display(frame_bgr, frame_bgr, [], None, None,
                                    frame_number=frame_number)
            cv2.imshow('D-PoSE vs MHRD-Pose', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Run D-PoSE, capture rendered frame ───────────────────────────────
        dpose_output = None
        with capture_imshow() as cap_d:
            try:
                with torch.cuda.amp.autocast():
                    dpose_output = dpose_tester.run_on_single_image_tensor(
                        frame, detection, render=True)
            except Exception as e:
                logger.warning(f'D-PoSE inference error: {e}')
        dpose_bgr = cap_d.get('front', frame_bgr)

        # ── Run MHRD-Pose, capture rendered frame ─────────────────────────────
        mhrd_output = None
        with capture_imshow() as cap_m:
            try:
                with torch.no_grad():
                    mhrd_output = mhrd_tester.run_on_single_image_tensor(
                        frame, detection)
            except Exception as e:
                logger.warning(f'MHRD-Pose inference error: {e}')
        mhrd_bgr = cap_m.get('front', frame_bgr)

        # ── Per-frame diagnostics (first 10 frames, then every 20th) ──────────
        if frame_number <= 10 or frame_number % 20 == 0:
            print_frame_diag(frame_number, raw_dets, dpose_output, mhrd_output, frame_bgr)

        # ── Assemble display ──────────────────────────────────────────────────
        display = build_display(
            dpose_bgr, mhrd_bgr, raw_dets,
            dpose_output, mhrd_output,
            frame_number=frame_number,
        )

        cv2.imshow('D-PoSE vs MHRD-Pose', display)

        if args.save:
            cv2.imwrite(f'dual_frame_{frame_number:05d}.jpg', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    del dpose_tester.model
    del mhrd_tester.model
    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Side-by-side D-PoSE vs MHRD-Pose comparison')

    parser.add_argument('--input', type=str, default='/dev/video0',
                        help='Video source: path, /dev/videoX, webcam, screen')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction,
                        help='Save dual frames as dual_frame_XXXXX.jpg')

    parser.add_argument('--dpose_cfg', type=str,
                        default='configs/dpose_conf.yaml',
                        help='D-PoSE config YAML')
    parser.add_argument('--dpose_ckpt', type=str,
                        default='data/ckpt/paper_arxiv.ckpt',
                        help='D-PoSE checkpoint')

    parser.add_argument('--mhrd_cfg', type=str,
                        default=os.path.join(_MHRD_DIR, 'config_mhr.yaml'),
                        help='MHRD-Pose config YAML')
    parser.add_argument('--mhrd_ckpt', type=str, default='',
                        help='MHRD-Pose checkpoint (.ckpt)')

    parser.add_argument('--image_folder', type=str, default='demo_images')
    parser.add_argument('--output_folder', type=str,
                        default='demo_images/results')

    args = parser.parse_args()
    main(args)
