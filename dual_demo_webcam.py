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
    """Extract SMPL-mapped joints from D-PoSE mhr_skel_state and draw them."""
    if dpose_output is None:
        return frame_bgr
    skel = dpose_output.get('mhr_skel_state')
    if skel is None:
        return frame_bgr
    skel = _to_numpy(skel)
    if skel.ndim == 2:
        skel = skel[np.newaxis]   # [1, 127, 8]
    for b in range(len(skel)):
        joints_cm = skel[b][MHR_TO_SMPL_JOINT_INDICES, :3]   # [24, 3] cm, Y-up
        joints_m  = joints_cm * 0.01
        joints2d  = project_yup_joints(joints_m)
        frame_bgr = draw_joints_2d(frame_bgr, joints2d,
                                   joint_color=(0, 255, 200), bone_color=(0, 180, 255))
    return frame_bgr


def overlay_joints_from_mhrd(frame_bgr, mhrd_output):
    """Draw 2D joints from MHRD-Pose joints2d_smpl (already in full-image px)."""
    if mhrd_output is None:
        return frame_bgr
    j2d = mhrd_output.get('joints2d_smpl')
    if j2d is None:
        # Fallback: project joints3d_smpl + pred_cam_t (Y-down OpenCV convention)
        j3d  = mhrd_output.get('joints3d_smpl')
        camt = mhrd_output.get('pred_cam_t')
        if j3d is None or camt is None:
            return frame_bgr
        j3d  = _to_numpy(j3d)
        camt = _to_numpy(camt)
        if j3d.ndim == 2: j3d = j3d[np.newaxis]
        if camt.ndim == 1: camt = camt[np.newaxis]
        for b in range(len(j3d)):
            xyz = j3d[b] + camt[b][None, :]
            z = np.where(np.abs(xyz[:, 2]) < 0.05, 0.05, xyz[:, 2])
            u = IMG_W / 2 + FOCAL_LENGTH * xyz[:, 0] / z
            v = IMG_H / 2 + FOCAL_LENGTH * xyz[:, 1] / z
            frame_bgr = draw_joints_2d(frame_bgr, np.stack([u, v], axis=1),
                                       joint_color=(0, 255, 255), bone_color=(255, 128, 0))
    else:
        j2d_np = _to_numpy(j2d)
        if j2d_np.ndim == 2:
            j2d_np = j2d_np[np.newaxis]   # [1, 24, 2]
        for b in range(len(j2d_np)):
            frame_bgr = draw_joints_2d(frame_bgr, j2d_np[b],
                                       joint_color=(0, 255, 255), bone_color=(255, 128, 0))
    return frame_bgr


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

    fallback_bgr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
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
            display = np.concatenate(
                [np.concatenate([frame_bgr, frame_bgr], axis=1)] * 2
                if False else [frame_bgr, frame_bgr], axis=1)
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
