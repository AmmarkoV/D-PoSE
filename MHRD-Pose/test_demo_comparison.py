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

Coordinate convention assumed for joints3d_smpl:
  Camera frame — X right, Y down, Z forward (into scene), units metres.
  A right-side-up person: head_Y < pelvis_Y.
  A forward-facing person: chest normal Z < 0.

Usage (run from D-PoSE root):
    python MHRD-Pose/test_demo_comparison.py \\
        --input crazydance.mp4 \\
        --ckpt  MHRD-Pose/test.ckpt [--frames 20]

    # With original model as reference (T6):
    python MHRD-Pose/test_demo_comparison.py \\
        --input crazydance.mp4 \\
        --ckpt  MHRD-Pose/test.ckpt \\
        --ref-ckpt data/ckpt/paper_arxiv.ckpt
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
              dets: np.ndarray) -> Optional[np.ndarray]:
        """
        frame_rgb: HxWx3 uint8 RGB
        dets:      [M, 5] — cx, cy, scale_px, unused, score  (MPT format)
        Returns:   [M, 22, 3] joints3d_smpl in metres, or None if no detections.
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

        p = out.get('pred_pose')   # [B, D] LBS params; None if key absent
        v = out.get('vertices')    # [B, V, 3] mesh vertices; None if absent
        return (j.detach().cpu().numpy(),
                p.detach().cpu().numpy() if p is not None else None,
                v.detach().cpu().numpy() if v is not None else None)


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
              dets: np.ndarray) -> Optional[np.ndarray]:
        """Returns [M, 22, 3] joints in metres (converted from cm)."""
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
        return joints_m


# ─── Detection helper ─────────────────────────────────────────────────────────

def detect_people(frames: List[np.ndarray],
                  device: torch.device) -> List[np.ndarray]:
    """Run YOLO+MPT detection on a list of RGB frames.

    Returns a list of [M, 5] arrays (cx, cy, scale, _, score).
    """
    from multi_person_tracker import MPT

    mot = MPT(device=device, batch_size=4, display=False,
              detector_type='yolo', output_format='dict',
              yolo_img_size=416)
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
                                dim=1).cpu().numpy()
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

    mhrd_joints_all = []
    mhrd_poses_all  = []
    mhrd_verts_all  = []
    for frame, dets in zip(frames, dets_list):
        result = mhrd.infer(frame, dets)
        if result is None:
            continue
        j, p, v = result
        if j is not None and len(j) > 0:
            mhrd_joints_all.append(j[0])
            if p is not None:
                mhrd_poses_all.append(p[0])
            if v is not None:
                mhrd_verts_all.append(v[0])

    if len(mhrd_joints_all) < 3:
        print('ERROR: MHRD model produced too few joint outputs — aborting')
        return 2

    mhrd_joints = np.stack(mhrd_joints_all)   # [N, J, 3]
    mhrd_poses  = np.stack(mhrd_poses_all) if mhrd_poses_all else None  # [N, D]
    mhrd_verts  = np.stack(mhrd_verts_all)  if mhrd_verts_all  else None  # [N, V, 3]
    print(f'  Collected {len(mhrd_joints)} MHRD joint frames, '
          f'shape {mhrd_joints.shape}, '
          f'range [{mhrd_joints.min():.3f}, {mhrd_joints.max():.3f}] m')
    if mhrd_poses is not None:
        print(f'  pred_pose shape {mhrd_poses.shape}, '
              f'mean-std across frames = {mhrd_poses.std(axis=0).mean():.4f}')

    # ── 4. Reference inference (optional) ────────────────────────────────────
    ref_joints = None
    if args.ref_ckpt:
        print(f'\nLoading original reference model ({args.ref_ckpt}) …')
        try:
            orig = OriginalInference(args.ref_ckpt, device)
            ref_all = []
            for frame, dets in zip(frames, dets_list):
                j = orig.infer(frame, dets)
                if j is not None and len(j) > 0:
                    ref_all.append(j[0])
            if ref_all:
                ref_joints = np.stack(ref_all)
                print(f'  Collected {len(ref_joints)} reference joint frames')
            else:
                print('  WARNING: reference model produced no output — T6 skipped')
        except Exception as e:
            print(f'  WARNING: could not load reference model ({e}) — T6 skipped')

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
    ]
    if mhrd_verts is not None:
        n_v = min(len(mhrd_joints), len(mhrd_verts))
        results.append(check_vertices_orientation(mhrd_joints[:n_v], mhrd_verts[:n_v]))
    else:
        results.append(CheckResult('T9_vertices_not_upside_down', False,
                                   'vertices not in model output — check output keys'))
    if ref_joints is not None:
        n_common = min(len(mhrd_joints), len(ref_joints))
        results.append(check_vs_reference(mhrd_joints[:n_common],
                                          ref_joints[:n_common]))

    for r in results:
        print(r)

    # ── 6. Diagnostic dump ────────────────────────────────────────────────────
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

    # ── 7. Summary ────────────────────────────────────────────────────────────
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
    args = parser.parse_args()

    sys.exit(run_tests(args))
