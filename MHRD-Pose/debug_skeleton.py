"""
debug_skeleton.py — Deep diagnostic for MHR skeleton vs SMPL joint alignment.

Compares:
  (a) SMPL joints derived from GT MHR vertices via barycentric mapping
  (b) MHR skeleton joints from skel_state
  (c) Predicted SMPL joints from model forward pass

Prints per-joint distances, coordinate ranges, and projection mismatches.

Usage:
  python debug_skeleton.py --n 3                     # GT-only mode
  python debug_skeleton.py --ckpt model.ckpt --n 3   # with model predictions
"""

import os
import sys
import argparse
import numpy as np
import torch
import pickle
import scipy.sparse

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import types


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# SMPL joint regressor setup
# ---------------------------------------------------------------------------


def build_smpl_joint_components():
    """Build barycentric mapping + J_regressor for SMPL joints from MHR vertices."""
    _mapping = np.load(os.path.join(_ROOT, 'assets', 'mhr2smpl_mapping.npz'))
    triangle_ids = _mapping['triangle_ids'].astype(np.int64)
    baryc = _mapping['baryc_coords'].astype(np.float32)

    from load_mhr_portable import load_portable
    _portable = load_portable(
        os.path.join(_ROOT, 'mhr_portable_dump', 'mhr_dump_lod1.pt'))
    _faces = _portable['interesting']['character.mesh.faces']
    if isinstance(_faces, torch.Tensor):
        faces = _faces.cpu().numpy()
    else:
        faces = np.array(_faces)

    tri_vids = faces[triangle_ids].astype(np.int64)  # [6890, 3]

    # Stub chumpy for SMPL pkl load
    _saved = {k: sys.modules.get(k) for k in ('chumpy', 'chumpy.ch')}
    if 'chumpy' not in sys.modules:
        _c = types.ModuleType('chumpy')
        _ch = types.ModuleType('chumpy.ch')

        class _Ch:
            pass

        _c.Ch = _Ch
        _c.array = lambda *a, **k: None
        _ch.Ch = _Ch
        sys.modules['chumpy'] = _c
        sys.modules['chumpy.ch'] = _ch
    try:
        _pkl_path = os.path.join(
            _ROOT,
            'data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_NEUTRAL.pkl'
        )
        with open(_pkl_path, 'rb') as _f:
            _smpl_data = pickle.load(_f, encoding='latin1')
    finally:
        for _k, _v in _saved.items():
            if _v is None:
                sys.modules.pop(_k, None)
            else:
                sys.modules[_k] = _v

    J_reg = _smpl_data['J_regressor']
    if scipy.sparse.issparse(J_reg):
        J_reg = np.array(J_reg.todense()).astype(np.float32)
    else:
        J_reg = np.array(J_reg).astype(np.float32)  # [24, 6890]

    return tri_vids, baryc, J_reg, faces


def compute_smpl_joints_from_verts(verts_m, tri_vids, baryc, J_reg):
    """Compute 24 SMPL joints from MHR vertices [V,3] in metres.

    verts_m: [V, 3] MHR vertices
    Returns: [24, 3] SMPL joints in the same coordinate space
    """
    v0 = verts_m[tri_vids[:, 0]]
    v1 = verts_m[tri_vids[:, 1]]
    v2 = verts_m[tri_vids[:, 2]]
    smpl_verts = (baryc[:, 0:1] * v0 + baryc[:, 1:2] * v1 + baryc[:, 2:3] * v2
                  )  # [6890, 3]
    return J_reg @ smpl_verts  # [24, 3]


# ---------------------------------------------------------------------------
# SMPL joint names for readable output
# ---------------------------------------------------------------------------

SMPL_JOINT_NAMES = [
    'hip',
    'L_hip',
    'R_hip',
    'spine1',
    'L_knee',
    'R_knee',
    'spine2',
    'L_ankle',
    'R_ankle',
    'spine3',
    'L_foot',
    'R_foot',
    'neck',
    'L_collar',
    'R_collar',
    'head',
    'L_shoulder',
    'R_shoulder',
    'L_elbow',
    'R_elbow',
    'L_wrist',
    'R_wrist',
    'L_hand',
    'R_hand',
]

# SMPL→MHR joint index mapping imported from mhr_constants — single source of truth.
from mhr_constants import MHR_TO_SMPL_JOINT_INDICES as _MHR_TO_SMPL

SMPL_MHR_INDEX_HINT = {
    smpl_i: mhr_i
    for smpl_i, mhr_i in enumerate(_MHR_TO_SMPL)
}

# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--dataset', default='agora-bfh')
    parser.add_argument('--split', default='train')
    parser.add_argument('--cfg', default='config_mhr.yaml')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='CUDA device (0=auto, -1=auto with most free)')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)

    # ---- select GPU -------------------------------------------------------
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        print(f'Using GPU {args.gpu}')
        torch.cuda.empty_cache()
    else:
        # Auto-select GPU with most free memory
        ng = torch.cuda.device_count()
        if ng > 1:
            frees = [torch.cuda.mem_get_info(i)[0] for i in range(ng)]
            args.gpu = max(range(ng), key=lambda i: frees[i])
            print(
                f'Auto-selected GPU {args.gpu} (free: {frees[args.gpu]/1024**3:.1f} GB)'
            )
            torch.cuda.set_device(args.gpu)
            torch.cuda.empty_cache()
        else:
            args.gpu = 0
            print(f'Using GPU {args.gpu} (single GPU)')

    # ---- load config & dataset -------------------------------------------
    from train.utils.train_utils import update_hparams
    hparams = update_hparams(args.cfg)

    from dataset_wrapper import DatasetHMR
    ds = DatasetHMR(hparams.DATASET,
                    dataset=args.dataset,
                    is_train=(args.split == 'train'))
    print(f'Dataset: {args.dataset} ({args.split}), size={len(ds)}')

    indices = np.random.choice(len(ds),
                               size=min(args.n, len(ds)),
                               replace=False)

    # ---- build SMPL joint components --------------------------------------
    tri_vids, baryc, J_reg, faces = build_smpl_joint_components()
    print(f'Barycentric mapping: {tri_vids.shape[0]} triangles, '
          f'MHR verts: {faces.shape[0]} faces')
    print(f'SMPL J_regressor: {J_reg.shape}')

    # ---- load MHR model ---------------------------------------------------
    from mhr_constants import MHR_MODEL_PT
    # Search for mhr_model.pt in common locations relative to the repo root
    _candidates = [
        os.path.join(_ROOT, MHR_MODEL_PT),  # D-PoSE/mhr_model.pt
        os.path.join(_ROOT, 'mhr', 'mhr_model.pt'),  # D-PoSE/mhr/mhr_model.pt
        os.path.join(_ROOT, 'MHR', 'mhr_model.pt'),  # D-PoSE/MHR/mhr_model.pt
        os.path.join(_ROOT, 'MHR', 'assets',
                     'mhr_model.pt'),  # D-PoSE/MHR/assets/mhr_model.pt
    ]
    _mhr_pt = None
    for _c in _candidates:
        if os.path.exists(_c):
            _mhr_pt = _c
            break
    if _mhr_pt is None:
        raise FileNotFoundError(
            f"mhr_model.pt not found. Searched: {_candidates}\n"
            f"_ROOT={_ROOT}")
    _dev = f'cuda:{args.gpu}'
    mhr_model = torch.load(_mhr_pt, map_location=_dev, weights_only=False)
    mhr_model.to(_dev)
    mhr_model.eval()

    # ---- load checkpoint (optional) ---------------------------------------
    model_trainer = None
    if args.ckpt:
        from mhr_trainer import MHRTrainer
        model_trainer = MHRTrainer.load_from_checkpoint(args.ckpt,
                                                        hparams=hparams)
        model_trainer.eval().to(_dev)
        print(f'Loaded checkpoint: {args.ckpt}')

    # ---- per-sample diagnostics -------------------------------------------
    for si, ds_idx in enumerate(indices):
        item = ds[int(ds_idx)]
        print(f'\n{"="*70}')
        print(f'SAMPLE {si} (dataset index {ds_idx})')
        print(f'{"="*70}')

        # ---- basic info ---------------------------------------------------
        img_h = int(item['orig_shape'][0])
        img_w = int(item['orig_shape'][1])
        bbox_scale = float(item['scale'])
        bbox_center = to_numpy(item['center'])
        print(
            f'  Image: {img_w}x{img_h}, bbox_scale={bbox_scale:.3f}, center={bbox_center}'
        )

        # ---- GT MHR vertices ----------------------------------------------
        gt_verts = to_numpy(item['vertices'])  # [V, 3] in metres
        print(f'\n--- GT MHR Vertices ---')
        print(f'  Shape: {gt_verts.shape}')
        print(
            f'  Range: x=[{gt_verts[:,0].min():.3f}, {gt_verts[:,0].max():.3f}], '
            f'y=[{gt_verts[:,1].min():.3f}, {gt_verts[:,1].max():.3f}], '
            f'z=[{gt_verts[:,2].min():.3f}, {gt_verts[:,2].max():.3f}]')
        print(
            f'  Body height (Y): {gt_verts[:,1].max() - gt_verts[:,1].min():.3f} m'
        )

        # ---- GT SMPL joints (barycentric) ---------------------------------
        gt_smpl_joints = compute_smpl_joints_from_verts(
            gt_verts, tri_vids, baryc, J_reg)
        print(f'\n--- GT SMPL Joints (from MHR verts, barycentric) ---')
        print(f'  Shape: {gt_smpl_joints.shape}')
        print(f'  Pelvis (joint 0): {gt_smpl_joints[0]}')
        print(f'  L_hip (joint 1): {gt_smpl_joints[1]}')
        print(f'  R_hip (joint 2): {gt_smpl_joints[2]}')
        print(f'  Pelvis-centre offset: '
              f'({(gt_smpl_joints[1]+gt_smpl_joints[2])/2})')
        for ji, name in enumerate(SMPL_JOINT_NAMES):
            print(f'    [{ji:2d}] {name:12s}: {gt_smpl_joints[ji]}')

        # ---- GT MHR skel_state joints -------------------------------------
        # Re-derive via MHR forward pass
        identity = to_numpy(item['identity_coeffs']).astype(np.float32)
        pose = to_numpy(item['lbs_model_params']).astype(np.float32)
        expr = to_numpy(item.get('face_expr_coeffs',
                                 np.zeros(72))).astype(np.float32)

        mhr_identity = torch.from_numpy(identity).unsqueeze(0).to(_dev)
        mhr_pose = torch.from_numpy(pose).unsqueeze(0).to(_dev)
        mhr_expr = torch.from_numpy(expr).unsqueeze(0).to(_dev)

        with torch.no_grad():
            verts_cm_out, skel_state = mhr_model(
                identity_coeffs=mhr_identity,
                model_parameters=mhr_pose,
                face_expr_coeffs=mhr_expr,
                apply_correctives=True,
            )
        verts_m_out = verts_cm_out.cpu().numpy() * 0.01  # [1, V, 3]
        skel_np = skel_state.cpu().numpy()  # [1, 127, 8]

        print(f'\n--- GT MHR Skel State (from MHR forward) ---')
        print(f'  skel_state shape: {skel_np.shape}')
        print(
            f'  Format: last_dim={skel_np.shape[-1]} (8=tz+quat, 16=col-major 4x4)'
        )

        # Extract joints: first 3 components = [tx, ty, tz] in cm
        skel_joints_cm = skel_np[0, :, :3]  # [127, 3] cm
        skel_joints_m = skel_joints_cm * 0.01  # [127, 3] metres

        print(f'  First 25 MHR skeleton joints (metres):')
        from mhr_constants import MHR_JOINT_NAMES as MHR_NAMES
        for ji in range(min(25, len(skel_joints_m))):
            name = MHR_NAMES[ji] if ji < len(MHR_NAMES) else f'joint_{ji}'
            print(f'    [{ji:3d}] {name:16s}: {skel_joints_m[ji]}')

        # Ensure skel_joints_m is 2D [N, 3] for indexing consistency
        skel_joints_m = np.atleast_2d(skel_joints_m)
        if skel_joints_m.shape[0] == 1 and skel_joints_m.shape[1] > 20:
            # Shape (1, 127, 3) — squeeze batch
            skel_joints_m = skel_joints_m[0]

        # ---- Compare GT SMPL joints vs GT MHR skel joints --------------------
        print(f'\n--- Coordinate Comparison ---')
        print(
            f'  GT SMPL joints range: '
            f'y=[{gt_smpl_joints[:,1].min():.3f}, {gt_smpl_joints[:,1].max():.3f}]'
        )
        print(
            f'  GT MHR skel range (first 24): '
            f'y=[{skel_joints_m[:24,1].min():.3f}, {skel_joints_m[:24,1].max():.3f}]'
        )

        # Pelvis positions
        smpl_pelvis = gt_smpl_joints[0]
        mhr_hip = skel_joints_m[1] if skel_joints_m.shape[
            0] > 1 else skel_joints_m[0]
        print(f'  SMPL pelvis (joint 0): {smpl_pelvis}')
        print(f'  MHR hip (joint 1):     {mhr_hip}')
        print(f'  Distance: {np.linalg.norm(smpl_pelvis - mhr_hip):.3f} m')

        # ---- Compare barycentric SMPL joints from MHR forward vs GT ----------
        gt_smpl_from_fwd = compute_smpl_joints_from_verts(
            verts_m_out[0], tri_vids, baryc, J_reg)
        print(f'\n--- Barycentric Consistency ---')
        print(f'  GT SMPL joints (from item.vertices):')
        print(f'    pelvis: {gt_smpl_joints[0]}')
        print(f'  GT SMPL joints (from MHR forward on same params):')
        print(f'    pelvis: {gt_smpl_from_fwd[0]}')
        per_joint_diff = np.linalg.norm(gt_smpl_joints - gt_smpl_from_fwd,
                                        axis=1)
        print(f'  Per-joint distance (item vs forward):')
        for ji, name in enumerate(SMPL_JOINT_NAMES):
            dist = per_joint_diff[ji]
            marker = ' *** BIG' if dist > 0.05 else ''
            print(f'    [{ji:2d}] {name:12s}: {dist*1000:.1f} mm{marker}')
        print(f'  Mean per-joint diff: {per_joint_diff.mean()*1000:.1f} mm')
        print(f'  Max per-joint diff:  {per_joint_diff.max()*1000:.1f} mm')

        # ---- Model predictions (if checkpoint) ------------------------------
        if model_trainer is not None:
            batch = {
                k:
                v.unsqueeze(0).to(_dev) if isinstance(v, torch.Tensor) else v
                for k, v in item.items()
            }
            for k in ('scale', 'center', 'orig_shape', 'focal_length'):
                if k in batch and isinstance(
                        batch[k], torch.Tensor) and batch[k].dim() == 1:
                    batch[k] = batch[k].unsqueeze(0)

            with torch.no_grad():
                out, *_ = model_trainer(
                    batch['img'],
                    bbox_scale=batch['scale'],
                    bbox_center=batch['center'],
                    img_w=batch['orig_shape'][:, 1],
                    img_h=batch['orig_shape'][:, 0],
                    fl=batch.get('focal_length'),
                )

            pred_verts = out['vertices'][0].cpu().numpy()  # [V, 3] m
            pred_smpl_joints = out['joints3d_smpl'][0].cpu().numpy()  # [24, 3]
            pred_cam_t = out['pred_cam_t'][0].cpu().numpy()
            pred_joints2d = out.get('joints2d_smpl')
            if pred_joints2d is not None:
                pred_joints2d = pred_joints2d[0].cpu().numpy()  # [24, 2]

            print(f'\n--- Model Predictions ---')
            print(f'  Pred verts shape: {pred_verts.shape}')
            print(
                f'  Pred verts range: '
                f'y=[{pred_verts[:,1].min():.3f}, {pred_verts[:,1].max():.3f}]'
            )
            print(f'  Pred cam_t: {pred_cam_t}')
            print(f'  Pred SMPL joints:')
            for ji, name in enumerate(SMPL_JOINT_NAMES):
                print(f'    [{ji:2d}] {name:12s}: {pred_smpl_joints[ji]}')

            # ---- Pred vs GT SMPL joints (both body-local, pelvis-centred) ----
            print(f'\n--- Pred vs GT SMPL Joints (pelvis-centred, mm) ---')
            gt_pel = (gt_smpl_joints[1] + gt_smpl_joints[2]) / 2.0
            pred_pel = (pred_smpl_joints[1] + pred_smpl_joints[2]) / 2.0
            gt_pc = gt_smpl_joints - gt_pel
            pred_pc = pred_smpl_joints - pred_pel
            per_joint_mpjpe = np.linalg.norm(gt_pc - pred_pc, axis=1) * 1000
            print(f'  Per-joint MPJPE (mm):')
            total_mpjpe = 0
            for ji, name in enumerate(SMPL_JOINT_NAMES):
                print(
                    f'    [{ji:2d}] {name:12s}: {per_joint_mpjpe[ji]:.1f} mm')
                total_mpjpe += per_joint_mpjpe[ji]
            print(f'  Mean MPJPE: {total_mpjpe / 24:.1f} mm')
            print(f'  Max MPJPE:  {per_joint_mpjpe.max():.1f} mm')

            # ---- Pred vs GT vertex distance ----------------------------------
            print(f'\n--- Pred vs GT Vertices ---')
            if pred_verts.shape == gt_verts.shape:
                pve = np.linalg.norm(pred_verts - gt_verts,
                                     axis=1).mean() * 1000
                print(f'  Mean PVE: {pve:.1f} mm')
                print(
                    f'  Max PVE:  {np.linalg.norm(pred_verts - gt_verts, axis=1).max()*1000:.1f} mm'
                )
            else:
                print(
                    f'  Shape mismatch: pred {pred_verts.shape} vs gt {gt_verts.shape}'
                )

        # ---- GT 2D keypoint alignment check ---------------------------------
        print(f'\n--- GT 2D Keypoint Check ---')
        if 'keypoints_orig' in item:
            kp_orig = to_numpy(item['keypoints_orig'])  # [J, 3] full-image px
            half = bbox_scale * 100.0
            # Project GT SMPL joints to 2D (simple weak-perspective via bbox)
            tz_est = 2.0 * (img_w**2 + img_h**2)**0.5 / (bbox_scale * 200.0)
            fl_est = (img_w**2 + img_h**2)**0.5
            fl_crop = fl_est * (224.0 / (bbox_scale * 200.0))

            # Pelvis 2D from keypoints (mid of L-hip[1], R-hip[2])
            pelvis_kp_full = (kp_orig[1, :2] + kp_orig[2, :2]) / 2.0
            pelvis_kp_crop = ((pelvis_kp_full[0] -
                               (bbox_center[0] - half)) / (2 * half) * 224,
                              (pelvis_kp_full[1] -
                               (bbox_center[1] - half)) / (2 * half) * 224)
            print(
                f'  Pelvis 2D from keypoints (crop px): ({pelvis_kp_crop[0]:.0f}, {pelvis_kp_crop[1]:.0f})'
            )

            # Pelvis-centre the 3D SMPL joints before projecting (the renderer
            # puts the SMPL pelvis at world origin and aligns it to the 2D pelvis
            # via cam_t). Previously we projected raw joints which carry the
            # MHR origin-at-feet offset — that always flagged MISMATCH spuriously.
            gt_pel = (gt_smpl_joints[1] + gt_smpl_joints[2]) / 2.0
            gt_joints_pc = gt_smpl_joints - gt_pel

            # Camera translation (tx, ty) derived from the observed pelvis 2D.
            tx = (pelvis_kp_crop[0] - 112.0) * tz_est / fl_crop
            ty = (pelvis_kp_crop[1] - 112.0) * tz_est / fl_crop

            for ji in [0, 1, 2, 4, 5, 10, 11, 19, 20]:
                if ji >= gt_joints_pc.shape[0]:
                    continue
                j3d = gt_joints_pc[ji]
                depth = j3d[2] + tz_est
                if depth <= 0:
                    continue
                jx = 112 + fl_crop * (j3d[0] + tx) / depth
                jy = 112 + fl_crop * (j3d[1] + ty) / depth
                kp_crop_x = (kp_orig[ji, 0] -
                             (bbox_center[0] - half)) / (2 * half) * 224
                kp_crop_y = (kp_orig[ji, 1] -
                             (bbox_center[1] - half)) / (2 * half) * 224
                proj_dist = np.sqrt((jx - kp_crop_x)**2 + (jy - kp_crop_y)**2)
                marker = ' *** MISMATCH' if proj_dist > 30 else ''
                print(
                    f'    [{ji:2d}] proj=({jx:.0f},{jy:.0f}) kp=({kp_crop_x:.0f},{kp_crop_y:.0f}) '
                    f'dist={proj_dist:.0f} px{marker}')

    print(f'\n{"="*70}')
    print('DONE. Check prints above for per-joint mismatches.')


if __name__ == '__main__':
    main()
