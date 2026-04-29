"""
Check global rotation convention distribution across training datasets.

Compares axis-angle root rotations (pose_cam[:,0:3]) between AGORA and BEDLAM
datasets to detect coordinate-frame mismatches that cause upside-down predictions.

Convention tested:
  - Camera-frame Y-down (typical CV): body-up direction has R[1,1] < 0
  - World-frame Y-up (typical MoCap):  body-up direction has R[1,1] > 0

Usage (from workspace root, inside Docker):
    python MHRD-Pose/scripts/check_rotation_convention.py
    python MHRD-Pose/scripts/check_rotation_convention.py --labels_dir data/training_labels/all_npz_12_training
    python MHRD-Pose/scripts/check_rotation_convention.py --also_check_cache
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Rodrigues axis-angle → rotation matrix (numpy, no torch dependency)
# ---------------------------------------------------------------------------

def axis_angle_to_rotmat(aa):
    """
    aa: (N, 3) axis-angle vectors
    Returns R: (N, 3, 3)
    """
    N = aa.shape[0]
    angle = np.linalg.norm(aa, axis=1, keepdims=True)  # (N,1)
    safe_angle = np.where(angle < 1e-8, np.ones_like(angle), angle)
    axis = aa / safe_angle  # (N,3)

    s = np.sin(angle)      # (N,1)
    c = np.cos(angle)      # (N,1)
    t = 1.0 - c            # (N,1)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    R = np.stack([
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c,
    ], axis=1).reshape(N, 3, 3)

    # Zero-angle: identity
    zero_mask = (angle[:, 0] < 1e-8)
    R[zero_mask] = np.eye(3)
    return R


def analyze_dataset(npz_path, max_samples=50_000):
    """
    Load a label NPZ and compute root-rotation statistics.

    Returns dict with keys:
      n, mean_R11, std_R11, pct_ydown, mean_up_dir (3,)
    or None if the file can't be read.
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return None, str(e)

    # Locate pose array — different datasets may use slightly different keys
    pose_key = None
    for k in ('pose_cam', 'pose', 'poses'):
        if k in data:
            pose_key = k
            break
    if pose_key is None:
        return None, f"no pose key found; keys={list(data.keys())[:10]}"

    pose = data[pose_key]  # (N, D) — first 3 dims are global root aa

    if pose.ndim != 2 or pose.shape[1] < 3:
        return None, f"unexpected pose shape {pose.shape}"

    N = min(len(pose), max_samples)
    aa = pose[:N, :3].astype(np.float32)

    R = axis_angle_to_rotmat(aa)  # (N,3,3)

    # Body "up" in SMPL is the +Y axis of the root bone.
    # After applying global rotation R, the body-up direction in camera space is R[:,1,:].
    # R[i,1,1] > 0 → body-up points along +Y in camera (typically "down" in image = upside-down)
    # R[i,1,1] < 0 → body-up points along -Y in camera (typically "up" in image = normal)

    R11 = R[:, 1, 1]  # body-up projected onto camera Y axis
    mean_up_dir = R[:, 1, :].mean(axis=0)  # mean body-up vector in camera

    pct_ydown = float((R11 < 0).mean() * 100)  # % samples where body-up is in camera -Y (correct)

    return {
        'n': N,
        'mean_R11': float(R11.mean()),
        'std_R11': float(R11.std()),
        'pct_correct': pct_ydown,      # % where body is upright in camera frame
        'mean_up_dir': mean_up_dir,
    }, None


def analyze_mhr_cache(cache_path, max_samples=50_000):
    """
    Check lbs_model_params[:,0:3] (global root rotation in MHR format).
    MHR params[0:3] = axis-angle root rotation (same convention as SMPL pose_cam[0:3]).
    """
    try:
        data = np.load(cache_path, allow_pickle=False)
    except Exception as e:
        return None, str(e)

    if 'lbs_model_params' not in data:
        return None, "no lbs_model_params key"

    params = data['lbs_model_params']
    if params.shape[1] < 3:
        return None, f"lbs_model_params shape {params.shape} too narrow"

    N = min(len(params), max_samples)
    aa = params[:N, :3].astype(np.float32)
    R = axis_angle_to_rotmat(aa)
    R11 = R[:, 1, 1]
    mean_up_dir = R[:, 1, :].mean(axis=0)
    pct_correct = float((R11 < 0).mean() * 100)

    return {
        'n': N,
        'mean_R11': float(R11.mean()),
        'std_R11': float(R11.std()),
        'pct_correct': pct_correct,
        'mean_up_dir': mean_up_dir,
    }, None


def fmt_up(vec):
    return f"[{vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_dir', default='data/training_labels/all_npz_12_training',
                        help='Directory containing per-dataset label NPZs')
    parser.add_argument('--cache_dir', default='data/mhr_cache',
                        help='Directory containing _mhr_params.npz cache files')
    parser.add_argument('--also_check_cache', action='store_true',
                        help='Also check MHR cache files (if they exist)')
    parser.add_argument('--max_samples', type=int, default=50_000,
                        help='Max samples per dataset to analyse (default 50000)')
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    cache_dir  = Path(args.cache_dir)

    if not labels_dir.exists():
        print(f"ERROR: labels dir not found: {labels_dir}")
        sys.exit(1)

    npz_files = sorted(labels_dir.glob('*.npz'))
    if not npz_files:
        print(f"ERROR: no .npz files found in {labels_dir}")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  Global Root Rotation Convention Check")
    print(f"  Labels:  {labels_dir}")
    print(f"  Files:   {len(npz_files)}")
    print(f"{'='*72}")
    print(f"\n  Interpretation:")
    print(f"    pct_correct = % samples where body-up aligns with camera -Y axis")
    print(f"    (i.e., person appears upright in image)")
    print(f"    mean_R11 < 0 → dataset is predominantly upright in camera frame")
    print(f"    mean_R11 > 0 → dataset is predominantly inverted (upside-down)")
    print()

    header = f"{'Dataset':<35} {'N':>7}  {'mean_R11':>9}  {'std_R11':>8}  {'pct_correct':>11}  mean_up_dir"
    print(header)
    print('-' * len(header))

    agora_r11 = []
    results = {}

    for npz_path in npz_files:
        ds_name = npz_path.stem  # e.g. 'agora-bfh' or 'static-hdri'
        stats, err = analyze_dataset(npz_path, args.max_samples)
        if stats is None:
            print(f"  {'[SKIP] '+ds_name:<33}  {err}")
            continue

        results[ds_name] = stats
        flag = ''
        if 'agora' in ds_name.lower():
            agora_r11.append(stats['mean_R11'])
            flag = ' [AGORA]'
        elif stats['mean_R11'] > 0.1:
            flag = ' <<< INVERTED?'

        print(
            f"  {ds_name:<33} {stats['n']:>7}  "
            f"{stats['mean_R11']:>+9.4f}  {stats['std_R11']:>8.4f}  "
            f"{stats['pct_correct']:>10.1f}%  "
            f"{fmt_up(stats['mean_up_dir'])}{flag}"
        )

    # Summary
    print()
    if agora_r11:
        agora_mean = np.mean(agora_r11)
        print(f"  AGORA baseline mean_R11 = {agora_mean:+.4f}")
        print()
        outliers = [(k, v) for k, v in results.items()
                    if 'agora' not in k.lower() and abs(v['mean_R11'] - agora_mean) > 0.3]
        if outliers:
            print("  CONVENTION MISMATCH CANDIDATES (|mean_R11 - AGORA| > 0.3):")
            for ds, v in outliers:
                delta = v['mean_R11'] - agora_mean
                print(f"    {ds:<35}  delta={delta:+.4f}  pct_correct={v['pct_correct']:.1f}%")
        else:
            print("  No obvious convention mismatches detected (all within 0.3 of AGORA).")

    # Optional MHR cache check
    if args.also_check_cache and cache_dir.exists():
        cache_files = sorted(cache_dir.glob('*_mhr_params.npz'))
        if cache_files:
            print(f"\n{'='*72}")
            print(f"  MHR Cache Check  ({cache_dir})")
            print(f"{'='*72}")
            print(f"\n  {'Dataset':<35} {'N':>7}  {'mean_R11':>9}  {'pct_correct':>11}")
            print(f"  {'-'*65}")
            for cp in cache_files:
                ds_name = cp.stem.replace('_mhr_params', '')
                stats, err = analyze_mhr_cache(cp, args.max_samples)
                if stats is None:
                    print(f"  {'[SKIP] '+ds_name:<35}  {err}")
                    continue
                flag = ' <<< INVERTED?' if stats['mean_R11'] > 0.1 else ''
                print(
                    f"  {ds_name:<35} {stats['n']:>7}  "
                    f"{stats['mean_R11']:>+9.4f}  "
                    f"{stats['pct_correct']:>10.1f}%{flag}"
                )
        else:
            print(f"\n  No cache files found in {cache_dir}")

    print()


if __name__ == '__main__':
    main()
