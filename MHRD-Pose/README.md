# MHRD-Pose: MHR-based D-Pose Training Pipeline

This directory contains the MHR-based training pipeline for D-Pose, with all SMPL/SMPL-X
dependencies removed from the neural network. The trained model predicts MHR parameters
directly from images, eliminating the 100–200 ms per-frame SMPL→MHR conversion that makes
`demo_webcam.py` slow.

## Why this exists

The original D-Pose pipeline is:
```
Image → HMR → SMPL-X vertices → [Optimization fit] → MHR parameters
                                  ^^^^^^^^^^^^^^^^^^^
                                  ~100–200 ms/person, kills real-time
```

MHRD-Pose trains a model that does:
```
Image → MHRHMR → MHR parameters directly
                  ^^^^^^^^^^^^^^^^^^^^^^
                  ~50 ms, same cost as original HMR inference
```

---

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐
│   Image     │──▶│ HRNet Back-  │──▶│ MHR Regressor│──▶│  MHR Head  │
│  [3,224,224]│   │   bone       │   │              │   │            │
└─────────────┘   └──────────────┘   └──────────────┘   └─────┬──────┘
                                                               │
                                                               ▼
                                                        ┌─────────────┐
                                                        │  MHR Model  │
                                                        │  (frozen)   │
                                                        └──────┬──────┘
                                                               │
                                                  ┌────────────┴────────────┐
                                                  ▼                         ▼
                                           Vertices [V,3]          Joints 3D [24,3]
                                           Joints 2D [24,2]        skel_state [24,16]
```

The MHR model is **frozen** during training (wrapped in `torch.no_grad()`). Only the
backbone and regressor weights are trained.

---

## MHR Parameter Structure

| Parameter | Shape | Description | Unit |
|-----------|-------|-------------|------|
| `identity_coeffs` | [B, 45] | Body shape blendshapes | — |
| `lbs_model_params` | [B, 144] | Pose (24 joints × 6D rotation) | — |
| `face_expr_coeffs` | [B, 72] | Face expression blendshapes | — |
| `pred_cam` | [B, 3] | Camera [scale, tx, ty] | — |
| Vertices output | [B, V, 3] | Mesh vertices from MHR forward | meters |
| Joints output | [B, 24, 3] | Extracted from `skel_state[:,12:15]` | meters |

MHR internally uses **centimetres**; the head converts to metres (×0.01) before returning.

---

## Files

| File | Description |
|------|-------------|
| `train_mhr.py` | Entry point — equivalent of `../train.py` |
| `config_mhr.yaml` | Training configuration |
| `mhr_trainer.py` | PyTorch Lightning module |
| `mhr_hmr.py` | Full model (backbone + regressor + head) |
| `mhr_regressor.py` | Iterative parameter regressor |
| `mhr_head.py` | MHR forward pass + 2D projection |
| `mhr_losses.py` | Loss functions (2D/3D keypoints, params, vertices) |
| `mhr_constants.py` | Dimension constants |
| `dataset_wrapper.py` | Wraps `DatasetHMR` with SMPL→MHR conversion |
| `config.py` | Path constants (MHR_MODEL_PT, etc.) |

---

## Quick Start

```bash
# From project root (D-PoSE/)

# 1. Pre-convert training data to MHR cache (do this once, before training)
python scripts/preconvert_to_mhr.py --dataset agora-bfh --output_dir data/mhr_cache
# or convert everything at once (takes hours but only done once):
python scripts/preconvert_to_mhr.py --all --output_dir data/mhr_cache

# 2. Train
python MHRD-Pose/train_mhr.py --cfg MHRD-Pose/config_mhr.yaml --log_dir logs/mhr

# 3. Fast dev-run sanity check (single batch, no epoch)
python MHRD-Pose/train_mhr.py --cfg MHRD-Pose/config_mhr.yaml --fdr

# 4. Test with a checkpoint
python MHRD-Pose/train_mhr.py --cfg MHRD-Pose/config_mhr.yaml --test --ckpt path/to/epoch=X.ckpt
```

Or use the helper shell script:
```bash
./scripts/train_mhr.sh                          # basic train
./scripts/train_mhr.sh --fast_dev               # single-batch sanity check
./scripts/train_mhr.sh --test --ckpt foo.ckpt  # evaluation
./scripts/train_mhr.sh --resume                 # resume last checkpoint
```

---

## Dataset Setup

### Required datasets

| Dataset | Purpose | Approx. size | Where to get |
|---------|---------|-------------|--------------|
| **BEDLAM** (full) | Primary training — synthetic, body+depth | ~400 GB (images+depth) | [bedlam.is.tue.mpg.de](https://bedlam.is.tue.mpg.de) — register, then `BEDLAM/fetch_training_data.sh` |
| **3DPW** | Validation (required) | ~16 GB | [virtualhumans.mpi-inf.mpg.de/3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW) |

### Optional extra training data

| Dataset | Purpose | Approx. size | Where to get |
|---------|---------|-------------|--------------|
| **AGORA** (body subset) | Synthetic training, diverse body shapes | ~25 GB | [agora.is.tue.mpg.de](https://agora.is.tue.mpg.de) |
| **RICH** | Real-world evaluation | ~30 GB | [rich.is.tue.mpg.de](https://rich.is.tue.mpg.de) |
| **EMDB** | Extra evaluation | ~8 GB | [ommer-lab.com/research/emdb](https://ommer-lab.com/research/emdb) |
| **H36M** | Real-image training | ~40 GB | [vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m) |
| **COCO** | 2D keypoint supervision | ~20 GB | [cocodataset.org](https://cocodataset.org) |
| **MPII** | 2D keypoint supervision | ~13 GB | [human-pose.mpi-inf.mpg.de](http://human-pose.mpi-inf.mpg.de) |

> **Recommendation**: Start with a single BEDLAM sequence (e.g., `zoom-suburbd`, ~8 GB)
> plus 3DPW to validate the pipeline end-to-end before downloading the full dataset.

### Required body models and checkpoints

| File | Size | Where to get |
|------|------|-------------|
| `data/body_models/smplx/models/smplx/` | ~700 MB | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) (needed for pre-conversion only) |
| `data/body_models/SMPL_python_v.1.1.0/smpl/models/` | ~100 MB | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) |
| `data/ckpt/pretrained/pose_hrnet_w32_256x192.pth` | ~120 MB | [HRNet release](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) |
| `data/utils/` (J_regressor_h36m.npy, smplx2smpl.pkl, …) | ~50 MB | included in original D-Pose repo / gDrive |
| `assets/mhr_model.pt` | ~80 MB | included in this repo (`assets/`) |

### Directory layout expected by the code

```
D-PoSE/
├── assets/
│   ├── mhr_model.pt                         ← MHR model weights
│   ├── lod1.fbx
│   ├── corrective_blendshapes_lod1.npz
│   └── corrective_activation.npz
├── data/
│   ├── body_models/
│   │   ├── SMPL_python_v.1.1.0/smpl/models/ ← SMPL
│   │   └── smplx/models/smplx/              ← SMPL-X
│   ├── ckpt/pretrained/                     ← HRNet pretrained weights
│   ├── utils/                               ← J_regressor_h36m.npy, smplx2smpl.pkl, …
│   ├── training_images/                     ← BEDLAM/AGORA images
│   │   ├── 20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png/
│   │   └── …
│   ├── training_labels/
│   │   └── all_npz_12_training/             ← per-sequence .npz annotation files
│   ├── bedlam_download/                     ← depth maps
│   │   └── <seq>/depth/
│   ├── eval_data_parsed/
│   │   ├── 3dpw_validation.npz
│   │   └── 3dpw_test.npz
│   └── mhr_cache/                           ← pre-converted MHR parameters (auto-created)
│       └── {dataset}_mhr_params.npz
├── MHR/                                     ← MHR library (pip install -e MHR/)
├── MHRD-Pose/                               ← this directory
└── logs/mhr/                                ← training logs and checkpoints
```

---

## Offline Pre-conversion (SMPL → MHR)

Training data is originally annotated in SMPL/SMPL-X format. Before training, convert it
to MHR parameters once and cache the results:

```bash
# Single dataset (fastest way to test the pipeline)
python scripts/preconvert_to_mhr.py --dataset zoom-suburbd --output_dir data/mhr_cache

# All training datasets (run overnight)
python scripts/preconvert_to_mhr.py --all --batch_size 128 --output_dir data/mhr_cache
```

The converter (`MHR/tools/mhr_smpl_conversion/conversion.py`) runs an optimisation-based
fit (the same one used in `demo_webcam.py`) but does it **offline in large batches** rather
than per-frame at inference time. Typical throughput: ~100–300 samples/second on a GPU,
so a 300K-sample BEDLAM split takes roughly 15–45 minutes.

`DatasetHMR` in `dataset_wrapper.py` auto-detects the cache and loads from it. If no cache
exists for a dataset, it falls back to on-the-fly conversion (slow — only acceptable for
tiny test datasets).

---

## Known bugs fixed

These bugs were present in the original generated code and have been corrected:

| File | Issue | Fix |
|------|-------|-----|
| `train_mhr.py` | `from MHRD_Pose.mhr_trainer` — hyphenated directory not importable | Add `MHRD-Pose/` to `sys.path`, import directly |
| `train_mhr.py` | `hparams.get('RUN_NAME', …)` — yacs CfgNode is not a dict | Use `hparams.DATASET.RUN_NAME` |
| `mhr_trainer.py` | `from ..utils.eval_utils` relative import resolves wrong | `from train.utils.eval_utils` |
| `mhr_trainer.py` | `error_verts` crashes when `gt_cam_vertices is None` | Guard with None check |
| `mhr_hmr.py` | `from .mhr_regressor` etc. — relative imports fail when loaded via sys.path | Remove leading dots |
| `mhr_hmr.py` | `return mhr_forward_output` at end of function was outside `bedlam_bbox` block → `NameError` | Raise clear `RuntimeError`; always return 5-tuple inside block |
| `dataset_wrapper.py` | `from ..dataset.dataset` relative import | `from train.dataset.dataset` |
| `dataset_wrapper.py` | `from MHR.tools…conversion import Conversion` — no `__init__.py` in `MHR/tools` | Add conversion dir to `sys.path`, then `from conversion import Conversion` |
| `train/core/config.py` | Missing `MODEL.IDENTITY_LOSS_WEIGHT`, `MODEL.EXPR_LOSS_WEIGHT`, `MHR` block — yacs rejects unknown keys from `config_mhr.yaml` | Added all defaults |
| `config_mhr.yaml` | `bedlam_bbox: False` — `MHRHMR.forward()` only implements the `True` path | Set to `True` |

---

## Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 12 GB | 24 GB (RTX 3090 / A5000) |
| RAM | 32 GB | 64 GB |
| Disk (datasets) | 20 GB (single BEDLAM seq + 3DPW) | 500 GB (full BEDLAM + extras) |
| Disk (cache) | 5 GB per 100K samples | ~15 GB for full BEDLAM |
| Disk (checkpoints) | 5 GB | 20 GB |

Reduce `BATCH_SIZE` in `config_mhr.yaml` (default 64) if you run out of VRAM.

---

## Troubleshooting

**`RuntimeError: MHRHMR requires TRIAL.bedlam_bbox=True`**
Set `bedlam_bbox: True` under the `TRIAL:` section in `config_mhr.yaml`.

**`KeyError: MHR` when loading config**
Ensure you are running from the project root (`D-PoSE/`) so that `train/core/config.py`
is on the path. The `hparams.MHR` block must be present in the base config.

**`ModuleNotFoundError: No module named 'mhr'`**
Install the MHR package: `pip install -e MHR/`

**`ModuleNotFoundError: No module named 'conversion'`**
The `MHR/tools/mhr_smpl_conversion/` directory must be on `sys.path`.
`dataset_wrapper.py` and `scripts/preconvert_to_mhr.py` do this automatically.

**`FileNotFoundError: MHR model not found`**
Check that `assets/mhr_model.pt` exists in the project root.

**CUDA out of memory during training**
```yaml
DATASET:
  BATCH_SIZE: 32   # halve from default 64
```

**CUDA out of memory during pre-conversion**
```bash
python scripts/preconvert_to_mhr.py --all --batch_size 32
```
