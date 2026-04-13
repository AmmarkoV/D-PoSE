# D-Pose MHR Training Scripts

This directory contains scripts for setting up, downloading data, and training the MHR-based D-Pose model.

## Quick Start

```bash
# 1. Setup environment
./setup.sh

# 2. Download datasets
./download_datasets.sh

# 3. (Optional) Pre-convert SMPL data to MHR
python preconvert_to_mhr.py --all

# 4. Train the model
./train_mhr.sh --config ../MHRD-Pose/config_mhr.yaml
```

## Scripts Overview

### `setup.sh`
Installs all dependencies and creates a virtual environment.

```bash
./setup.sh
```

**What it does:**
- Creates virtual environment at `venv/`
- Installs PyTorch with CUDA support (if available)
- Installs PyTorch Lightning, smplx, MPT, and other dependencies
- Installs PyMomentum (MHR library)

### `download_datasets.sh`
Downloads required datasets and pretrained models.

```bash
./download_datasets.sh
```

**What it downloads:**
- HRNet pretrained checkpoints (automatically)
- Instructions for manual downloads (SMPL, SMPL-X, AGORA, 3DPW)

### `preconvert_to_mhr.py`
Pre-converts SMPL ground truth to MHR parameters for efficient training.

```bash
# Convert all datasets
python preconvert_to_mhr.py --all

# Convert specific dataset
python preconvert_to_mhr.py --dataset agora-bfh

# With custom batch size
python preconvert_to_mhr.py --all --batch_size 128
```

### `train_mhr.sh`
Trains the MHR-based HMR model.

```bash
# Basic training
./train_mhr.sh --config ../MHRD-Pose/config_mhr.yaml

# Fast development run (single batch)
./train_mhr.sh --config ../MHRD-Pose/config_mhr.yaml --fast_dev

# Test with checkpoint
./train_mhr.sh --config ../MHRD-Pose/config_mhr.yaml --test --ckpt path/to/checkpoint.ckpt

# Resume training
./train_mhr.sh --config ../MHRD-Pose/config_mhr.yaml --resume

# Use specific GPU
./train_mhr.sh --config ../MHRD-Pose/config_mhr.yaml --gpu 0
```

## Dataset Download Sources

### Required Datasets

| Dataset | Purpose | Download |
|---------|---------|----------|
| **SMPL** | Body model | https://smpl.isic.ucl.ac.uk/ |
| **SMPL-X** | Body model with hands/face | https://smpl-x.is.tue.mpg.de/ |
| **AGORA** | Primary training data | https://agarwal-anab.github.io/agora/ |

### Synthetic Training Data

| Dataset | Purpose | Download |
|---------|---------|----------|
| **Bedlam** | Synthetic training data | Register at https://bedlam.is.tue.mpg.de/, then run `./BEDLAM/fetch_training_data.sh` |
| **3DPW** | Validation | https://vcai.maxplanck.org/3dpw/ |
| **EMDB** | Evaluation | https://ommer-lab.com/recovery/emdb/ |

### Pretrained Models

| Model | Download |
|-------|----------|
| **HRNet W32** | https://download.openmmlab.com/hrnet/hrnetv2/hrnetv2_w32-36af84mm.pth |
| **HRNet W48** | https://download.openmmlab.com/hrnet/hrnetv2/hrnetv2_w48-8f125fad.pth |

## Directory Structure After Setup

```
D-PoSE/
├── data/
│   ├── body_models/
│   │   ├── SMPL_python_v.1.1.0/smpl/models/    # SMPL models
│   │   └── smplx/models/smplx/                  # SMPL-X models
│   ├── ckpt/
│   │   └── pretrained/                          # HRNet checkpoints
│   ├── training_images/                         # Training images
│   ├── training_labels/                         # Training annotations
│   ├── eval_data_parsed/                        # Validation data
│   └── mhr_cache/                               # Pre-converted MHR data
├── logs/
│   └── mhr/                                     # Training logs
├── venv/                                        # Virtual environment
├── MHRD-Pose/                                   # MHR training code
└── scripts/                                     # These scripts
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```yaml
DATASET:
  BATCH_SIZE: 32  # Default is 64
```

### MHR Model Not Found
Ensure `mhr_model.pt` exists in project root:
```bash
ls -la mhr_model.pt
```

### SMPLX Import Error
Install smplx:
```bash
pip install smplx
```

### Bedlam BBox Warning
Enable bedlam_bbox in config for proper training:
```yaml
TRIAL:
  bedlam_bbox: true
```

## Expected Training Time

| Dataset | GPU | Time to Convergence |
|---------|-----|---------------------|
| AGORA only | RTX 3090 | ~24 hours |
| AGORA + Bedlam | RTX 3090 | ~48 hours |

## Memory Requirements

- **GPU**: Minimum 12GB VRAM (16GB+ recommended)
- **RAM**: 32GB minimum, 64GB recommended
- **Disk**: 100GB for datasets, 50GB for checkpoints
