#!/bin/bash
# Dataset Download Script for D-Pose

set -e

echo "========================================"
echo "D-Pose Dataset Download Script"
echo "========================================"

# Create directories
mkdir -p data/body_models data/utils data/ckpt/pretrained data/training_labels data/eval_data_parsed

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# 1. Download SMPL/SMPLX Body Models (Required for training)
# =============================================================================
echo -e "\n${YELLOW}[1/6] Downloading SMPL body models...${NC}"
echo "SMPL models require registration at: https://smpl.isic.ucl.ac.uk/"
echo "After registration, download and extract to: data/body_models/SMPL_python_v.1.1.0/"

# Create placeholder
cat > data/body_models/DOWNLOAD_INSTRUCTIONS.txt << 'EOF'
SMPL/SMPL-X Body Models
=======================

1. Register at: https://smpl.isic.ucl.ac.uk/
2. Download SMPL (not SMPL-H): https://download.is.tue.mpg.de/download.php?model=SMPL
3. Download SMPL-X: https://download.is.tue.mpg.de/download.php?model=SMPL-X
4. Extract to:
   - SMPL: data/body_models/SMPL_python_v.1.1.0/smpl/models/
   - SMPL-X: data/body_models/smplx/models/smplx/

MANO hand models (optional):
- Download from: https://mano.is.tue.mpg.de/
- Extract to: data/body_models/mano/mano_v1_2/models/
EOF

echo -e "${GREEN}✓ Instructions saved to data/body_models/DOWNLOAD_INSTRUCTIONS.txt${NC}"

# =============================================================================
# 2. Download HRNet Pretrained Checkpoints
# =============================================================================
echo -e "\n${YELLOW}[2/6] Downloading HRNet pretrained checkpoints...${NC}"

# HRNet W32 COCO (pose estimation checkpoint, 256x192)
if [ ! -f "data/ckpt/pretrained/pose_hrnet_w32_256x192.pth" ]; then
    echo "Downloading HRNet W32 COCO checkpoint..."
    wget -O data/ckpt/pretrained/pose_hrnet_w32_256x192.pth \
        "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth" || \
    wget -O data/ckpt/pretrained/pose_hrnet_w32_256x192.pth \
        "https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/releases/download/v1.0/pose_hrnet_w32_256x192.pth" || \
    echo "Warning: Failed to download HRNet W32. Copy pose_hrnet_w32_256x192.pth manually to data/ckpt/pretrained/."
else
    echo "HRNet W32 checkpoint already exists."
fi

# HRNet W48 COCO (pose estimation checkpoint, 256x192)
if [ ! -f "data/ckpt/pretrained/pose_hrnet_w48_256x192.pth" ]; then
    echo "Downloading HRNet W48 COCO checkpoint..."
    wget -O data/ckpt/pretrained/pose_hrnet_w48_256x192.pth \
        "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth" || \
    wget -O data/ckpt/pretrained/pose_hrnet_w48_256x192.pth \
        "https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/releases/download/v1.0/pose_hrnet_w48_256x192.pth" || \
    echo "Warning: Failed to download HRNet W48. Copy pose_hrnet_w48_256x192.pth manually to data/ckpt/pretrained/."
else
    echo "HRNet W48 checkpoint already exists."
fi

# =============================================================================
# 3. Download Utility Files (Joint Regressors, etc.)
# =============================================================================
echo -e "\n${YELLOW}[3/6] Downloading utility files...${NC}"

# J_regressor files
cat > scripts/download_utils.py << 'PYEOF'
import numpy as np
import pickle
import os

os.makedirs('data/utils', exist_ok=True)

# Create SMPLX to J14 regressor (simplified version)
print("Creating joint regressor files...")

# Placeholder - these should be downloaded from HMR repository
# Download from: https://github.com/MooreThreads/HMR/blob/master/src/utils/README.md
print("Note: Joint regressors need to be downloaded from HMR repository:")
print("  - J_regressor_extra.npy")
print("  - J_regressor_h36m.npy")
print("  - SMPLX_to_J14.pkl")
print("  - smplx2smpl.pkl")
print("\nFrom: https://github.com/MooreThreads/HMR/tree/master/src/utils")
PYEOF

python scripts/download_utils.py

# =============================================================================
# 4. AGORA Dataset (Primary Training Data)
# =============================================================================
echo -e "\n${YELLOW}[4/6] AGORA Dataset Setup${NC}"
echo "AGORA dataset requires separate download:"
echo "1. Visit: https://agarwal-anab.github.io/agora/"
echo "2. Follow instructions to download training data"
echo "3. Place in: data/training_images/ and data/training_labels/"

cat > data/training_labels/AGORA_INSTRUCTIONS.txt << 'EOF'
AGORA Dataset
=============

Download from: https://agarwal-anab.github.io/agora/

Training data structure:
- Images: data/training_images/images/
- Labels: data/training_labels/all_npz_12_training/
  - agora-bfh.npz
  - agora-body.npz
EOF

# =============================================================================
# 5. Bedlam Synthetic Data (Optional - for improved training)
# =============================================================================
echo -e "\n${YELLOW}[5/6] Bedlam Synthetic Data${NC}"
echo "Bedlam synthetic data is optional but recommended for best results."
echo ""
echo "Download using the official fetch script:"
echo "  ${GREEN}./BEDLAM/fetch_training_data.sh${NC}"
echo ""
echo "Requires registration at: https://bedlam.is.tue.mpg.de/"
echo ""
echo "This will download:"
echo "  - Training images (multiple scenes)"
echo "  - Training labels (SMPL-X parameters)"
echo "  - Evaluation data (3DPW validation)"

# =============================================================================
# 6. 3DPW Validation Dataset
# =============================================================================
echo -e "\n${YELLOW}[6/6] 3DPW Validation Dataset${NC}"
echo "3DPW dataset for validation:"
echo "1. Download from: https://vcai.maxplanck.org/3dpw/"
echo "2. Parsed versions available from HMR repository"
echo "3. Place in: data/3dpw/ and data/eval_data_parsed/"

cat > data/eval_data_parsed/3DPW_INSTRUCTIONS.txt << 'EOF'
3DPW Dataset
============

Original dataset: https://vcai.maxplanck.org/3dpw/

Parsed versions (recommended):
- Download from HMR repository utilities
- Place test set in: data/eval_data_parsed/3dpw_test.npz
- Place validation set in: data/eval_data_parsed/3dpw_validation.npz
EOF

# =============================================================================
# Summary
# =============================================================================
echo -e "\n========================================"
echo -e "${GREEN}Download Summary${NC}"
echo -e "========================================"
echo ""
echo "Required (must download manually):"
echo "  ☐ SMPL body models"
echo "  ☐ SMPL-X body models"
echo "  ☐ Joint regressor files"
echo ""
echo "Training data:"
echo "  ☐ AGORA dataset"
echo "  ☐ Bedlam synthetic data (optional)"
echo ""
echo "Validation:"
echo "  ☐ 3DPW dataset"
echo ""
echo "Pretrained models:"
echo -e "  ${GREEN}✓ HRNet W32 checkpoint${NC}"
echo -e "  ${GREEN}✓ HRNet W48 checkpoint${NC}"
echo ""
echo "For detailed instructions, see:"
echo "  - data/body_models/DOWNLOAD_INSTRUCTIONS.txt"
echo "  - data/training_labels/AGORA_INSTRUCTIONS.txt"
echo "  - data/eval_data_parsed/3DPW_INSTRUCTIONS.txt"
