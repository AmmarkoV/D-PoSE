#!/bin/bash
# Setup Script for D-Pose MHR Training
# Installs dependencies and prepares the environment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# =============================================================================
# CUDA / PyTorch wheel tag — resolved once, used throughout the script.
# Override by exporting TORCH_CUDA_TAG before running, e.g.:
#   export TORCH_CUDA_TAG=cu124
# =============================================================================
if [ -z "${TORCH_CUDA_TAG}" ]; then
    if command -v nvidia-smi &> /dev/null; then
        SYS_CUDA=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
        CUDA_MAJOR=$(echo "$SYS_CUDA" | cut -d. -f1)
        CUDA_MINOR=$(echo "$SYS_CUDA" | cut -d. -f2)
        if   [ "$CUDA_MAJOR" -eq 11 ];                                        then TORCH_CUDA_TAG="cu118"
        elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -le 3 ];            then TORCH_CUDA_TAG="cu121"
        elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -le 5 ];            then TORCH_CUDA_TAG="cu124"
        else                                                                       TORCH_CUDA_TAG="cu126"
        fi
        echo "[INFO] System CUDA ${SYS_CUDA} → PyTorch wheel tag: ${TORCH_CUDA_TAG}"
    else
        TORCH_CUDA_TAG="cpu"
        echo "[WARNING] No nvidia-smi found — will install CPU-only PyTorch."
    fi
fi

# =============================================================================
# GitHub PAT (Personal Access Token)
# =============================================================================
# GitHub no longer accepts passwords for git operations over HTTPS.
# Export your PAT before running this script:
#
#   export GITHUB_PAT="ghp_xxxxxxxxxxxxxxxxxxxx"
#
# Generate one at: GitHub → Settings → Developer settings →
#                  Personal access tokens → Fine-grained tokens
# Required permission: Contents (read-only) for any private repo,
# or no permissions needed for public repos via authenticated clone.
# =============================================================================
if [ -z "${GITHUB_PAT}" ]; then
    echo ""
    echo "WARNING: GITHUB_PAT is not set."
    echo "  pip installs from GitHub may fail with authentication errors."
    echo "  Export your token before running this script:"
    echo ""
    echo "    export GITHUB_PAT=\"ghp_xxxxxxxxxxxxxxxxxxxx\""
    echo ""
    echo "  Continuing without PAT (will work for public repos without auth)..."
    echo ""
    GIT_PIP_PREFIX="https://github.com"
else
    GIT_PIP_PREFIX="https://${GITHUB_PAT}@github.com"
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Check Python Version
# =============================================================================
log_info "Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "  $PYTHON_VERSION"

# =============================================================================
# Create Virtual Environment
# =============================================================================
VENV_PATH="${PROJECT_ROOT}/venv"

if [ ! -d "$VENV_PATH" ]; then
    log_info "Creating virtual environment at $VENV_PATH..."
    $PYTHON_CMD -m venv $VENV_PATH
    log_success "Virtual environment created"
else
    log_warning "Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source ${VENV_PATH}/bin/activate

# =============================================================================
# Upgrade pip
# =============================================================================
log_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools pkgutil_resolve_name

# =============================================================================
# Install PyTorch
# =============================================================================
log_info "Installing PyTorch (tag: ${TORCH_CUDA_TAG})..."
if [ "$TORCH_CUDA_TAG" = "cpu" ]; then
    pip install --force-reinstall torch torchvision torchaudio
else
    pip install --force-reinstall torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
fi
log_success "PyTorch installed"

# =============================================================================
# Install neural_renderer_pytorch (requires CUDA + nvcc to compile)
# The PyPI package (1.1.3) uses AT_CHECK which was removed in PyTorch 2.x.
# We clone, patch to TORCH_CHECK, and install from source.
# =============================================================================
log_info "Installing neural_renderer_pytorch from source (with PyTorch 2.x patch)..."
if command -v nvcc &> /dev/null; then
    NR_TMP=$(mktemp -d)
    git clone --quiet https://github.com/daniilidis-group/neural_renderer "$NR_TMP/neural_renderer"
    # AT_CHECK was removed in PyTorch 2.x; replace with TORCH_CHECK across all CUDA extension sources
    for f in "$NR_TMP/neural_renderer/neural_renderer/cuda/"*.cpp; do
        sed -i 's/AT_CHECK/TORCH_CHECK/g' "$f"
    done
    pip install --no-build-isolation "$NR_TMP/neural_renderer" || {
        log_warning "neural_renderer_pytorch build failed."
        log_warning "This is a hard dependency of train/losses/losses.py."
    }
    rm -rf "$NR_TMP"
else
    log_warning "nvcc not found — skipping neural_renderer_pytorch build."
    log_warning "This is a hard dependency of train/losses/losses.py."
    log_warning "Install the CUDA toolkit and re-run this script."
fi

# =============================================================================
# Install PyTorch Lightning
# =============================================================================
log_info "Installing PyTorch Lightning..."
pip install "pytorch-lightning>=2.0"
log_success "PyTorch Lightning installed"

# =============================================================================
# Install System Dependencies (jpeg4py requires libturbojpeg)
# =============================================================================
log_info "Installing system dependencies for jpeg4py..."
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y libturbojpeg0-dev || log_warning "Could not install libturbojpeg0-dev — jpeg4py may fail at runtime."
else
    log_warning "apt-get not found. Install libturbojpeg manually for jpeg4py support."
fi

# =============================================================================
# Install Core Dependencies
# =============================================================================
log_info "Installing core dependencies..."
pip install \
    numpy \
    opencv-python \
    Pillow \
    scipy \
    scikit-image \
    matplotlib \
    tqdm \
    loguru \
    pyyaml \
    joblib \
    flatten-dict \
    tensorboard \
    wandb \
    albumentations \
    trimesh \
    meshcat \
    yacs \
    jpeg4py \
    torchmetrics
log_success "Core dependencies installed"

# =============================================================================
# Install smplx
# =============================================================================
log_info "Installing smplx..."
pip install smplx
log_success "smplx installed"

# =============================================================================
# Install PyMomentum (MHR)
# =============================================================================
log_info "Installing PyMomentum (MHR library)..."
log_warning "PyMomentum may require manual installation from NVIDIA."
pip install pymomentum || {
    log_warning "Failed to install pymomentum via pip."
    log_info "Manual installation required:"
    log_info "  1. Clone: git clone https://github.com/NVlabs/momentum-human-rig.git"
    log_info "  2. Install: pip install -e ./momentum-human-rig"
}

# =============================================================================
# Install Additional Utilities
# =============================================================================
log_info "Installing additional utilities..."
pip install \
    einops \
    timm \
    addict
log_success "Additional utilities installed"

# =============================================================================
# Install Renderer (pyrender)
# =============================================================================
log_info "Installing pyrender..."
pip install pyrender
log_success "pyrender installed"

# =============================================================================
# Setup Complete
# =============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Virtual environment created at: ${BLUE}${VENV_PATH}${NC}"
echo ""
echo -e "To activate the environment, run:"
echo -e "  ${BLUE}source ${VENV_PATH}/bin/activate${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. ${YELLOW}Download datasets:${NC}"
echo -e "     ${BLUE}./scripts/download_datasets.sh${NC}"
echo ""
echo -e "  2. ${YELLOW}Configure data paths in:${NC}"
echo -e "     ${BLUE}train/core/config.py${NC}"
echo ""
echo -e "  3. ${YELLOW}Start training:${NC}"
echo -e "     ${BLUE}./scripts/train_mhr.sh --config MHRD-Pose/config_mhr.yaml${NC}"
echo ""

# Keep venv activated if script was run directly
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    log_info "Virtual environment is now activated. You can install additional packages."
fi
