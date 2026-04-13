#!/bin/bash
# Setup Script for D-Pose MHR Training
# Installs dependencies and prepares the environment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_CMD="${PYTHON_CMD:-python3}"

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
pip install --upgrade pip wheel setuptools

# =============================================================================
# Install PyTorch
# =============================================================================
log_info "Installing PyTorch..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$CUDA_VERSION" ]; then
        log_info "CUDA detected. Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_warning "nvidia-smi found but couldn't determine CUDA version. Installing CPU-only PyTorch."
        pip install torch torchvision torchaudio
    fi
else
    log_warning "No CUDA detected. Installing CPU-only PyTorch."
    pip install torch torchvision torchaudio
fi
log_success "PyTorch installed"

# =============================================================================
# Install PyTorch Lightning
# =============================================================================
log_info "Installing PyTorch Lightning..."
pip install pytorch-lightning==1.9.0
log_success "PyTorch Lightning installed"

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
    meshcat
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
