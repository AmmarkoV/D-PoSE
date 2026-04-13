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
# Ensure Python 3.12 is available (required for pymomentum from conda-forge)
# pymomentum>=0.1.90 only has conda-forge builds for Python >=3.12.
# =============================================================================
log_info "Checking for Python 3.12 (required for pymomentum)..."
if ! command -v python3.12 &> /dev/null; then
    log_info "Python 3.12 not found — installing via deadsnakes PPA..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
        log_success "Python 3.12 installed"
    else
        log_warning "apt-get not available — install Python 3.12 manually"
    fi
else
    log_success "Python 3.12 already available: $(python3.12 --version)"
fi

# Use Python 3.12 if available, otherwise fall back to default
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    log_info "Using Python 3.12 for venv"
fi

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
log_info "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y \
        libturbojpeg0-dev \
        libopenexr-dev \
        || log_warning "Could not install some system dependencies — jpeg4py or OpenEXR may fail at runtime."
else
    log_warning "apt-get not found. Install libturbojpeg and libopenexr manually."
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
    torchmetrics \
    OpenEXR
log_success "Core dependencies installed"

# =============================================================================
# Install smplx
# =============================================================================
log_info "Installing smplx..."
pip install smplx
log_success "smplx installed"

# =============================================================================
# Install chumpy (required to load legacy SMPL .pkl files)
# chumpy's setup.py uses `import pip` which was removed from newer pip,
# so --no-build-isolation is required.
# =============================================================================
log_info "Installing chumpy (SMPL .pkl loader dependency)..."
pip install chumpy --no-build-isolation
# Patch chumpy for Python 3.11+ (getargspec removed) and NumPy 1.24+ (deprecated aliases removed)
# Note: chumpy crashes on import before patching, so use find instead of python -c
CHUMPY_PATH=$(find "${VENV_PATH}" -name "ch.py" -path "*/chumpy/*" | head -1 | xargs dirname)
# Fix 1: inspect.getargspec removed in Python 3.11
sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' "$CHUMPY_PATH/ch.py"
# Fix 2: numpy no longer exports built-in aliases (bool, int, float, etc.) — strip them from the import
sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' \
    "$CHUMPY_PATH/__init__.py"
log_success "chumpy installed and patched"

# =============================================================================
# Install PyMomentum (MHR) via conda-forge
# The pip package 'pymomentum' is a stub (no actual C++ bindings).
# The real package (>=0.1.90) with geometry bindings is only on conda-forge.
# We install miniforge, install pymomentum there, then symlink into the venv.
# =============================================================================
log_info "Installing PyMomentum via conda-forge (miniforge)..."

MINIFORGE_DIR="${PROJECT_ROOT}/miniforge3"
# pymomentum requires Python >=3.12; use a dedicated conda env for it
PYMOMENTUM_CONDA_ENV="${MINIFORGE_DIR}/envs/pymomentum_env"
VENV_SP="${VENV_PATH}/lib/python$(${PYTHON_CMD} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"

if [ ! -d "${MINIFORGE_DIR}" ]; then
    log_info "Downloading Miniforge installer..."
    wget -q -O /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "${MINIFORGE_DIR}"
    rm -f /tmp/miniforge.sh
    log_success "Miniforge installed at ${MINIFORGE_DIR}"
else
    log_warning "Miniforge already exists at ${MINIFORGE_DIR}"
fi

log_info "Creating conda env with Python 3.12 and pymomentum..."
"${MINIFORGE_DIR}/bin/mamba" create -y -n pymomentum_env python=3.12 "pymomentum>=0.1.90" -c conda-forge || \
    "${MINIFORGE_DIR}/bin/conda" create -y -n pymomentum_env python=3.12 "pymomentum>=0.1.90" -c conda-forge || {
        log_warning "Failed to install pymomentum from conda-forge."
        log_warning "MHR on-the-fly conversion will not work without it."
    }

# Symlink pymomentum from the conda env into the venv's site-packages
CONDA_SP="${PYMOMENTUM_CONDA_ENV}/lib/python3.12/site-packages"
if [ -d "${CONDA_SP}/pymomentum" ]; then
    ln -sf "${CONDA_SP}/pymomentum" "${VENV_SP}/pymomentum"
    log_success "pymomentum symlinked into venv from ${CONDA_SP}"
else
    log_warning "pymomentum not found in ${CONDA_SP} — symlink skipped"
fi

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
