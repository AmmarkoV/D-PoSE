#!/bin/bash
# Pre-convert datasets to MHR cache files.
# Run this ONCE before training to avoid per-sample SMPL→MHR optimisation
# during training (which would be ~200 iterations × N samples = very slow).
#
# Usage:
#   ./scripts/preconvert_mhr.sh [--config path/to/config.yaml] [--gpu N]
#   ./scripts/preconvert_mhr.sh --resume          # skip already-cached datasets
#
# After this completes, train_mhr.sh will load from cache automatically.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PROJECT_ROOT}/venv"


# Activate venv
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "[WARNING] No venv found at ${VENV_PATH} — using system Python"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

# Preload conda's libstdc++ if needed (pymomentum requires GLIBCXX_3.4.31+
# which is newer than the system libstdc++ in the Docker base image)
CONDA_LIBSTDCXX="${PROJECT_ROOT}/miniforge3/envs/pymomentum_env/lib/libstdc++.so.6"
if [ -f "${CONDA_LIBSTDCXX}" ]; then
    export LD_PRELOAD="${CONDA_LIBSTDCXX}"
    echo "[INFO] Preloading ${CONDA_LIBSTDCXX} for pymomentum"
fi
 

cd "${PROJECT_ROOT}"
python MHRD-Pose/debug_vis.py "$@" 

