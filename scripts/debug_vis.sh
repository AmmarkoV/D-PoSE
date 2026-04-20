#!/bin/bash
#
# Usage:
#      docker exec -it mhrd-pose-container bash -c "cd /home/user/workspace && /home/user/workspace/scripts/debug_vis.sh --n 6 --dataset agora-bfh --test --cfg MHRD-Pose/config_mhr.yaml"
#

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

#export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES="1"

# Preload conda's libstdc++ if needed (pymomentum requires GLIBCXX_3.4.31+
# which is newer than the system libstdc++ in the Docker base image)
CONDA_LIBSTDCXX="${PROJECT_ROOT}/miniforge3/envs/pymomentum_env/lib/libstdc++.so.6"
if [ -f "${CONDA_LIBSTDCXX}" ]; then
    export LD_PRELOAD="${CONDA_LIBSTDCXX}"
    echo "[INFO] Preloading ${CONDA_LIBSTDCXX} for pymomentum"
fi
 

cd "${PROJECT_ROOT}"
python MHRD-Pose/debug_vis.py "$@" 

