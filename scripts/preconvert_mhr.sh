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

# Defaults
CONFIG="${PROJECT_ROOT}/MHRD-Pose/config_mhr.yaml"
GPU=0
BATCH_SIZE=64
RESUME_FLAG=""
REGENERATE_FLAG=""
SKIP_EXTRACT=0
EXTRA_ARGS=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)          CONFIG="$2";                    shift 2 ;;
        --gpu)             GPU="$2";                       shift 2 ;;
        --batch_size)      BATCH_SIZE="$2";                shift 2 ;;
        --resume)          RESUME_FLAG="--resume";         shift ;;
        --regenerate)      REGENERATE_FLAG="--regenerate"; shift ;;
        --skip-extraction) SKIP_EXTRACT=1;                 shift ;;
        *)                 EXTRA_ARGS="$EXTRA_ARGS $1";    shift ;;
    esac
done

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

echo "============================================="
echo "  MHR Cache Pre-generation"
echo "  Config:     ${CONFIG}"
echo "  GPU:        ${GPU}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Resume:     ${RESUME_FLAG:-no}
  Regenerate: ${REGENERATE_FLAG:-no}
  Extract:    $([ $SKIP_EXTRACT -eq 1 ] && echo no || echo auto)"
echo "============================================="
echo ""

# ---------------------------------------------------------------------------
# Auto-extract training image tars if the corresponding directory is missing.
# Each foo_6fps.tar extracts to foo_6fps/ in the same directory.
# Training does not need images for cache generation, but does for the actual
# training loop — extracting here ensures a single setup step covers both.
# Pass --skip-extraction to disable (e.g. when images are already extracted
# or when running cache-only on a machine without the image storage mounted).
# ---------------------------------------------------------------------------
IMAGES_DIR="${PROJECT_ROOT}/data/training_images"
if [ $SKIP_EXTRACT -eq 0 ] && [ -d "${IMAGES_DIR}" ]; then
    echo "[extract] Checking for unextracted tars in ${IMAGES_DIR} ..."
    found_any=0
    for tar_file in "${IMAGES_DIR}"/*.tar; do
        [ -f "${tar_file}" ] || continue
        dir_name="${tar_file%.tar}"
        if [ ! -d "${dir_name}" ]; then
            found_any=1
            echo "[extract] Extracting $(basename ${tar_file}) ..."
            tar xf "${tar_file}" -C "${IMAGES_DIR}"
            echo "[extract] Done: $(basename ${dir_name})"
        fi
    done
    if [ $found_any -eq 0 ]; then
        echo "[extract] All tars already extracted — nothing to do."
    fi
elif [ $SKIP_EXTRACT -eq 1 ]; then
    echo "[extract] Skipped (--skip-extraction)"
else
    echo "[extract] training_images dir not found at ${IMAGES_DIR} — skipping extraction"
fi
echo ""

cd "${PROJECT_ROOT}"
python MHRD-Pose/preconvert_mhr.py \
    --cfg "${CONFIG}" \
    --batch_size "${BATCH_SIZE}" \
    ${RESUME_FLAG} \
    ${REGENERATE_FLAG} \
    ${EXTRA_ARGS}

echo ""
echo "Pre-conversion complete. You can now run training:"
echo "  ./scripts/train_mhr.sh --config ${CONFIG}"
