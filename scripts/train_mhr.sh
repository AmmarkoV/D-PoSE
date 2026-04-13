#!/bin/bash
# MHR Training Script for D-Pose
# This script trains the MHR-based HMR model

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_FILE="${PROJECT_ROOT}/MHRD-Pose/config_mhr.yaml"
LOG_DIR="${PROJECT_ROOT}/logs/mhr"
FAST_DEV_RUN=false
TEST_MODE=false
RESUME=false
CKPT_PATH=""
GPU="0"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --fast_dev)
            FAST_DEV_RUN=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config PATH   Path to config file (default: MHRD-Pose/config_mhr.yaml)"
            echo "  --log_dir PATH  Log directory (default: logs/mhr)"
            echo "  --fast_dev      Fast development run (single batch)"
            echo "  --test          Run testing instead of training"
            echo "  --resume        Resume from last checkpoint"
            echo "  --ckpt PATH     Specific checkpoint to load"
            echo "  --gpu ID        GPU ID (default: 0)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set GPU
echo -e "${BLUE}Setting GPU to: $GPU${NC}"
export CUDA_VISIBLE_DEVICES=$GPU

# Source virtual environment if it exists
if [ -d "${PROJECT_ROOT}/venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source ${PROJECT_ROOT}/venv/bin/activate
else
    echo -e "${YELLOW}Warning: Virtual environment not found at ${PROJECT_ROOT}/venv${NC}"
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Build command
CMD="python ${PROJECT_ROOT}/MHRD-Pose/train_mhr.py \
    --cfg ${CONFIG_FILE} \
    --log_dir ${LOG_DIR}"

if [ "$FAST_DEV_RUN" = true ]; then
    CMD="$CMD --fdr"
fi

if [ "$TEST_MODE" = true ]; then
    CMD="$CMD --test"
fi

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

if [ -n "$CKPT_PATH" ]; then
    CMD="$CMD --ckpt ${CKPT_PATH}"
fi

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}MHR Training Configuration${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "Config file:   ${GREEN}${CONFIG_FILE}${NC}"
echo -e "Log directory: ${GREEN}${LOG_DIR}${NC}"
echo -e "GPU:           ${GREEN}${GPU}${NC}"
echo -e "Fast dev:      ${GREEN}${FAST_DEV_RUN}${NC}"
echo -e "Test mode:     ${GREEN}${TEST_MODE}${NC}"
echo -e "Resume:        ${GREEN}${RESUME}${NC}"
echo -e "Checkpoint:    ${GREEN}${CKPT_PATH:-N/A}${NC}"
echo -e "\n${YELLOW}Command:${NC}"
echo -e "${BLUE}$CMD${NC}"
echo -e "\n${YELLOW}Starting training...${NC}\n"

# Run training
eval $CMD
