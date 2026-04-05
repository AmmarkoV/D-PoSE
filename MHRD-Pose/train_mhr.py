"""
MHR Training Entry Point.

This script trains the MHR-based HMR model on human pose estimation data.
It is the MHR equivalent of train.py, with all SMPL dependencies removed.

Usage:
    python MHRD-Pose/train_mhr.py --cfg MHRD-Pose/config_mhr.yaml --log_dir logs/mhr

Arguments:
    --cfg: Path to configuration file
    --log_dir: Directory for logs and checkpoints
    --fdr: Fast development run (single batch)
    --test: Run testing instead of training
    --resume: Resume from last checkpoint
    --ckpt: Specific checkpoint to load
"""

import os
import sys
import torch
import time
import yaml
import argparse
import numpy as np
import pytorch_lightning as pl
from loguru import logger
from flatten_dict import flatten, unflatten
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Add project root and MHRD-Pose dir to path
# Note: the directory is named 'MHRD-Pose' (hyphen), which is not a valid Python
# identifier, so we cannot use 'from MHRD-Pose.xxx import yyy'. Instead we
# add the directory itself to sys.path and import its modules directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)   # D-PoSE root  (for 'train.*' imports)
sys.path.insert(0, _THIS_DIR)       # MHRD-Pose dir (for 'mhr_trainer' etc.)

from mhr_trainer import MHRTrainer
from train.utils.train_utils import update_hparams


def train(hparams, fast_dev_run=False):
    """Main training function.

    Args:
        hparams: Hyperparameters
        fast_dev_run: Whether to run fast development mode
    """
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup logging
    logger.add(
        os.path.join(log_dir, 'train_mhr.log'),
        level='INFO',
        colorize=False,
    )

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')

    # Loggers
    wandb_logger = WandbLogger(
        project='Depth HMR MHR',
        log_model=True,
        name=hparams.DATASET.RUN_NAME
    )
    experiment_loggers = []

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        log_graph=False,
    )
    experiment_loggers.append(tb_logger)
    experiment_loggers.append(wandb_logger)

    # Create model
    model = MHRTrainer(hparams=hparams).to(device)

    # Checkpoint callback
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=10,
        mode='min',
    )

    # Trainer
    trainer = pl.Trainer(
        gpus=1,
        logger=experiment_loggers,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        callbacks=[ckpt_callback],
        default_root_dir=log_dir,
        val_check_interval=hparams.DATASET.VAL_INTERVAL,
        num_sanity_val_steps=1,
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.5,
    )

    if args.test:
        logger.info('*** Started testing ***')
        ckpt = hparams.TRAINING.RESUME

        # Load state dict
        ckpt_loaded = torch.load(ckpt)
        ckpt_state_dict = ckpt_loaded['state_dict']
        model_state_dict = model.state_dict()

        # Filter out SMPL keys if present
        for key in ckpt_state_dict.keys():
            if ('smpl' not in key) and ('smplx' not in key):
                model_state_dict[key] = ckpt_state_dict[key]
        model.load_state_dict(model_state_dict, strict=False)

        trainer.test(model)
    else:
        logger.info('*** Started MHR training ***')
        trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--log_dir', type=str, help='log dir path', default='./logs/mhr')
    parser.add_argument('--fdr', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt', type=str)

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')
    torch.cuda.empty_cache()

    # Update hparams with config file and from args
    hparams = update_hparams(args.cfg)
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = os.path.join(args.log_dir, hparams.EXP_NAME + logtime)
    os.makedirs(logdir, exist_ok=True)
    hparams.LOG_DIR = logdir

    if args.ckpt:
        hparams.TRAINING.RESUME = args.ckpt

    # Load the last checkpoint using the epoch id
    if args.resume and hparams.TRAINING.RESUME is None:
        ckpt_files = []
        for root, dirs, files in os.walk(args.log_dir, topdown=False):
            for f in files:
                if f.endswith('.ckpt'):
                    ckpt_files.append(os.path.join(root, f))

        epoch_idx = [int(x.split('=')[-1].split('.')[0]) for x in ckpt_files]
        if len(epoch_idx) == 0:
            ckpt_file = None
        else:
            last_epoch_idx = np.argsort(epoch_idx)[-1]
            ckpt_file = ckpt_files[last_epoch_idx]
        logger.info('Loading CKPT', ckpt_file)
        hparams.TRAINING.RESUME = ckpt_file

    if args.test:
        hparams.RUN_TEST = True

    def save_dict_to_yaml(obj, filename, mode='w'):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # Save final config
    save_dict_to_yaml(
        unflatten(flatten(hparams)),
        os.path.join(hparams.LOG_DIR, 'config_mhr_to_run.yaml')
    )

    train(hparams, fast_dev_run=args.fdr)
