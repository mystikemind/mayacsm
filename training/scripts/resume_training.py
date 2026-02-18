#!/usr/bin/env python3
"""
Resume Training Script - For Recovery After Interruption
=========================================================
Usage:
    python resume_training.py --checkpoint training/checkpoints/csm_maya_ultimate/epoch-8

Or automatically find the latest checkpoint:
    python resume_training.py --auto
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints" / "csm_maya_ultimate"
DATA_DIR = PROJECT_ROOT / "training" / "data" / "csm_ready_ex04"


def find_latest_checkpoint():
    """Find the checkpoint with the highest step number."""
    if not CHECKPOINT_DIR.exists():
        return None

    latest_step = -1
    latest_checkpoint = None

    for checkpoint_dir in CHECKPOINT_DIR.iterdir():
        if not checkpoint_dir.is_dir():
            continue

        state_file = checkpoint_dir / "training_state.json"
        if not state_file.exists():
            continue

        with open(state_file) as f:
            state = json.load(f)

        step = state.get("global_step", 0)
        if step > latest_step:
            latest_step = step
            latest_checkpoint = checkpoint_dir

    return latest_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--auto", action="store_true", help="Auto-find latest checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs (will continue from checkpoint)")
    args = parser.parse_args()

    # Find checkpoint
    if args.auto:
        checkpoint_dir = find_latest_checkpoint()
        if checkpoint_dir is None:
            print("ERROR: No checkpoints found!")
            sys.exit(1)
        print(f"Found latest checkpoint: {checkpoint_dir}")
    elif args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        if not checkpoint_dir.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_dir}")
            sys.exit(1)
    else:
        print("ERROR: Must specify --checkpoint or --auto")
        sys.exit(1)

    # Load checkpoint state
    state_file = checkpoint_dir / "training_state.json"
    with open(state_file) as f:
        state = json.load(f)

    print(f"\n=== RESUMING TRAINING ===")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Global step: {state['global_step']}")
    print(f"Epoch: {state['epoch']}")
    print(f"Best val loss: {state['best_val_loss']:.4f}")
    print(f"Total epochs: {args.epochs}")
    print(f"========================\n")

    # Import trainer
    from training.scripts.resume_training_impl import resume_training
    resume_training(checkpoint_dir, args.epochs)


if __name__ == "__main__":
    main()
