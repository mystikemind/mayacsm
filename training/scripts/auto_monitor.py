#!/usr/bin/env python3
"""
Automated Training Monitor - Runs in background
Checks training every 30 minutes and logs status
"""

import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("/home/ec2-user/SageMaker/project_maya/training/logs/ultimate_training.log")
MONITOR_LOG = Path("/home/ec2-user/SageMaker/project_maya/training/logs/monitor_status.log")
CHECK_INTERVAL = 1800  # 30 minutes

def get_latest_metrics():
    """Parse latest metrics from training log."""
    if not LOG_FILE.exists():
        return None

    with open(LOG_FILE) as f:
        lines = f.readlines()

    # Find latest step
    latest_step = None
    latest_loss = None
    latest_val = None
    best_val = None

    for line in reversed(lines):
        if "Step" in line and "Loss:" in line and latest_step is None:
            parts = line.split("|")
            for p in parts:
                if "Step" in p:
                    latest_step = int(p.split()[1])
                if "Loss:" in p:
                    latest_loss = float(p.split()[1])
                    break
        if "Val Loss:" in line and latest_val is None:
            latest_val = float(line.split("Val Loss:")[1].split()[0])
        if "New best model" in line and best_val is None:
            best_val = latest_val
        if latest_step and latest_val:
            break

    return {
        "step": latest_step,
        "loss": latest_loss,
        "val_loss": latest_val,
        "progress": latest_step / 12900 * 100 if latest_step else 0
    }

def log_status(msg):
    """Log to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"{timestamp} | {msg}"
    print(full_msg)
    with open(MONITOR_LOG, "a") as f:
        f.write(full_msg + "\n")

def main():
    log_status("=" * 60)
    log_status("AUTOMATED TRAINING MONITOR STARTED")
    log_status(f"Checking every {CHECK_INTERVAL // 60} minutes")
    log_status("=" * 60)

    last_step = 0
    stall_count = 0

    while True:
        metrics = get_latest_metrics()

        if metrics is None:
            log_status("WARNING: Could not read training log")
        elif metrics["step"] is None:
            log_status("WARNING: No training steps found")
        else:
            status = f"Step {metrics['step']}/{12900} ({metrics['progress']:.1f}%) | "
            status += f"Loss: {metrics['loss']:.4f}"
            if metrics['val_loss']:
                status += f" | Val: {metrics['val_loss']:.4f}"

            # Check for stall
            if metrics['step'] == last_step:
                stall_count += 1
                if stall_count >= 2:
                    log_status(f"WARNING: Training may have stalled! No progress for {stall_count * CHECK_INTERVAL // 60} minutes")
            else:
                stall_count = 0

            last_step = metrics['step']

            # Estimate remaining time
            if metrics['step'] > 0:
                steps_remaining = 12900 - metrics['step']
                # Approximate: ~100 seconds per step
                hours_remaining = steps_remaining * 100 / 3600
                status += f" | ETA: ~{hours_remaining:.0f}h"

            log_status(status)

            # Check for completion
            if metrics['step'] >= 12900:
                log_status("=" * 60)
                log_status("TRAINING COMPLETE!")
                log_status("=" * 60)
                break

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
