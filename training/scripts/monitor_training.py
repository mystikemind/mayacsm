#!/usr/bin/env python3
"""
ULTIMATE Training Monitor - Senior AI Engineer Level
=====================================================
Monitors training for:
1. Loss progression (train + val)
2. Overfitting detection (train vs val gap)
3. Learning rate schedule
4. Gradient health
5. Convergence analysis
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
import time

def parse_log_file(log_path):
    """Parse training log and extract metrics."""
    metrics = defaultdict(list)

    with open(log_path) as f:
        lines = f.readlines()

    for line in lines:
        # Parse training steps
        step_match = re.search(r'Step (\d+) \| Loss: ([\d.]+) \(C0: ([\d.]+), Dec: ([\d.]+)\) \| LR: ([\d.e+-]+) \| Grad: ([\d.]+)', line)
        if step_match:
            step = int(step_match.group(1))
            metrics['step'].append(step)
            metrics['loss'].append(float(step_match.group(2)))
            metrics['c0_loss'].append(float(step_match.group(3)))
            metrics['dec_loss'].append(float(step_match.group(4)))
            metrics['lr'].append(float(step_match.group(5)))
            metrics['grad'].append(float(step_match.group(6)))

        # Parse validation
        val_match = re.search(r'Val Loss: ([\d.]+) \(C0: ([\d.]+), Dec: ([\d.]+)\)', line)
        if val_match:
            metrics['val_loss'].append(float(val_match.group(1)))
            metrics['val_c0'].append(float(val_match.group(2)))
            metrics['val_dec'].append(float(val_match.group(3)))

    return metrics

def analyze_metrics(metrics):
    """Analyze training health."""
    print("=" * 70)
    print("TRAINING HEALTH ANALYSIS - Senior AI Engineer Assessment")
    print("=" * 70)

    if not metrics['step']:
        print("No training steps found yet.")
        return

    # Current progress
    current_step = metrics['step'][-1]
    print(f"\n📊 CURRENT PROGRESS: Step {current_step}")
    print(f"   Total steps planned: 12,900")
    print(f"   Progress: {current_step/12900*100:.1f}%")

    # Loss analysis
    print(f"\n📉 LOSS ANALYSIS:")
    print(f"   Latest total loss: {metrics['loss'][-1]:.4f}")
    print(f"   Latest C0 loss: {metrics['c0_loss'][-1]:.4f}")
    print(f"   Latest decoder loss: {metrics['dec_loss'][-1]:.4f}")

    if len(metrics['loss']) > 1:
        first_loss = metrics['loss'][0]
        last_loss = metrics['loss'][-1]
        loss_change = (last_loss - first_loss) / first_loss * 100
        print(f"   Loss change: {loss_change:+.2f}% {'✅ IMPROVING' if loss_change < 0 else '⚠️ WORSENING'}")

    # Trend analysis
    if len(metrics['loss']) >= 5:
        recent = metrics['loss'][-5:]
        trend = recent[-1] - recent[0]
        print(f"   Recent trend (5 steps): {trend:+.4f} {'✅ DECREASING' if trend < 0 else '⚠️ INCREASING'}")

    # C0 loss trend (critical for voice quality)
    if len(metrics['c0_loss']) >= 3:
        recent_c0 = metrics['c0_loss'][-3:]
        c0_trend = recent_c0[-1] - recent_c0[0]
        print(f"\n🎯 CODEBOOK 0 (Critical for voice):")
        print(f"   Recent C0 trend: {c0_trend:+.4f} {'✅ GOOD' if c0_trend < 0 else '⚠️ WATCH'}")

    # Decoder loss trend
    if len(metrics['dec_loss']) >= 3:
        recent_dec = metrics['dec_loss'][-3:]
        dec_trend = recent_dec[-1] - recent_dec[0]
        print(f"\n🔧 DECODER (Audio quality):")
        print(f"   Recent decoder trend: {dec_trend:+.4f} {'✅ GOOD' if dec_trend < 0 else '⚠️ WATCH'}")

    # Learning rate
    print(f"\n📈 LEARNING RATE:")
    print(f"   Current LR: {metrics['lr'][-1]:.2e}")
    warmup_complete = current_step >= 645
    print(f"   Warmup: {'✅ Complete' if warmup_complete else f'{current_step}/645 steps'}")

    # Gradient health
    if metrics['grad']:
        avg_grad = sum(metrics['grad']) / len(metrics['grad'])
        max_grad = max(metrics['grad'])
        print(f"\n🌊 GRADIENT HEALTH:")
        print(f"   Average gradient norm: {avg_grad:.2f}")
        print(f"   Max gradient norm: {max_grad:.2f}")
        if max_grad > 50:
            print(f"   ⚠️ WARNING: High gradients detected!")

    # Overfitting check
    if metrics['val_loss']:
        latest_train = metrics['loss'][-1]
        latest_val = metrics['val_loss'][-1]
        gap = latest_val - latest_train
        print(f"\n🔍 OVERFITTING CHECK:")
        print(f"   Train loss: {latest_train:.4f}")
        print(f"   Val loss: {latest_val:.4f}")
        print(f"   Gap: {gap:+.4f} {'✅ OK' if gap < 0.5 else '⚠️ POTENTIAL OVERFITTING'}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if len(metrics['loss']) < 10:
        print("   - Still in early training, continue monitoring")
    elif metrics['loss'][-1] > 6.0:
        print("   - Loss still high, model is learning, continue training")
    elif metrics['loss'][-1] < 4.0:
        print("   - Loss is good, watch for overfitting")

    if len(metrics['c0_loss']) >= 3:
        c0_increase = metrics['c0_loss'][-1] > metrics['c0_loss'][0]
        if c0_increase:
            print("   - C0 loss increasing: This is normal during warmup phase")
            print("   - Will stabilize after warmup (step 645)")

    print("\n" + "=" * 70)

def main():
    log_path = Path("/home/ec2-user/SageMaker/project_maya/training/logs/ultimate_training.log")

    if not log_path.exists():
        print("Training log not found!")
        return

    print(f"Monitoring: {log_path}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    metrics = parse_log_file(log_path)
    analyze_metrics(metrics)

if __name__ == "__main__":
    main()
