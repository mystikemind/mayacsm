#!/bin/bash
# RESUME TRAINING AFTER INTERRUPTION
# ===================================
# Run this script if training was interrupted (power/network outage)
# It will automatically find the latest checkpoint and continue training

cd /home/ec2-user/SageMaker/project_maya
source venv/bin/activate

# Find the latest epoch checkpoint
LATEST_EPOCH=$(ls -d training/checkpoints/csm_maya_ultimate/epoch-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$LATEST_EPOCH" ]; then
    echo "ERROR: No epoch checkpoints found!"
    exit 1
fi

echo "=================================================="
echo "RESUMING TRAINING FROM: $LATEST_EPOCH"
echo "=================================================="

# Resume with 4 GPUs
nohup torchrun --nproc_per_node=4 \
  training/scripts/04_train_csm_ultimate.py \
  --data training/data/csm_ready_ex04 \
  --epochs 100 \
  --lr 3e-5 \
  --output training/checkpoints/csm_maya_ultimate \
  --resume "$LATEST_EPOCH" \
  >> training/logs/ultimate_training.log 2>&1 &

echo "Training resumed with PID: $!"
echo "Monitor with: tail -f training/logs/ultimate_training.log"
