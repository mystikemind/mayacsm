#!/bin/bash
# Quick start training after instance restart/upgrade
# Just run: ./training/START_TRAINING.sh

set -e

cd /home/ec2-user/SageMaker/project_maya

echo "=============================================="
echo "  MAYA VOICE FINE-TUNING - QUICK START"
echo "=============================================="

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi

# Activate environment
source venv/bin/activate

# Clear any old checkpoints
rm -rf training/checkpoints/csm_maya/* 2>/dev/null || true
mkdir -p training/checkpoints/csm_maya

echo ""
echo "Starting CSM-1B fine-tuning..."
echo "This will take approximately 10-15 hours for 25 epochs"
echo ""

# Run training with logging
PYTHONUNBUFFERED=1 python training/scripts/04_train_csm_efficient.py \
    --data training/data/csm_ready_ex04 \
    --epochs 25 \
    --lr 3e-5 \
    --output training/checkpoints/csm_maya \
    2>&1 | tee training/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=============================================="
echo "  TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Evaluate: python training/scripts/05_evaluate_model.py --checkpoint training/checkpoints/csm_maya/best_model"
echo "  2. Integrate: python training/scripts/06_integrate_model.py --checkpoint training/checkpoints/csm_maya/best_model"
echo "  3. Test: python run.py"
