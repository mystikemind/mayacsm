#!/bin/bash
# =============================================================
# Maya SOTA Training Pipeline
# =============================================================
# This script orchestrates the full training pipeline:
# 1. Wait for Phase 1 (decoder-only ex04) to complete
# 2. Evaluate Phase 1 model
# 3. Run Phase 2 (combined naturalness training)
# 4. Evaluate Phase 2 model
# =============================================================

set -e

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/home/ec2-user/SageMaker/.cache/huggingface/datasets

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/home/ec2-user/SageMaker/project_maya"
CHECKPOINT_DIR="$PROJECT_DIR/training/checkpoints"

cd "$PROJECT_DIR"

echo "=============================================="
echo "Maya SOTA Training Pipeline"
echo "=============================================="
echo "$(date): Starting pipeline"
echo ""

# Step 1: Check if Phase 1 is still running
echo "[Step 1] Checking Phase 1 (decoder-only ex04) status..."
PHASE1_PID=$(pgrep -f "05_train_decoder_only" 2>/dev/null || true)

if [ -n "$PHASE1_PID" ]; then
    echo "  Phase 1 is still running (PID: $PHASE1_PID)"
    echo "  Waiting for completion..."
    while kill -0 "$PHASE1_PID" 2>/dev/null; do
        sleep 60
        echo "  $(date): Still waiting..."
    done
    echo "  Phase 1 completed!"
else
    echo "  Phase 1 is not running."
fi

# Step 2: Evaluate Phase 1 model
echo ""
echo "[Step 2] Evaluating Phase 1 model..."
if [ -d "$CHECKPOINT_DIR/csm_maya_decoder_only/best_model" ]; then
    python3 "$SCRIPT_DIR/08_evaluate_model_quality.py" \
        --checkpoint "$CHECKPOINT_DIR/csm_maya_decoder_only/best_model" \
        --device cuda:0
    echo "  Phase 1 evaluation complete!"
else
    echo "  WARNING: No Phase 1 checkpoint found, skipping evaluation"
fi

# Step 3: Run Phase 2 (combined naturalness)
echo ""
echo "[Step 3] Starting Phase 2 (combined naturalness training)..."
echo "  Using all 24K+ samples from 4 datasets"
echo "  CoVoC codebook weighting + ERVQ monitoring"
echo ""

python3 "$SCRIPT_DIR/07_train_combined_naturalness.py" \
    --gpu 0 \
    --epochs 10 \
    --lr 3e-5 \
    --grad-accum 8 \
    --frames-per-sample 16 \
    --warmup-steps 300 \
    --save-steps 500 \
    --log-steps 10

echo ""
echo "  Phase 2 training complete!"

# Step 4: Evaluate Phase 2 model
echo ""
echo "[Step 4] Evaluating Phase 2 model..."
if [ -d "$CHECKPOINT_DIR/csm_maya_combined_naturalness/best_model" ]; then
    python3 "$SCRIPT_DIR/08_evaluate_model_quality.py" \
        --checkpoint "$CHECKPOINT_DIR/csm_maya_combined_naturalness/best_model" \
        --device cuda:0
    echo "  Phase 2 evaluation complete!"
fi

echo ""
echo "=============================================="
echo "Pipeline complete! $(date)"
echo "=============================================="
echo ""
echo "Check evaluation results in:"
echo "  $PROJECT_DIR/training/eval_samples/"
echo ""
echo "Best models saved in:"
echo "  Phase 1: $CHECKPOINT_DIR/csm_maya_decoder_only/best_model/"
echo "  Phase 2: $CHECKPOINT_DIR/csm_maya_combined_naturalness/best_model/"
