#!/bin/bash
# POST-TRAINING PIPELINE
# ======================
# Run this after training completes (or early stops).
# It will evaluate the best model, generate samples, and prepare for integration.
#
# Usage: bash training/scripts/post_training_pipeline.sh
# Or let it auto-wait: bash training/scripts/post_training_pipeline.sh --wait

set -e

PROJECT_ROOT="/home/ec2-user/SageMaker/project_maya"
CHECKPOINT_DIR="$PROJECT_ROOT/training/checkpoints/csm_maya_ultimate"
BEST_MODEL="$CHECKPOINT_DIR/best_model"
EVAL_OUTPUT="$PROJECT_ROOT/training/evaluation"
VOICE_PROMPT_DIR="$PROJECT_ROOT/assets/voice_prompt"
LOG_FILE="$PROJECT_ROOT/training/logs/post_training.log"

cd "$PROJECT_ROOT"
source venv/bin/activate

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_FILE"
}

log "============================================================"
log "POST-TRAINING PIPELINE STARTED"
log "============================================================"

# Step 0: Wait for training if --wait flag
if [ "$1" = "--wait" ]; then
    log "Waiting for training to finish..."
    while pgrep -f "04_train_csm_ultimate" > /dev/null 2>&1; do
        sleep 300  # Check every 5 minutes
        LAST_LINE=$(tail -1 "$PROJECT_ROOT/training/logs/ultimate_training.log" 2>/dev/null)
        log "Training still running: $LAST_LINE"
    done
    log "Training process has finished!"
    sleep 10  # Let filesystem settle
fi

# Step 1: Validate best model exists
log ""
log "STEP 1: Validating best model checkpoint"
log "============================================================"

if [ ! -f "$BEST_MODEL/model.pt" ]; then
    log "ERROR: Best model not found at $BEST_MODEL/model.pt"
    exit 1
fi

MODEL_SIZE=$(du -sh "$BEST_MODEL/model.pt" | cut -f1)
log "Best model found: $MODEL_SIZE"

# Show training state
if [ -f "$BEST_MODEL/training_state.json" ]; then
    log "Training state:"
    python3 -c "
import json
with open('$BEST_MODEL/training_state.json') as f:
    state = json.load(f)
print(f'  Global step: {state[\"global_step\"]}')
print(f'  Epoch: {state[\"epoch\"]}')
print(f'  Best val loss: {state[\"best_val_loss\"]:.4f}')
print(f'  Patience counter: {state[\"patience_counter\"]}')
" 2>&1 | tee -a "$LOG_FILE"
fi

# Step 2: Clean up old checkpoints to free GPU memory
log ""
log "STEP 2: Cleaning up old epoch checkpoints"
log "============================================================"

# Keep only last 3 epochs and best_model
EPOCH_DIRS=$(ls -d "$CHECKPOINT_DIR"/epoch-* 2>/dev/null | sort -t- -k2 -n)
NUM_EPOCHS=$(echo "$EPOCH_DIRS" | wc -l)

if [ "$NUM_EPOCHS" -gt 3 ]; then
    TO_REMOVE=$(echo "$EPOCH_DIRS" | head -n -3)
    for dir in $TO_REMOVE; do
        log "  Removing $(basename $dir)"
        rm -rf "$dir"
    done
    log "  Kept last 3 epoch checkpoints"
fi

DISK_AVAIL=$(df -h /home/ec2-user/SageMaker/ | tail -1 | awk '{print $4}')
log "Disk available: $DISK_AVAIL"

# Step 3: Run full evaluation
log ""
log "STEP 3: Running model evaluation"
log "============================================================"

REFERENCE_AUDIO="$VOICE_PROMPT_DIR/maya_voice_prompt.wav"
if [ ! -f "$REFERENCE_AUDIO" ]; then
    REFERENCE_AUDIO=""
    log "No reference audio found, skipping speaker similarity"
fi

mkdir -p "$EVAL_OUTPUT"

if [ -n "$REFERENCE_AUDIO" ]; then
    python3 training/scripts/05_evaluate_model.py \
        --checkpoint "$BEST_MODEL" \
        --reference "$REFERENCE_AUDIO" \
        --output "$EVAL_OUTPUT" \
        2>&1 | tee -a "$LOG_FILE"
else
    python3 training/scripts/05_evaluate_model.py \
        --checkpoint "$BEST_MODEL" \
        --output "$EVAL_OUTPUT" \
        2>&1 | tee -a "$LOG_FILE"
fi

# Step 4: Check results
log ""
log "STEP 4: Evaluation Results"
log "============================================================"

if [ -f "$EVAL_OUTPUT/metrics.json" ]; then
    python3 -c "
import json
with open('$EVAL_OUTPUT/metrics.json') as f:
    m = json.load(f)
print(f'UTMOS Score:        {m[\"utmos_score\"]:.2f} (target: > 4.0)')
print(f'Speaker Similarity: {m[\"speaker_similarity\"]:.2f} (target: > 0.85)')
print(f'CER:                {m[\"cer\"]:.2%} (target: < 5%)')
print(f'WER:                {m[\"wer\"]:.2%} (target: < 10%)')
print(f'Samples Generated:  {m[\"samples_generated\"]}')
print(f'Passes Targets:     {m[\"passes_targets\"]}')
" 2>&1 | tee -a "$LOG_FILE"
fi

if [ -f "$EVAL_OUTPUT/report.md" ]; then
    log "Full report: $EVAL_OUTPUT/report.md"
fi

# Step 5: Generate new voice prompt with fine-tuned model
log ""
log "STEP 5: Generating voice prompt with fine-tuned model"
log "============================================================"

if [ -f "$PROJECT_ROOT/training/scripts/06_integrate_model.py" ]; then
    python3 training/scripts/06_integrate_model.py \
        --checkpoint "$BEST_MODEL" \
        2>&1 | tee -a "$LOG_FILE"
else
    log "Integration script not found, skipping voice prompt generation"
fi

# Step 6: Summary
log ""
log "============================================================"
log "POST-TRAINING PIPELINE COMPLETE"
log "============================================================"
log ""
log "Results:"
log "  Evaluation: $EVAL_OUTPUT/"
log "  Samples:    $EVAL_OUTPUT/samples/"
log "  Report:     $EVAL_OUTPUT/report.md"
log "  Metrics:    $EVAL_OUTPUT/metrics.json"
log ""
log "Next steps:"
log "  1. Listen to audio samples in $EVAL_OUTPUT/samples/"
log "  2. Review report at $EVAL_OUTPUT/report.md"
log "  3. If quality is good, run: python run.py"
log "============================================================"
