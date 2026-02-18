#!/bin/bash
#
# MAYA QUALITY TRAINING PIPELINE
#
# This script runs the complete fine-tuning pipeline to achieve
# Sesame Maya-level voice quality.
#
# Usage:
#   ./run_pipeline.sh              # Run full pipeline
#   ./run_pipeline.sh --speaker elisabeth  # Use different speaker
#   ./run_pipeline.sh --resume     # Resume from last step
#
# Steps:
#   1. Download Expresso dataset
#   2. Extract single speaker
#   3. Preprocess audio for CSM
#   4. Fine-tune CSM-1B
#   5. Evaluate model
#   6. Integrate into Maya
#

set -e  # Exit on error

# Configuration
SPEAKER="${SPEAKER:-talia}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"

# Paths
PROJECT_ROOT="/home/ec2-user/SageMaker/project_maya"
SCRIPTS_DIR="$PROJECT_ROOT/training/scripts"
DATA_DIR="$PROJECT_ROOT/training/data"
CHECKPOINT_DIR="$PROJECT_ROOT/training/checkpoints"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check CUDA
check_cuda() {
    log_info "Checking CUDA availability..."
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
}

# Step 1: Download Expresso
step1_download() {
    log_info "=========================================="
    log_info "STEP 1: Download Expresso Dataset"
    log_info "=========================================="

    cd "$PROJECT_ROOT"
    python3 "$SCRIPTS_DIR/01_download_expresso.py"
}

# Step 2: Extract single speaker
step2_extract() {
    log_info "=========================================="
    log_info "STEP 2: Extract Single Speaker ($SPEAKER)"
    log_info "=========================================="

    cd "$PROJECT_ROOT"
    python3 "$SCRIPTS_DIR/02_extract_single_speaker.py" --speaker "$SPEAKER"
}

# Step 3: Preprocess audio
step3_preprocess() {
    log_info "=========================================="
    log_info "STEP 3: Preprocess Audio for CSM"
    log_info "=========================================="

    cd "$PROJECT_ROOT"
    python3 "$SCRIPTS_DIR/03_preprocess_audio.py" \
        --input "$DATA_DIR/expresso_$SPEAKER" \
        --workers 4
}

# Step 4: Fine-tune CSM
step4_train() {
    log_info "=========================================="
    log_info "STEP 4: Fine-tune CSM-1B"
    log_info "=========================================="
    log_info "Epochs: $EPOCHS"
    log_info "Batch size: $BATCH_SIZE"
    log_info "Learning rate: $LEARNING_RATE"
    log_info "=========================================="

    cd "$PROJECT_ROOT"
    python3 "$SCRIPTS_DIR/04_train_csm.py" \
        --data "$DATA_DIR/csm_ready_$SPEAKER" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --output "$CHECKPOINT_DIR/csm_maya"
}

# Step 5: Evaluate model
step5_evaluate() {
    log_info "=========================================="
    log_info "STEP 5: Evaluate Fine-tuned Model"
    log_info "=========================================="

    cd "$PROJECT_ROOT"
    python3 "$SCRIPTS_DIR/05_evaluate_model.py" \
        --checkpoint "$CHECKPOINT_DIR/csm_maya/best_model"
}

# Step 6: Integrate model
step6_integrate() {
    log_info "=========================================="
    log_info "STEP 6: Integrate into Maya Pipeline"
    log_info "=========================================="

    cd "$PROJECT_ROOT"
    python3 "$SCRIPTS_DIR/06_integrate_model.py" \
        --checkpoint "$CHECKPOINT_DIR/csm_maya/best_model"
}

# Main
main() {
    echo ""
    echo "========================================"
    echo "  MAYA QUALITY TRAINING PIPELINE"
    echo "  Target: Sesame Maya-level Quality"
    echo "========================================"
    echo ""

    # Check CUDA
    check_cuda
    echo ""

    # Parse arguments
    START_STEP=1
    while [[ $# -gt 0 ]]; do
        case $1 in
            --speaker)
                SPEAKER="$2"
                shift 2
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --resume)
                # Find last completed step
                if [ -f "$CHECKPOINT_DIR/csm_maya/best_model/model.pt" ]; then
                    START_STEP=5
                elif [ -d "$DATA_DIR/csm_ready_$SPEAKER" ]; then
                    START_STEP=4
                elif [ -d "$DATA_DIR/expresso_$SPEAKER" ]; then
                    START_STEP=3
                elif [ -d "$DATA_DIR/expresso" ]; then
                    START_STEP=2
                fi
                log_info "Resuming from step $START_STEP"
                shift
                ;;
            --step)
                START_STEP="$2"
                shift 2
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    log_info "Speaker: $SPEAKER"
    log_info "Starting from step: $START_STEP"
    echo ""

    # Run steps
    if [ $START_STEP -le 1 ]; then step1_download; fi
    if [ $START_STEP -le 2 ]; then step2_extract; fi
    if [ $START_STEP -le 3 ]; then step3_preprocess; fi
    if [ $START_STEP -le 4 ]; then step4_train; fi
    if [ $START_STEP -le 5 ]; then step5_evaluate; fi
    if [ $START_STEP -le 6 ]; then step6_integrate; fi

    echo ""
    log_info "========================================"
    log_info "  PIPELINE COMPLETE!"
    log_info "========================================"
    echo ""
    log_info "To test the fine-tuned model:"
    log_info "  ./start_maya.sh"
    log_info "  python run.py"
    echo ""
}

main "$@"
