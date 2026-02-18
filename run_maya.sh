#!/bin/bash
#
# MAYA STARTUP SCRIPT - Local Engines (No Docker)
#
# This script sets up the correct environment and runs Maya
# with all local engines (STT, LLM, TTS).
#
# Usage:
#   ./run_maya.sh              # Start Maya server
#   ./run_maya.sh --port 8080  # Start on custom port
#

set -e

PROJECT_DIR="/home/ec2-user/SageMaker/project_maya"
cd "$PROJECT_DIR"

# Set cuDNN 9 library path for faster-whisper (CTranslate2)
# This is CRITICAL - without it, STT will crash
export LD_LIBRARY_PATH="/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Enable TF32 for faster matmuls
export NVIDIA_TF32_OVERRIDE=1

echo "============================================================"
echo "  MAYA - Sesame AI Level Voice Assistant"
echo "============================================================"
echo ""
echo "  Environment:"
echo "    cuDNN path: Set"
echo "    TF32: Enabled"
echo ""
echo "  Expected Latency: ~450ms first audio"
echo "    STT: ~10-20ms (faster-whisper)"
echo "    LLM: ~200-300ms (Llama 3.2 3B)"
echo "    TTS: ~100-200ms (CSM-1B streaming)"
echo ""
echo "============================================================"
echo ""

# Run Maya
exec python3 run.py "$@"
