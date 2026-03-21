#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# HuggingFace auth (needed for sesame/csm-1b and llama tokenizer downloads)
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Add CSM library to PYTHONPATH so 'from models import Model' works
export PYTHONPATH="${MAYA_CSM_ROOT:-$(pwd)/csm}:${PYTHONPATH}"

# cuDNN path (common locations on Vast.ai)
for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
    [ -d "$dir" ] && export LD_LIBRARY_PATH="$dir:${LD_LIBRARY_PATH}"
done

echo "Starting Maya CSM-1B TTS server on port ${PORT:-8002}…"
python server.py
