#!/usr/bin/env python3
"""Generate pre-cached greeting for instant playback."""

import sys
import os

# Disable torch.compile for stability
os.environ["NO_TORCH_COMPILE"] = "1"

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import torchaudio
from pathlib import Path

def main():
    print("=" * 60)
    print("GENERATING PRE-CACHED GREETING")
    print("=" * 60)

    output_path = Path("/home/ec2-user/SageMaker/project_maya/assets/fillers/greeting.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load CSM
    print("\nLoading CSM-1B...")
    from generator import load_csm_1b
    generator = load_csm_1b(device="cuda")
    print(f"CSM loaded. Sample rate: {generator.sample_rate}")

    # Generate greeting
    greeting_text = "Hi! I'm Maya. How can I help you today?"
    print(f"\nGenerating: '{greeting_text}'")

    audio = generator.generate(
        text=greeting_text,
        speaker=0,
        context=[],
        max_audio_length_ms=5000,
    )

    # Normalize
    if audio.abs().max() > 0:
        audio = audio / audio.abs().max() * 0.95

    # Save
    torchaudio.save(str(output_path), audio.unsqueeze(0).cpu(), generator.sample_rate)

    duration = len(audio) / generator.sample_rate
    print(f"\nSaved: {output_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {generator.sample_rate}")

    print("\n" + "=" * 60)
    print("GREETING GENERATED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
