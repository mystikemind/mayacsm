#!/usr/bin/env python3
"""
Create voice prompt from TRAINING DATA for perfect voice consistency.

Key insight: The voice prompt should match the fine-tuned voice exactly.
We trained on Expresso ex04, so we use ex04 samples as voice prompt.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import numpy as np

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
TRAINING_DATA = PROJECT_ROOT / "training/data/csm_ready_ex04"
OUTPUT_PATH = PROJECT_ROOT / "assets/voice_prompt/maya_voice_prompt_matched.pt"

def main():
    print("=" * 60)
    print("CREATING MATCHED VOICE PROMPT")
    print("Using Expresso ex04 samples (same as training data)")
    print("=" * 60)

    # Load training manifest
    train_json = TRAINING_DATA / "train.json"
    with open(train_json) as f:
        samples = json.load(f)

    # Select diverse samples for voice prompt (target: 30-60 seconds)
    # Mix of styles for natural variation
    selected = []
    target_duration = 45  # seconds
    current_duration = 0

    # Prioritize neutral/default style for cleaner voice identity
    neutral = [s for s in samples if s.get("style") == "default"]
    other = [s for s in samples if s.get("style") != "default"]

    # Take mostly neutral, some expressive
    candidates = neutral[:20] + other[:10]

    for sample in candidates:
        if current_duration >= target_duration:
            break

        audio_path = TRAINING_DATA / sample["path"]
        if not audio_path.exists():
            continue

        duration = sample.get("duration", 3.0)
        if duration < 1.5 or duration > 8.0:  # Skip too short/long
            continue

        selected.append(sample)
        current_duration += duration

    print(f"\nSelected {len(selected)} samples ({current_duration:.1f}s total)")

    # Concatenate audio
    all_audio = []
    all_text = []

    for sample in selected:
        audio_path = TRAINING_DATA / sample["path"]

        # Load audio
        audio, sr = torchaudio.load(str(audio_path))
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, 24000)
        audio = audio.mean(dim=0) if audio.dim() > 1 and audio.size(0) > 1 else audio.squeeze(0)

        all_audio.append(audio)

        # Clean text
        text = sample["text"]
        if text.startswith("["):
            text = text.split("]", 1)[-1].strip()
        all_text.append(text)

        print(f"  + {sample['path']}: '{text[:40]}...' ({len(audio)/24000:.1f}s)")

    # Concatenate with small silence gaps
    silence = torch.zeros(int(24000 * 0.3))  # 300ms silence between clips
    combined_audio = []
    for i, audio in enumerate(all_audio):
        combined_audio.append(audio)
        if i < len(all_audio) - 1:
            combined_audio.append(silence)

    final_audio = torch.cat(combined_audio)
    final_text = " ".join(all_text)

    print(f"\nFinal voice prompt:")
    print(f"  Duration: {len(final_audio)/24000:.1f}s")
    print(f"  Text length: {len(final_text)} chars")

    # Normalize audio
    peak = final_audio.abs().max()
    if peak > 0:
        final_audio = final_audio / peak * 0.9

    # Save
    torch.save({
        "audio": final_audio,
        "text": final_text,
        "sample_rate": 24000,
        "source": "expresso_ex04_training_data",
        "num_samples": len(selected),
    }, OUTPUT_PATH)

    print(f"\nSaved to: {OUTPUT_PATH}")

    # Also save as WAV for listening
    wav_path = OUTPUT_PATH.with_suffix(".wav")
    import scipy.io.wavfile as wav
    wav.write(str(wav_path), 24000, (final_audio.numpy() * 32767).astype(np.int16))
    print(f"WAV preview: {wav_path}")

    print("\n" + "=" * 60)
    print("VOICE PROMPT CREATED")
    print("This prompt matches the fine-tuned model's voice exactly!")
    print("=" * 60)

if __name__ == "__main__":
    main()
