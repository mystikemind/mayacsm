#!/usr/bin/env python3
"""Analyze if severe clicks are crossfade issues or natural speech."""

import numpy as np
import wave
from pathlib import Path

def analyze(filepath: str):
    with wave.open(filepath, 'r') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        sr = wav.getframerate()

    print(f"\nAnalyzing: {Path(filepath).name}")
    print(f"Duration: {len(audio)/sr:.2f}s")

    # Find severe clicks (>0.5)
    diff = np.abs(np.diff(audio))
    severe = np.where(diff > 0.5)[0]

    print(f"Severe clicks (>0.5): {len(severe)}")

    # Expected chunk boundaries (first chunk 3600, rest 15120)
    boundaries = [3600]
    pos = 3600
    while pos < len(audio):
        pos += 15120
        boundaries.append(pos)

    for pos in severe:
        time_ms = pos / sr * 1000

        # Check if near boundary
        near = "NOT at boundary"
        for b in boundaries:
            if abs(pos - b) < 100:
                near = f"NEAR boundary {b}"
                break

        print(f"  Click at {pos} ({time_ms:.1f}ms): diff={diff[pos]:.4f} - {near}")

        # Show context
        before = audio[max(0, pos-5):pos+1]
        after = audio[pos:min(len(audio), pos+6)]
        print(f"    Before: {before}")
        print(f"    After: {after}")

audio_dir = Path("/home/ec2-user/SageMaker/project_maya/audio_final_test")
analyze(str(audio_dir / "10_longer.wav"))
