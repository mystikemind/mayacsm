#!/usr/bin/env python3
"""
Investigate the specific samples that have audio issues.
"""

import numpy as np
import wave
from pathlib import Path

def analyze_problem_sample(filepath: str):
    """Deep analysis of a problematic audio file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {filepath}")
    print('='*70)

    with wave.open(filepath, 'r') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        sr = wav.getframerate()

    print(f"Duration: {len(audio)/sr:.2f}s, Samples: {len(audio)}")

    # Find all severe discontinuities
    diff = np.abs(np.diff(audio))
    severe_positions = np.where(diff > 0.5)[0]

    print(f"\nSevere discontinuities (>0.5): {len(severe_positions)}")

    # Analyze each
    chunk_boundaries = []
    # Assuming first chunk is 3600, rest are 15120
    pos = 3600
    while pos < len(audio):
        chunk_boundaries.append(pos)
        pos += 15120

    print(f"Expected chunk boundaries: {chunk_boundaries[:10]}...")

    for pos in severe_positions[:20]:  # First 20
        time_ms = pos / sr * 1000

        # Is it near a chunk boundary?
        near_boundary = False
        boundary_dist = float('inf')
        for b in chunk_boundaries:
            d = abs(pos - b)
            if d < 100:  # Within 100 samples
                near_boundary = True
                boundary_dist = min(boundary_dist, d)

        before_10 = audio[max(0, pos-10):pos+1]
        after_10 = audio[pos:min(len(audio), pos+11)]

        print(f"\n  Position {pos} ({time_ms:.1f}ms):")
        print(f"    Diff: {diff[pos]:.4f}")
        print(f"    Before: {audio[pos-1]:.4f} -> After: {audio[pos]:.4f}")
        if near_boundary:
            print(f"    ⚠ Near chunk boundary! Distance: {boundary_dist} samples")
        else:
            print(f"    Not near boundary (likely speech transient)")

        # Check if it looks like a natural speech transient
        energy_before = np.sqrt(np.mean(before_10**2))
        energy_after = np.sqrt(np.mean(after_10**2))
        print(f"    Energy before: {energy_before:.4f}, after: {energy_after:.4f}")

    # Visualize where the severe discontinuities are
    print(f"\n--- Distribution of severe clicks ---")
    if len(severe_positions) > 0:
        # Divide audio into 10 segments
        segment_len = len(audio) // 10
        for i in range(10):
            start = i * segment_len
            end = (i + 1) * segment_len
            count = np.sum((severe_positions >= start) & (severe_positions < end))
            bar = '*' * min(count, 50)
            print(f"  Segment {i} ({start/sr*1000:.0f}-{end/sr*1000:.0f}ms): {count:3d} {bar}")

def main():
    audio_dir = Path("/home/ec2-user/SageMaker/project_maya/audio_final_test")

    # Analyze the problematic samples
    problem_files = [
        "11_story.wav",
        "15_gentle.wav"
    ]

    for f in problem_files:
        path = audio_dir / f
        if path.exists():
            analyze_problem_sample(str(path))

if __name__ == "__main__":
    main()
