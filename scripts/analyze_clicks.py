#!/usr/bin/env python3
"""
Analyze if detected "clicks" are real crossfade issues or natural speech transients.
"""

import numpy as np
import wave
from pathlib import Path

def analyze_wav(filepath: str):
    """Analyze a WAV file for click characteristics."""
    with wave.open(filepath, 'r') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        sr = wav.getframerate()

    print(f"\nAnalyzing: {filepath}")
    print(f"  Duration: {len(audio)/sr:.2f}s, Samples: {len(audio)}")

    # Find large sample differences
    diff = np.abs(np.diff(audio))
    threshold = 0.2  # Lower threshold to catch more

    large_diffs = np.where(diff > threshold)[0]

    if len(large_diffs) == 0:
        print("  ✓ No significant transients detected")
        return

    print(f"  Transients > {threshold}: {len(large_diffs)}")

    # Classify transients
    for pos in large_diffs[:15]:  # First 15
        before = audio[max(0, pos-10):pos+1]
        after = audio[pos:min(len(audio), pos+11)]

        # Check if it's a click (sudden spike) or gradual transition
        before_trend = np.diff(before)
        after_trend = np.diff(after)

        is_sudden_both = np.max(np.abs(before_trend)) > 0.15 and np.max(np.abs(after_trend)) > 0.15
        is_plosive = diff[pos] > 0.25 and np.mean(np.abs(audio[max(0,pos-50):pos])) < 0.1

        time_ms = pos / sr * 1000

        classification = ""
        if is_plosive:
            classification = "(plosive)"
        elif is_sudden_both:
            classification = "(speech transient)"
        else:
            classification = "(possible click)"

        print(f"    {time_ms:.1f}ms: diff={diff[pos]:.4f} {classification}")

def main():
    audio_dir = Path("/home/ec2-user/SageMaker/project_maya/audio_quality_test")

    files = [
        "short_hi.wav",
        "medium_question.wav",
        "long_response.wav",
        "with_pauses.wav",
        "real_tts_test.wav",
    ]

    print("="*70)
    print("CLICK ANALYSIS - Distinguishing Speech from Artifacts")
    print("="*70)

    for f in files:
        path = audio_dir / f
        if path.exists():
            analyze_wav(str(path))

    print("\n" + "="*70)
    print("Note: Most 'clicks' in natural speech are plosives (p,t,k,b,d,g)")
    print("True crossfade clicks would appear at regular chunk intervals")
    print("="*70)

if __name__ == "__main__":
    main()
