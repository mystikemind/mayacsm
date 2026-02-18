#!/usr/bin/env python3
"""Analyze audio files for quality issues like clicks, screeching, etc."""

import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

def analyze_audio(filepath):
    """Analyze audio file for quality issues."""
    sr, audio = wav.read(filepath)

    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32767.0

    results = {
        "file": filepath.name,
        "duration": len(audio) / sr,
        "peak": float(np.abs(audio).max()),
        "rms": float(np.sqrt(np.mean(audio ** 2))),
    }

    # Detect clicks/pops (sudden amplitude changes)
    diff = np.abs(np.diff(audio))
    results["clicks"] = int(np.sum(diff > 0.3))
    results["severe_clicks"] = int(np.sum(diff > 0.5))
    results["max_diff"] = float(diff.max())

    # Detect high-frequency content (potential screeching)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
    zcr = zero_crossings / len(audio) * sr
    results["zero_crossing_rate"] = float(zcr)

    # Detect clipping
    results["clipping_samples"] = int(np.sum(np.abs(audio) > 0.99))

    return results


def main():
    output_dir = Path("/home/ec2-user/SageMaker/project_maya/audio_comparison_test")

    for subdir in ["official_generator", "pipeline_tts"]:
        print(f"\n{'=' * 60}")
        print(f"{subdir.upper()}")
        print("=" * 60)

        dir_path = output_dir / subdir
        if not dir_path.exists():
            continue

        total_clicks = 0
        total_severe = 0

        for wav_file in sorted(dir_path.glob("*.wav")):
            results = analyze_audio(wav_file)

            status = "OK"
            if results["severe_clicks"] > 0:
                status = "SEVERE CLICKS!"
            elif results["clicks"] > 5:
                status = "CLICKS"

            print(f"\n  {results['file']}:")
            print(f"    Duration: {results['duration']:.1f}s, Clicks: {results['clicks']} (severe: {results['severe_clicks']})")
            print(f"    Max diff: {results['max_diff']:.3f}, ZCR: {results['zero_crossing_rate']:.0f}/s")
            print(f"    Status: {status}")

            total_clicks += results["clicks"]
            total_severe += results["severe_clicks"]

        print(f"\n  TOTAL: {total_clicks} clicks, {total_severe} severe")

    # Check user-reported problem file
    problem_file = Path("/home/ec2-user/SageMaker/project_maya/audio_pipeline_test/pipeline_2_oh_wow_thats_amazing.wav")
    if problem_file.exists():
        print(f"\n{'=' * 60}")
        print("USER-REPORTED PROBLEM FILE")
        print("=" * 60)
        results = analyze_audio(problem_file)
        print(f"\n  {results['file']}:")
        print(f"    Duration: {results['duration']:.1f}s, Clicks: {results['clicks']} (severe: {results['severe_clicks']})")
        print(f"    Max diff: {results['max_diff']:.3f}, ZCR: {results['zero_crossing_rate']:.0f}/s")


if __name__ == "__main__":
    main()
