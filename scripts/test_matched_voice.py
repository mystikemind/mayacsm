#!/usr/bin/env python3
"""
Test fine-tuned model with MATCHED voice prompt.
Compare: mismatched (Sesame) vs matched (ex04 training data)
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.io.wavfile as wav

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
OUTPUT_DIR = PROJECT_ROOT / "audio_matched_test"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_PHRASES = [
    ("greeting", "hi there how are you doing today"),
    ("happy", "oh wow thats amazing i love it"),
    ("sad", "oh no im so sorry to hear that"),
    ("thinking", "hmm let me think about that"),
    ("question", "wait what do you mean by that"),
    ("natural", "you know i was just thinking the same thing"),
    ("long", "thats a great point actually and i think youre absolutely right about that"),
]

def calculate_metrics(audio_np):
    diff = np.abs(np.diff(audio_np))
    return {
        "duration": len(audio_np) / 24000,
        "clicks": int(np.sum(diff > 0.3)),
        "harsh_clicks": int(np.sum(diff > 0.5)),
        "rms": float(np.sqrt(np.mean(audio_np ** 2))),
    }

def test_with_voice_prompt(voice_prompt_path, name, output_subdir):
    """Test TTS with specific voice prompt."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Voice prompt: {voice_prompt_path}")
    print("=" * 60)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Temporarily override voice prompt
    original = RealStreamingTTSEngine.VOICE_PROMPT_PATH
    RealStreamingTTSEngine.VOICE_PROMPT_PATH = voice_prompt_path

    tts = RealStreamingTTSEngine()
    tts.initialize()

    output_subdir.mkdir(exist_ok=True)
    results = []

    for phrase_name, phrase_text in TEST_PHRASES:
        print(f"\n  '{phrase_text[:40]}...'")

        try:
            start = time.time()
            first_chunk_time = None
            chunks = []

            for chunk in tts.generate_stream(phrase_text, use_context=False):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start
                chunks.append(chunk)

            if chunks:
                audio = torch.cat(chunks)
                audio_np = audio.cpu().numpy()
                audio_np = audio_np / max(abs(audio_np.min()), abs(audio_np.max())) * 0.9

                metrics = calculate_metrics(audio_np)
                metrics["first_chunk_ms"] = first_chunk_time * 1000
                metrics["phrase"] = phrase_name

                wav.write(str(output_subdir / f"{phrase_name}.wav"), 24000,
                         (audio_np * 32767).astype(np.int16))

                print(f"    {metrics['duration']:.1f}s | {metrics['clicks']} clicks | First: {first_chunk_time*1000:.0f}ms")
                results.append(metrics)

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"phrase": phrase_name, "error": str(e)})

        torch.cuda.empty_cache()

    # Restore
    RealStreamingTTSEngine.VOICE_PROMPT_PATH = original
    del tts
    torch.cuda.empty_cache()

    return results

def main():
    print("=" * 70)
    print("MATCHED vs MISMATCHED VOICE PROMPT COMPARISON")
    print("=" * 70)

    # Test 1: Matched (training data voice)
    matched_results = test_with_voice_prompt(
        str(PROJECT_ROOT / "assets/voice_prompt/maya_voice_prompt_matched.pt"),
        "MATCHED (ex04 training data)",
        OUTPUT_DIR / "matched"
    )

    # Test 2: Mismatched (Sesame demo voice) - if exists
    sesame_path = PROJECT_ROOT / "assets/voice_prompt/maya_voice_prompt_sesame.pt"
    if sesame_path.exists():
        mismatched_results = test_with_voice_prompt(
            str(sesame_path),
            "MISMATCHED (Sesame demo voice)",
            OUTPUT_DIR / "mismatched"
        )
    else:
        mismatched_results = []

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for name, results in [("Matched (ex04)", matched_results), ("Mismatched (Sesame)", mismatched_results)]:
        valid = [r for r in results if "error" not in r]
        if valid:
            avg_clicks = np.mean([r["clicks"] for r in valid])
            avg_first = np.mean([r["first_chunk_ms"] for r in valid])
            avg_dur = np.mean([r["duration"] for r in valid])
            print(f"\n{name}:")
            print(f"  Avg clicks: {avg_clicks:.0f}")
            print(f"  Avg first chunk: {avg_first:.0f}ms")
            print(f"  Avg duration: {avg_dur:.1f}s")
            print(f"  Success: {len(valid)}/{len(results)}")

    print(f"\n\nAudio samples: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
