#!/usr/bin/env python3
"""
Quick Quality Comparison - Test existing TTS with different voice prompts.
Uses the working RealStreamingTTSEngine.
"""

import sys
import os
import time
import json
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
OUTPUT_DIR = PROJECT_ROOT / "audio_comparison_test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Test phrases - covering all problematic scenarios
TEST_PHRASES = [
    ("01_greeting", "hi there how are you doing today"),
    ("02_happy", "oh wow thats amazing i love it"),
    ("03_sad", "oh no im so sorry to hear that"),
    ("04_thinking", "hmm let me think about that"),
    ("05_question", "wait what do you mean by that"),
    ("06_agree", "yeah that makes a lot of sense"),
    ("07_natural", "you know i was just thinking the same thing"),
]

# Voice prompts to test
VOICE_PROMPTS = [
    ("human_10s", "assets/voice_prompt/maya_voice_prompt_human.pt"),
    ("sesame_158s", "assets/voice_prompt/maya_voice_prompt_sesame.pt"),
    ("expresso_32s", "assets/voice_prompt/maya_voice_prompt.pt"),
]

def calculate_metrics(audio_np):
    """Calculate audio quality metrics."""
    return {
        "duration": len(audio_np) / 24000,
        "peak": float(np.abs(audio_np).max()),
        "rms": float(np.sqrt(np.mean(audio_np ** 2))),
        "clicks": int(np.sum(np.abs(np.diff(audio_np)) > 0.3)),
        "harsh_clicks": int(np.sum(np.abs(np.diff(audio_np)) > 0.5)),
    }


def test_voice_prompt(vp_name, vp_path):
    """Test TTS with a specific voice prompt."""
    print(f"\n{'='*60}")
    print(f"Testing: {vp_name}")
    print(f"Path: {vp_path}")
    print("=" * 60)

    if not Path(vp_path).exists():
        print(f"  SKIPPED: File not found")
        return []

    # Check voice prompt duration
    try:
        vp_data = torch.load(vp_path, weights_only=False)
        if isinstance(vp_data, dict):
            vp_audio = vp_data.get("audio", torch.zeros(24000))
            vp_dur = len(vp_audio) / 24000
        else:
            vp_dur = len(vp_data.audio) / 24000 if hasattr(vp_data, 'audio') else 0
        print(f"Voice prompt duration: {vp_dur:.1f}s")
    except Exception as e:
        print(f"  Error loading voice prompt: {e}")
        return []

    # Import TTS engine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Temporarily override voice prompt path
    original_path = RealStreamingTTSEngine.VOICE_PROMPT_PATH
    RealStreamingTTSEngine.VOICE_PROMPT_PATH = str(PROJECT_ROOT / vp_path)

    try:
        tts = RealStreamingTTSEngine()
        tts.initialize()
    except Exception as e:
        print(f"  Error initializing TTS: {e}")
        RealStreamingTTSEngine.VOICE_PROMPT_PATH = original_path
        return []

    output_subdir = OUTPUT_DIR / vp_name
    output_subdir.mkdir(exist_ok=True)

    results = []

    for phrase_name, phrase_text in TEST_PHRASES:
        print(f"\n  Generating: '{phrase_text[:40]}...'")

        try:
            start = time.time()
            chunks = list(tts.generate_stream(phrase_text, use_context=False))
            gen_time = time.time() - start

            if chunks:
                audio = torch.cat(chunks)
                audio_np = audio.cpu().numpy()

                metrics = calculate_metrics(audio_np)
                metrics["generation_time"] = gen_time
                metrics["phrase"] = phrase_name
                metrics["text"] = phrase_text

                # Save audio
                output_path = output_subdir / f"{phrase_name}.wav"
                wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))

                print(f"    {metrics['duration']:.1f}s | {metrics['clicks']} clicks | {gen_time:.1f}s gen")
                results.append(metrics)
            else:
                print(f"    No audio generated")
                results.append({"phrase": phrase_name, "error": "No audio"})

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"phrase": phrase_name, "error": str(e)})

        torch.cuda.empty_cache()

    # Cleanup
    RealStreamingTTSEngine.VOICE_PROMPT_PATH = original_path
    del tts
    torch.cuda.empty_cache()

    return results


def main():
    print("=" * 70)
    print("VOICE PROMPT COMPARISON TEST")
    print("Testing CSM base model with different voice prompts")
    print("=" * 70)

    all_results = {}

    for vp_name, vp_path in VOICE_PROMPTS:
        results = test_voice_prompt(vp_name, vp_path)
        all_results[vp_name] = results

        # Force cleanup between tests
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for vp_name, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        if valid:
            avg_clicks = np.mean([r["clicks"] for r in valid])
            avg_harsh = np.mean([r["harsh_clicks"] for r in valid])
            avg_dur = np.mean([r["duration"] for r in valid])
            total_errors = len(results) - len(valid)

            print(f"\n{vp_name}:")
            print(f"  Successful: {len(valid)}/{len(results)}")
            print(f"  Avg clicks: {avg_clicks:.1f} (harsh: {avg_harsh:.1f})")
            print(f"  Avg duration: {avg_dur:.1f}s")
            if total_errors > 0:
                print(f"  Errors: {total_errors}")
        else:
            print(f"\n{vp_name}: All failed")

    # Save results
    results_path = OUTPUT_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {results_path}")
    print(f"Audio samples: {OUTPUT_DIR}/")
    print("\n" + "=" * 70)
    print("LISTEN TO THE SAMPLES AND COMPARE QUALITY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
