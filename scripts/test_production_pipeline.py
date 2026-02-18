#!/usr/bin/env python3
"""
PRODUCTION PIPELINE TEST - Comprehensive validation of fine-tuned Maya TTS.

Tests:
1. Model loading and initialization
2. Voice quality across multiple phrases
3. Streaming latency (time to first audio)
4. Audio artifact detection
5. Consistency across repeated generations
6. End-to-end pipeline integration

Target: Sesame AI level quality
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.io.wavfile as wav

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
OUTPUT_DIR = PROJECT_ROOT / "audio_production_test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Comprehensive test phrases covering all scenarios
TEST_PHRASES = [
    # Basic greetings
    ("greeting_1", "hi there how are you doing today"),
    ("greeting_2", "hey whats up"),
    ("greeting_3", "hello its so nice to meet you"),

    # Emotional responses
    ("happy_1", "oh wow thats amazing i love it"),
    ("happy_2", "thats so exciting congratulations"),
    ("sad_1", "oh no im so sorry to hear that"),
    ("sad_2", "that sounds really difficult"),

    # Thinking/processing (historically problematic)
    ("thinking_1", "hmm let me think about that"),
    ("thinking_2", "well thats an interesting question"),

    # Questions
    ("question_1", "wait what do you mean by that"),
    ("question_2", "really are you sure about that"),

    # Agreement/acknowledgment
    ("agree_1", "yeah that makes a lot of sense"),
    ("agree_2", "i totally agree with you"),

    # Natural conversation
    ("natural_1", "you know i was just thinking the same thing"),
    ("natural_2", "oh i see what you mean now"),

    # Longer responses
    ("long_1", "thats a great point actually and i think youre right about that"),
    ("long_2", "let me explain what i mean by that so you understand better"),
]


def calculate_quality_metrics(audio_np: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive audio quality metrics."""
    sr = 24000

    metrics = {
        "duration": len(audio_np) / sr,
        "peak": float(np.abs(audio_np).max()),
        "rms": float(np.sqrt(np.mean(audio_np ** 2))),
    }

    # Click/artifact detection
    diff = np.abs(np.diff(audio_np))
    metrics["clicks"] = int(np.sum(diff > 0.3))
    metrics["harsh_clicks"] = int(np.sum(diff > 0.5))
    metrics["max_diff"] = float(diff.max())

    # Silence analysis
    silence_threshold = 0.01
    metrics["silence_ratio"] = float(np.sum(np.abs(audio_np) < silence_threshold) / len(audio_np))

    # Dynamic range
    if metrics["rms"] > 0:
        metrics["crest_factor"] = float(metrics["peak"] / metrics["rms"])
    else:
        metrics["crest_factor"] = 0

    # Spectral analysis
    try:
        from scipy import signal
        freqs, psd = signal.welch(audio_np, sr, nperseg=min(2048, len(audio_np)//4))
        low = np.sum(psd[freqs < 500])
        mid = np.sum(psd[(freqs >= 500) & (freqs < 2000)])
        high = np.sum(psd[freqs >= 2000])
        total = low + mid + high + 1e-10
        metrics["low_freq_ratio"] = float(low / total)
        metrics["mid_freq_ratio"] = float(mid / total)
        metrics["high_freq_ratio"] = float(high / total)
    except:
        metrics["low_freq_ratio"] = 0
        metrics["mid_freq_ratio"] = 0
        metrics["high_freq_ratio"] = 0

    return metrics


def test_streaming_tts():
    """Test the production TTS pipeline with fine-tuned model."""
    print("=" * 70)
    print("PRODUCTION PIPELINE TEST")
    print("Fine-tuned CSM with Sesame AI optimizations")
    print("=" * 70)

    # Import TTS engine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    print("\n[1/5] Initializing TTS engine...")
    start = time.time()
    tts = RealStreamingTTSEngine()
    tts.initialize()
    init_time = time.time() - start
    print(f"      Initialization: {init_time:.1f}s")

    print("\n[2/5] Testing streaming generation...")
    results = []
    first_chunk_times = []

    for phrase_name, phrase_text in TEST_PHRASES:
        print(f"\n  '{phrase_text[:40]}{'...' if len(phrase_text) > 40 else ''}'")

        try:
            start = time.time()
            first_chunk_time = None
            chunks = []

            for chunk in tts.generate_stream(phrase_text, use_context=True):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start
                    first_chunk_times.append(first_chunk_time)
                chunks.append(chunk)

            total_time = time.time() - start

            if chunks:
                audio = torch.cat(chunks)
                audio_np = audio.cpu().numpy()

                # Normalize for saving
                if audio_np.max() > 0:
                    audio_np = audio_np / max(abs(audio_np.min()), abs(audio_np.max())) * 0.9

                metrics = calculate_quality_metrics(audio_np)
                metrics["first_chunk_ms"] = first_chunk_time * 1000
                metrics["total_time"] = total_time
                metrics["phrase"] = phrase_name
                metrics["text"] = phrase_text

                # Save audio
                output_path = OUTPUT_DIR / f"{phrase_name}.wav"
                wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))

                status = "OK" if metrics["clicks"] < 50 and metrics["duration"] > 0.5 else "CHECK"
                print(f"    [{status}] {metrics['duration']:.1f}s | {metrics['clicks']} clicks | First: {first_chunk_time*1000:.0f}ms")

                results.append(metrics)
            else:
                print(f"    [FAIL] No audio generated")
                results.append({"phrase": phrase_name, "error": "No audio"})

        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()
            results.append({"phrase": phrase_name, "error": str(e)})

        torch.cuda.empty_cache()

    # Analysis
    print("\n" + "=" * 70)
    print("[3/5] QUALITY ANALYSIS")
    print("=" * 70)

    valid_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]

    if valid_results:
        avg_duration = np.mean([r["duration"] for r in valid_results])
        avg_clicks = np.mean([r["clicks"] for r in valid_results])
        avg_harsh = np.mean([r["harsh_clicks"] for r in valid_results])
        avg_first_chunk = np.mean([r["first_chunk_ms"] for r in valid_results])
        avg_rms = np.mean([r["rms"] for r in valid_results])

        print(f"\n  Successful: {len(valid_results)}/{len(results)}")
        print(f"  Average duration: {avg_duration:.1f}s")
        print(f"  Average clicks: {avg_clicks:.0f} (harsh: {avg_harsh:.0f})")
        print(f"  Average first chunk: {avg_first_chunk:.0f}ms")
        print(f"  Average RMS energy: {avg_rms:.3f}")

        # Quality thresholds
        print("\n  Quality Thresholds:")
        click_pass = avg_clicks < 30
        latency_pass = avg_first_chunk < 500
        print(f"    Clicks < 30: {'PASS' if click_pass else 'FAIL'} ({avg_clicks:.0f})")
        print(f"    First chunk < 500ms: {'PASS' if latency_pass else 'FAIL'} ({avg_first_chunk:.0f}ms)")

    if failed_results:
        print(f"\n  Failed generations: {len(failed_results)}")
        for r in failed_results:
            print(f"    - {r['phrase']}: {r.get('error', 'Unknown')}")

    # Consistency test
    print("\n" + "=" * 70)
    print("[4/5] CONSISTENCY TEST (3 generations of same phrase)")
    print("=" * 70)

    test_phrase = "hi there how are you"
    consistency_results = []

    for i in range(3):
        try:
            start = time.time()
            chunks = list(tts.generate_stream(test_phrase, use_context=False))
            gen_time = time.time() - start

            if chunks:
                audio = torch.cat(chunks)
                audio_np = audio.cpu().numpy()
                metrics = calculate_quality_metrics(audio_np)

                # Save
                output_path = OUTPUT_DIR / f"consistency_test_{i+1}.wav"
                audio_np_norm = audio_np / max(abs(audio_np.min()), abs(audio_np.max())) * 0.9
                wav.write(str(output_path), 24000, (audio_np_norm * 32767).astype(np.int16))

                print(f"  Run {i+1}: {metrics['duration']:.1f}s | {metrics['clicks']} clicks | RMS: {metrics['rms']:.3f}")
                consistency_results.append(metrics)
        except Exception as e:
            print(f"  Run {i+1}: ERROR - {e}")

        torch.cuda.empty_cache()

    if len(consistency_results) >= 2:
        dur_std = np.std([r["duration"] for r in consistency_results])
        rms_std = np.std([r["rms"] for r in consistency_results])
        print(f"\n  Duration variance: {dur_std:.2f}s (lower is better)")
        print(f"  Energy variance: {rms_std:.4f} (lower is better)")

    # Summary
    print("\n" + "=" * 70)
    print("[5/5] PRODUCTION READINESS ASSESSMENT")
    print("=" * 70)

    if valid_results:
        overall_pass = (
            len(valid_results) / len(results) >= 0.9 and
            avg_clicks < 50 and
            avg_first_chunk < 600
        )

        print(f"\n  Success rate: {len(valid_results)/len(results)*100:.0f}%")
        print(f"  Audio quality: {'GOOD' if avg_clicks < 30 else 'ACCEPTABLE' if avg_clicks < 50 else 'NEEDS WORK'}")
        print(f"  Latency: {'EXCELLENT' if avg_first_chunk < 300 else 'GOOD' if avg_first_chunk < 500 else 'ACCEPTABLE'}")
        print(f"\n  OVERALL: {'PRODUCTION READY' if overall_pass else 'NEEDS IMPROVEMENT'}")

    # Save results
    results_path = OUTPUT_DIR / "production_test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "total": len(results),
                "successful": len(valid_results),
                "failed": len(failed_results),
                "avg_clicks": avg_clicks if valid_results else None,
                "avg_first_chunk_ms": avg_first_chunk if valid_results else None,
            }
        }, f, indent=2, default=str)

    print(f"\n  Results saved: {results_path}")
    print(f"  Audio samples: {OUTPUT_DIR}/")
    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    test_streaming_tts()
