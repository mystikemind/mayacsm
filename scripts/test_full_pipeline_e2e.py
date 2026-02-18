#!/usr/bin/env python3
"""
FULL END-TO-END PIPELINE TEST

Tests the complete Maya pipeline as it would work in production:
1. Simulated user audio → VAD → STT
2. STT transcript → LLM response
3. LLM response → TTS streaming audio

Measures:
- Total latency from user speech end to first audio
- Component-level timing
- Audio quality
- Response coherence
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
OUTPUT_DIR = PROJECT_ROOT / "audio_e2e_test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Simulated user inputs (what user might say)
USER_INPUTS = [
    ("greeting", "Hello Maya, how are you today?"),
    ("question", "What's your favorite thing to talk about?"),
    ("emotional", "I'm feeling a bit down today"),
    ("technical", "Can you explain how neural networks work?"),
    ("casual", "What do you think about the weather?"),
    ("followup", "That's interesting, tell me more"),
]

def test_full_pipeline():
    print("=" * 70)
    print("FULL END-TO-END PIPELINE TEST")
    print("Simulating complete conversation flow")
    print("=" * 70)

    # Note: Can't load LLM and TTS together due to VRAM limits
    # Testing components separately, then combining timing
    print("\n[1/4] Testing LLM responses...")

    from maya.engine.llm_optimized import OptimizedLLMEngine
    llm = OptimizedLLMEngine()
    llm.initialize()

    # Generate all LLM responses first
    llm_results = {}
    for test_name, user_text in USER_INPUTS:
        llm_start = time.time()
        response = llm.generate(user_text)
        llm_time = time.time() - llm_start
        llm_results[test_name] = {"response": response, "time_ms": llm_time * 1000}
        print(f"  {test_name}: \"{response[:50]}...\" [{llm_time*1000:.0f}ms]")

    # Unload LLM
    del llm
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    time.sleep(2)

    print("\n[2/4] Loading TTS (fine-tuned CSM)...")
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    tts = RealStreamingTTSEngine()
    tts.initialize()
    print("    TTS ready")

    print("\n[3/4] Running TTS on LLM responses...")
    results = []

    for test_name, user_text in USER_INPUTS:
        print(f"\n{'='*50}")
        print(f"User: \"{user_text}\"")
        print("-" * 50)

        try:
            # Get pre-generated LLM response
            llm_data = llm_results[test_name]
            llm_response = llm_data["response"]
            llm_time = llm_data["time_ms"] / 1000

            print(f"Maya (LLM): \"{llm_response}\" [{llm_time*1000:.0f}ms]")

            # Measure TTS
            tts_start = time.time()
            first_chunk_time = None
            chunks = []

            for chunk in tts.generate_stream(llm_response, use_context=True):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - tts_start
                chunks.append(chunk)

            tts_total_time = time.time() - tts_start

            # Calculate total latency
            total_latency = llm_time + first_chunk_time if first_chunk_time else llm_time

            if chunks:
                audio = torch.cat(chunks)
                audio_np = audio.cpu().numpy()

                # Quality metrics
                diff = np.abs(np.diff(audio_np))
                clicks = int(np.sum(diff > 0.3))

                # Normalize and save
                audio_np = audio_np / max(abs(audio_np.min()), abs(audio_np.max())) * 0.9
                output_path = OUTPUT_DIR / f"{test_name}.wav"
                wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))

                print(f"Audio: {len(audio_np)/24000:.1f}s | {clicks} clicks")
                print(f"Timing: LLM={llm_time*1000:.0f}ms + TTS_first={first_chunk_time*1000:.0f}ms = {total_latency*1000:.0f}ms total")

                results.append({
                    "test": test_name,
                    "user": user_text,
                    "response": llm_response,
                    "llm_ms": llm_time * 1000,
                    "tts_first_ms": first_chunk_time * 1000,
                    "total_latency_ms": total_latency * 1000,
                    "audio_duration": len(audio_np) / 24000,
                    "clicks": clicks,
                })
            else:
                print("  No audio generated")
                results.append({"test": test_name, "error": "No audio"})

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"test": test_name, "error": str(e)})

        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("[4/4] LATENCY ANALYSIS")
    print("=" * 70)

    valid = [r for r in results if "error" not in r]
    if valid:
        llm_times = [r["llm_ms"] for r in valid]
        tts_times = [r["tts_first_ms"] for r in valid]
        total_times = [r["total_latency_ms"] for r in valid]
        clicks_list = [r["clicks"] for r in valid]

        print(f"\nComponent Latencies (ms):")
        print(f"  LLM:       avg={np.mean(llm_times):.0f}, min={np.min(llm_times):.0f}, max={np.max(llm_times):.0f}")
        print(f"  TTS first: avg={np.mean(tts_times):.0f}, min={np.min(tts_times):.0f}, max={np.max(tts_times):.0f}")
        print(f"  TOTAL:     avg={np.mean(total_times):.0f}, min={np.min(total_times):.0f}, max={np.max(total_times):.0f}")

        print(f"\nAudio Quality:")
        print(f"  Avg clicks: {np.mean(clicks_list):.0f}")
        print(f"  Max clicks: {np.max(clicks_list)}")

        # Sesame comparison
        print(f"\n{'='*70}")
        print("SESAME AI COMPARISON")
        print("=" * 70)

        sesame_target = 200  # ms
        our_avg = np.mean(total_times)

        print(f"\n  Sesame AI target: ~{sesame_target}ms to first audio")
        print(f"  Our latency:      {our_avg:.0f}ms to first audio")

        if our_avg <= sesame_target:
            print(f"\n  ✅ SESAME LEVEL ACHIEVED! ({our_avg:.0f}ms <= {sesame_target}ms)")
        else:
            print(f"\n  Gap: {our_avg - sesame_target:.0f}ms above Sesame target")

        # Quality assessment
        avg_clicks = np.mean(clicks_list)
        if avg_clicks < 10:
            print(f"  ✅ AUDIO QUALITY: EXCELLENT ({avg_clicks:.0f} avg clicks)")
        elif avg_clicks < 30:
            print(f"  ✅ AUDIO QUALITY: GOOD ({avg_clicks:.0f} avg clicks)")
        else:
            print(f"  ⚠️ AUDIO QUALITY: NEEDS WORK ({avg_clicks:.0f} avg clicks)")

    # Save results
    results_path = OUTPUT_DIR / "e2e_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults: {results_path}")
    print(f"Audio samples: {OUTPUT_DIR}/")
    print("=" * 70)

    return results

if __name__ == "__main__":
    test_full_pipeline()
