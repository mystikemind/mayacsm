#!/usr/bin/env python3
"""
Test Production Pipeline Warmup

Verifies that the warmup pass eliminates latency spikes.
Target: Consistent P50 ~195ms with NO spikes (>300ms).

This test:
1. Initializes production pipeline (includes warmup)
2. Runs 10 mock turns measuring latency
3. Verifies no latency spikes occur
"""

import sys
import os
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np


def test_warmup():
    print("=" * 70)
    print("PRODUCTION PIPELINE WARMUP TEST")
    print("Verifying warmup eliminates latency spikes")
    print("=" * 70)

    # Import and initialize
    print("\nLoading production pipeline (with warmup)...")
    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    llm = VLLMEngine()
    tts = RealStreamingTTSEngine()

    print("\nInitializing LLM...")
    llm.initialize()

    print("\nInitializing TTS...")
    tts.initialize()

    # Warmup (as production pipeline does)
    print("\n" + "=" * 70)
    print("WARMUP PASS")
    print("=" * 70)

    warmup_start = time.time()

    # LLM warmup
    print("\nLLM warmup...")
    for prompt in ["hi how are you", "whats your name"]:
        _ = llm.generate(prompt)
    llm.clear_history()
    print(f"  LLM warmup: {(time.time() - warmup_start)*1000:.0f}ms")

    # TTS warmup
    tts_start = time.time()
    print("\nTTS warmup...")
    for text in ["hello there", "nice to meet you"]:
        for _ in tts.generate_stream(text, use_context=False):
            pass
    tts.clear_context()
    print(f"  TTS warmup: {(time.time() - tts_start)*1000:.0f}ms")

    print(f"\nTotal warmup: {(time.time() - warmup_start)*1000:.0f}ms")

    # Test turns
    print("\n" + "=" * 70)
    print("POST-WARMUP LATENCY TEST (10 turns)")
    print("=" * 70)

    test_inputs = [
        "hi there",
        "how are you doing today",
        "whats the weather like",
        "tell me a joke",
        "thats funny",
        "what do you think about that",
        "interesting perspective",
        "i agree with you",
        "thanks for chatting",
        "goodbye",
    ]

    llm_times = []
    tts_first_times = []
    total_times = []

    for i, user_input in enumerate(test_inputs):
        turn_start = time.time()

        # LLM
        llm_start = time.time()
        torch.cuda.synchronize()
        response = llm.generate(user_input)
        torch.cuda.synchronize()
        llm_time = (time.time() - llm_start) * 1000

        # TTS (measure first chunk)
        tts_start = time.time()
        first_chunk = True
        first_chunk_time = 0

        for chunk in tts.generate_stream(response, use_context=True):
            if first_chunk:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - tts_start) * 1000
                first_chunk = False

        total_time = (time.time() - turn_start) * 1000
        first_audio = llm_time + first_chunk_time

        llm_times.append(llm_time)
        tts_first_times.append(first_chunk_time)
        total_times.append(first_audio)

        # Check for spike
        spike = "⚠️ SPIKE" if first_audio > 300 else "✓"
        print(f"  Turn {i+1}: LLM={llm_time:.0f}ms, TTS_first={first_chunk_time:.0f}ms, FIRST_AUDIO={first_audio:.0f}ms {spike}")

        # Add TTS context for next turn
        # Skip for last turn
        if i < len(test_inputs) - 1:
            full_audio = tts.generate(response, use_context=False)
            tts.add_context(response, full_audio, is_user=False)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Calculate percentiles
    llm_p50 = np.percentile(llm_times, 50)
    llm_p95 = np.percentile(llm_times, 95)
    llm_max = max(llm_times)

    tts_p50 = np.percentile(tts_first_times, 50)
    tts_p95 = np.percentile(tts_first_times, 95)
    tts_max = max(tts_first_times)

    total_p50 = np.percentile(total_times, 50)
    total_p95 = np.percentile(total_times, 95)
    total_max = max(total_times)

    print(f"\nLLM:       P50={llm_p50:.0f}ms, P95={llm_p95:.0f}ms, max={llm_max:.0f}ms")
    print(f"TTS first: P50={tts_p50:.0f}ms, P95={tts_p95:.0f}ms, max={tts_max:.0f}ms")
    print(f"TOTAL:     P50={total_p50:.0f}ms, P95={total_p95:.0f}ms, max={total_max:.0f}ms")

    # Check for spikes
    spikes = sum(1 for t in total_times if t > 300)
    print(f"\nSpikes (>300ms): {spikes}/{len(total_times)}")

    # Verdict
    print("\n" + "=" * 70)
    if spikes == 0 and total_p50 < 200:
        print("✅ SUCCESS: Warmup eliminates latency spikes!")
        print(f"   P50={total_p50:.0f}ms < 200ms target")
        print(f"   Zero spikes detected")
    elif spikes == 0:
        print("⚠️ PARTIAL: No spikes but P50 above target")
        print(f"   P50={total_p50:.0f}ms (target: <200ms)")
    else:
        print("❌ ISSUE: Latency spikes still present")
        print(f"   {spikes} spikes detected")
        print(f"   Max latency: {total_max:.0f}ms")
    print("=" * 70)


if __name__ == "__main__":
    test_warmup()
