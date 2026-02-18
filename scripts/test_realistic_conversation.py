#!/usr/bin/env python3
"""
Realistic conversation test - simulates actual usage patterns.

Key insight: Don't clear GPU cache between calls (causes reallocation spikes).
In real conversations, calls happen in rapid succession.
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

def main():
    print("=" * 70)
    print("REALISTIC CONVERSATION TEST")
    print("Simulating actual usage (no artificial memory clearing)")
    print("=" * 70)

    # Load components
    print("\nLoading vLLM...")
    from maya.engine.llm_vllm import VLLMEngine
    llm = VLLMEngine()
    llm.initialize()

    print("\nLoading TTS...")
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Warmup (important for CUDA graphs)
    print("\nWarming up...")
    for _ in range(3):
        resp = llm.generate("hello")
        for chunk in tts.generate_stream(resp, use_context=False):
            pass
    print("  Warmup complete")

    # Simulate conversation
    print("\n" + "=" * 70)
    print("SIMULATED CONVERSATION (20 turns)")
    print("=" * 70)

    conversation = [
        "Hello Maya",
        "How are you today",
        "What do you like to talk about",
        "Tell me something interesting",
        "Wow thats cool",
        "What else do you know",
        "I love learning new things",
        "You are really smart",
        "What is your favorite topic",
        "That sounds fascinating",
        "Can you explain more",
        "I see what you mean",
        "That makes sense",
        "What do you think about that",
        "Interesting perspective",
        "I agree with you",
        "Thanks for explaining",
        "You are great at this",
        "One more question",
        "Goodbye Maya",
    ]

    results = []

    for i, user_input in enumerate(conversation):
        # LLM
        llm_start = time.time()
        response = llm.generate(user_input)
        llm_time = (time.time() - llm_start) * 1000

        # TTS (no memory clearing!)
        tts_start = time.time()
        first_chunk_time = None

        for chunk in tts.generate_stream(response, use_context=False):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - tts_start) * 1000
            # Keep generating but only measure first chunk
            pass

        total = llm_time + first_chunk_time

        status = "✓" if total < 250 else "○" if total < 300 else "✗"
        print(f"  [{status}] Turn {i+1:2d}: {total:5.0f}ms (LLM:{llm_time:4.0f} + TTS:{first_chunk_time:4.0f}) | '{response[:35]}...'")

        results.append({
            "turn": i + 1,
            "llm": llm_time,
            "tts": first_chunk_time,
            "total": total,
        })

    # Statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    llm_times = [r["llm"] for r in results]
    tts_times = [r["tts"] for r in results]
    total_times = [r["total"] for r in results]

    print(f"\n  Turns: {len(results)}")
    print(f"\n  LLM:   avg={np.mean(llm_times):5.0f}ms  p50={np.median(llm_times):5.0f}ms  p95={np.percentile(llm_times, 95):5.0f}ms  min={min(llm_times):5.0f}ms  max={max(llm_times):5.0f}ms")
    print(f"  TTS:   avg={np.mean(tts_times):5.0f}ms  p50={np.median(tts_times):5.0f}ms  p95={np.percentile(tts_times, 95):5.0f}ms  min={min(tts_times):5.0f}ms  max={max(tts_times):5.0f}ms")
    print(f"  TOTAL: avg={np.mean(total_times):5.0f}ms  p50={np.median(total_times):5.0f}ms  p95={np.percentile(total_times, 95):5.0f}ms  min={min(total_times):5.0f}ms  max={max(total_times):5.0f}ms")

    under_200 = len([t for t in total_times if t < 200])
    under_250 = len([t for t in total_times if t < 250])
    under_300 = len([t for t in total_times if t < 300])
    outliers = len([t for t in total_times if t > 400])

    print(f"\n  Under 200ms: {under_200}/{len(total_times)} ({under_200/len(total_times)*100:.0f}%)")
    print(f"  Under 250ms: {under_250}/{len(total_times)} ({under_250/len(total_times)*100:.0f}%)")
    print(f"  Under 300ms: {under_300}/{len(total_times)} ({under_300/len(total_times)*100:.0f}%)")
    print(f"  Outliers (>400ms): {outliers}/{len(total_times)}")

    print(f"\n  ═══════════════════════════════════════")
    print(f"  Sesame AI Target:  ~200ms")
    print(f"  Our P50:           {np.median(total_times):.0f}ms")
    print(f"  Our Best:          {min(total_times):.0f}ms")
    print(f"  Gap from target:   {np.median(total_times) - 200:.0f}ms")
    print(f"  ═══════════════════════════════════════")

    if np.median(total_times) <= 220:
        print(f"\n  ✅ SESAME AI LEVEL ACHIEVED!")
    elif np.median(total_times) <= 250:
        print(f"\n  ✅ NEAR SESAME LEVEL (within acceptable margin)")
    elif np.median(total_times) <= 300:
        print(f"\n  ○ CLOSE - Minor optimization needed")
    else:
        print(f"\n  ✗ NEEDS SIGNIFICANT OPTIMIZATION")

if __name__ == "__main__":
    main()
