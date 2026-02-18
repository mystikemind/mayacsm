#!/usr/bin/env python3
"""
Stable latency test - run multiple iterations to measure true performance.
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

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")

def main():
    print("=" * 70)
    print("STABLE LATENCY TEST - 10 Iterations")
    print("=" * 70)

    # Load vLLM
    print("\nLoading vLLM...")
    from maya.engine.llm_vllm import VLLMEngine
    llm = VLLMEngine()
    llm.initialize()

    # Clear memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Load TTS
    print("\nLoading TTS on GPU 1...")
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Extra warmup for TTS
    print("\nExtra TTS warmup (5 iterations)...")
    warmup_texts = ["hello there", "how are you", "thats great", "oh wow", "nice"]
    for text in warmup_texts:
        for chunk in tts.generate_stream(text, use_context=False):
            pass
        torch.cuda.empty_cache()
    print("  Warmup complete")

    # Test iterations
    print("\n" + "=" * 70)
    print("Running 10 test iterations...")
    print("=" * 70)

    test_inputs = [
        "Hello how are you",
        "Whats your favorite thing",
        "Tell me something fun",
        "Thats really cool",
        "I love that idea",
        "How does that work",
        "What do you think",
        "Im feeling great today",
        "That sounds amazing",
        "You are so smart",
    ]

    results = []

    for i, user_input in enumerate(test_inputs):
        # LLM
        llm_start = time.time()
        response = llm.generate(user_input)
        llm_time = (time.time() - llm_start) * 1000

        # TTS
        tts_start = time.time()
        first_chunk_time = None

        for chunk in tts.generate_stream(response, use_context=False):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - tts_start) * 1000
            break

        total = llm_time + first_chunk_time

        status = "✓" if total < 300 else "○" if total < 400 else "✗"
        print(f"  [{status}] {i+1:2d}: LLM={llm_time:5.0f}ms + TTS={first_chunk_time:5.0f}ms = {total:5.0f}ms | '{response[:30]}...'")

        results.append({
            "llm": llm_time,
            "tts": first_chunk_time,
            "total": total
        })

        torch.cuda.empty_cache()

    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS (excluding outliers)")
    print("=" * 70)

    # Remove outliers (>2x median)
    totals = [r["total"] for r in results]
    median = np.median(totals)
    clean_results = [r for r in results if r["total"] < median * 2]

    if len(clean_results) < len(results):
        print(f"  Removed {len(results) - len(clean_results)} outlier(s)")

    llm_times = [r["llm"] for r in clean_results]
    tts_times = [r["tts"] for r in clean_results]
    total_times = [r["total"] for r in clean_results]

    print(f"\n  LLM:   avg={np.mean(llm_times):5.0f}ms  p50={np.median(llm_times):5.0f}ms  p95={np.percentile(llm_times, 95):5.0f}ms")
    print(f"  TTS:   avg={np.mean(tts_times):5.0f}ms  p50={np.median(tts_times):5.0f}ms  p95={np.percentile(tts_times, 95):5.0f}ms")
    print(f"  TOTAL: avg={np.mean(total_times):5.0f}ms  p50={np.median(total_times):5.0f}ms  p95={np.percentile(total_times, 95):5.0f}ms")

    under_250 = len([t for t in total_times if t < 250])
    under_300 = len([t for t in total_times if t < 300])

    print(f"\n  Under 250ms: {under_250}/{len(clean_results)} ({under_250/len(clean_results)*100:.0f}%)")
    print(f"  Under 300ms: {under_300}/{len(clean_results)} ({under_300/len(clean_results)*100:.0f}%)")

    print(f"\n  Sesame Target: ~200ms")
    print(f"  Our P50:       {np.median(total_times):.0f}ms")
    gap = np.median(total_times) - 200
    print(f"  Gap:           {gap:.0f}ms")

    if np.median(total_times) <= 250:
        print("\n  ✅ SESAME LEVEL ACHIEVED (within acceptable margin)")
    elif np.median(total_times) <= 300:
        print("\n  ○ CLOSE TO SESAME LEVEL")
    else:
        print("\n  ✗ NEEDS OPTIMIZATION")

if __name__ == "__main__":
    main()
