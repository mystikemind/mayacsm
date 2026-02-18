#!/usr/bin/env python3
"""
Test with Python GC disabled during inference.
"""

import sys
import os
import time
import gc
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
    print("TEST WITH GC DISABLED")
    print("=" * 70)

    # Load components
    print("\nLoading...")
    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    llm = VLLMEngine()
    llm.initialize()

    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Warmup
    print("\nWarmup...")
    for _ in range(5):
        resp = llm.generate("hi")
        for chunk in tts.generate_stream(resp, use_context=False):
            pass
    print("  Done")

    # Force GC now before disabling
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Disable GC
    print("\n*** DISABLING GARBAGE COLLECTION ***")
    gc.disable()

    # Test
    print("\n20 rapid turns with GC disabled:")
    print("-" * 60)

    inputs = [f"test message {i}" for i in range(20)]
    results = []

    for i, text in enumerate(inputs):
        llm_start = time.time()
        response = llm.generate(text)
        llm_time = (time.time() - llm_start) * 1000

        tts_start = time.time()
        for chunk in tts.generate_stream(response, use_context=False):
            break
        tts_time = (time.time() - tts_start) * 1000

        total = llm_time + tts_time
        status = "✓" if total < 250 else "○" if total < 300 else "✗"
        print(f"  [{status}] {i+1:2d}: {total:5.0f}ms (LLM:{llm_time:4.0f} + TTS:{tts_time:4.0f})")

        results.append(total)

    # Re-enable GC
    gc.enable()
    gc.collect()

    # Stats
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  P50: {np.median(results):.0f}ms")
    print(f"  P95: {np.percentile(results, 95):.0f}ms")
    print(f"  Outliers (>300ms): {len([r for r in results if r > 300])}")
    print(f"  Under 250ms: {len([r for r in results if r < 250])}/20")

if __name__ == "__main__":
    main()
