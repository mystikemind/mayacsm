#!/usr/bin/env python3
"""
Diagnose TTS latency outliers.

Goal: Understand why some TTS calls take 400ms instead of 130ms.
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

def main():
    print("=" * 70)
    print("TTS OUTLIER DIAGNOSIS")
    print("=" * 70)

    # Load TTS
    print("\nLoading TTS...")
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Check GPU memory
    print("\nGPU Memory Status:")
    for i in range(min(2, torch.cuda.device_count())):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Warmup
    print("\nWarmup (10 iterations)...")
    warmup_times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        for chunk in tts.generate_stream("hello there friend", use_context=False):
            pass
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        warmup_times.append(elapsed)
        print(f"  Warmup {i+1}: {elapsed:.0f}ms")

    print(f"\n  Warmup avg: {sum(warmup_times)/len(warmup_times):.0f}ms")
    print(f"  Warmup stabilized at iteration: {next((i for i, t in enumerate(warmup_times) if t < 200), 'never')}")

    # Test different text lengths
    print("\n" + "=" * 70)
    print("Testing different text lengths...")
    print("=" * 70)

    test_texts = [
        ("short", "hi there"),
        ("medium", "oh wow that is really cool"),
        ("long", "i think that is a really interesting point and i totally agree with you"),
        ("with_emotion", "[happy] oh wow that is amazing"),
    ]

    for name, text in test_texts:
        times = []
        first_chunks = []

        for _ in range(5):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            start = time.time()
            first_chunk_time = None

            for chunk in tts.generate_stream(text, use_context=False):
                if first_chunk_time is None:
                    torch.cuda.synchronize()
                    first_chunk_time = (time.time() - start) * 1000
                break  # Only measure first chunk

            times.append(first_chunk_time)
            first_chunks.append(first_chunk_time)

        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        print(f"\n  {name}: avg={avg:.0f}ms, min={min_t:.0f}ms, max={max_t:.0f}ms")
        print(f"    All times: {[f'{t:.0f}' for t in times]}")

    # Test rapid succession
    print("\n" + "=" * 70)
    print("Testing rapid succession (no cleanup between)...")
    print("=" * 70)

    rapid_times = []
    for i in range(10):
        start = time.time()
        for chunk in tts.generate_stream("hello friend", use_context=False):
            break
        elapsed = (time.time() - start) * 1000
        rapid_times.append(elapsed)

    print(f"  Times: {[f'{t:.0f}' for t in rapid_times]}")
    print(f"  Avg: {sum(rapid_times)/len(rapid_times):.0f}ms")
    print(f"  Outliers (>200ms): {len([t for t in rapid_times if t > 200])}")

    # Conclusion
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    outlier_count = len([t for t in rapid_times if t > 200])
    if outlier_count == 0:
        print("\n  ✅ No outliers detected in rapid succession test")
        print("  Outliers may be caused by GC or memory allocation between tests")
    else:
        print(f"\n  ⚠️ {outlier_count} outliers detected")
        print("  Possible causes:")
        print("    - CUDA graph recompilation")
        print("    - Kernel compilation")
        print("    - Memory fragmentation")

if __name__ == "__main__":
    main()
