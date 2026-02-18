#!/usr/bin/env python3
"""Test TTS context overhead with pre-tokenization optimization."""

import sys
import os
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np


def main():
    print("=" * 60)
    print("TTS PRE-TOKENIZATION OPTIMIZATION TEST")
    print("=" * 60)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    tts = RealStreamingTTSEngine()
    tts.initialize()

    print("\n--- Test WITHOUT context (use_context=False) ---")
    times_without = []
    for i in range(5):
        start = time.time()
        torch.cuda.synchronize()
        first = True
        for chunk in tts.generate_stream('oh thats really cool', use_context=False):
            if first:
                torch.cuda.synchronize()
                first_time = (time.time() - start) * 1000
                first = False
                break
        times_without.append(first_time)
        print(f"  Turn {i+1}: TTS_first={first_time:.0f}ms")

    print("\n--- Test WITH context (use_context=True) ---")
    tts.clear_context()

    times_with = []
    for i in range(5):
        text = f"response number {i+1} is here"

        start = time.time()
        torch.cuda.synchronize()
        first = True
        all_chunks = []
        for chunk in tts.generate_stream(text, use_context=True):
            all_chunks.append(chunk)
            if first:
                torch.cuda.synchronize()
                first_time = (time.time() - start) * 1000
                first = False

        if all_chunks:
            full_audio = torch.cat(all_chunks)
            tts.add_context(text, full_audio, is_user=False)

        times_with.append(first_time)
        context_len = len(tts._context)
        context_tokens = tts._estimate_context_tokens()
        print(f"  Turn {i+1}: TTS_first={first_time:.0f}ms (context: {context_len} turns, ~{context_tokens} tokens)")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Without context: P50={np.percentile(times_without, 50):.0f}ms, max={max(times_without):.0f}ms")
    print(f"With context:    P50={np.percentile(times_with, 50):.0f}ms, max={max(times_with):.0f}ms")

    # Check improvement
    without_avg = np.mean(times_without)
    with_avg = np.mean(times_with)

    if with_avg < 180:
        print(f"\n*** SUCCESS: Context overhead is now minimal! ***")
        print(f"    With context avg: {with_avg:.0f}ms (target: <200ms)")
    elif with_avg < without_avg * 2:
        print(f"\n*** IMPROVED: Context overhead reduced ***")
        print(f"    With context avg: {with_avg:.0f}ms")
    else:
        print(f"\n*** ISSUE: Context overhead still high ***")
        print(f"    With context avg: {with_avg:.0f}ms vs without: {without_avg:.0f}ms")


if __name__ == "__main__":
    main()
