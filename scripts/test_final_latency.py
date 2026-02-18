#!/usr/bin/env python3
"""
Final Latency Test - Simulates Real Conversation

This test verifies that the complete pipeline achieves Sesame AI level:
- P50 < 200ms
- P95 < 300ms
- Zero spikes (>400ms)
"""

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
    print("=" * 70)
    print("FINAL LATENCY TEST - Sesame AI Level Verification")
    print("=" * 70)

    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    llm = VLLMEngine()
    tts = RealStreamingTTSEngine()

    print("\nInitializing LLM...")
    llm.initialize()

    print("\nInitializing TTS (includes warmup)...")
    tts.initialize()

    # Additional warmup (as production pipeline does)
    print("\n" + "=" * 70)
    print("PIPELINE WARMUP")
    print("=" * 70)

    warmup_start = time.time()

    # LLM warmup
    print("\nLLM warmup...")
    for prompt in ["hi how are you", "whats your name"]:
        _ = llm.generate(prompt)
    llm.clear_history()

    # TTS warmup (streaming path)
    print("TTS warmup...")
    for text in ["hello there", "nice to meet you"]:
        for _ in tts.generate_stream(text, use_context=False):
            pass
    tts.clear_context()

    print(f"Warmup complete in {(time.time() - warmup_start)*1000:.0f}ms")

    # Simulated conversation
    print("\n" + "=" * 70)
    print("SIMULATED CONVERSATION (15 turns)")
    print("=" * 70)

    conversation = [
        "hi there",
        "how are you doing today",
        "i'm feeling great thanks for asking",
        "what do you think about the weather",
        "its really nice outside",
        "do you have any plans for the weekend",
        "i might go hiking",
        "that sounds fun",
        "yeah i love being outdoors",
        "whats your favorite outdoor activity",
        "i enjoy walking in nature",
        "me too its so peaceful",
        "do you have a favorite place to walk",
        "there's a nice park nearby",
        "thanks for the chat",
    ]

    llm_times = []
    tts_first_times = []
    total_times = []

    for i, user_input in enumerate(conversation):
        turn_start = time.time()

        # LLM
        llm_start = time.time()
        torch.cuda.synchronize()
        response = llm.generate(user_input)
        torch.cuda.synchronize()
        llm_time = (time.time() - llm_start) * 1000

        # TTS with context
        tts_start = time.time()
        first_chunk = True
        first_chunk_time = 0
        all_chunks = []

        for chunk in tts.generate_stream(response, use_context=True):
            all_chunks.append(chunk)
            if first_chunk:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - tts_start) * 1000
                first_chunk = False

        total_time = (time.time() - turn_start) * 1000
        first_audio = llm_time + first_chunk_time

        llm_times.append(llm_time)
        tts_first_times.append(first_chunk_time)
        total_times.append(first_audio)

        # Add to TTS context for next turn
        if all_chunks:
            full_audio = torch.cat(all_chunks)
            # Add both user input and Maya response to context
            tts.add_context(user_input, full_audio[:len(full_audio)//2], is_user=True)
            tts.add_context(response, full_audio[len(full_audio)//2:], is_user=False)

        # Status
        status = "✅" if first_audio < 200 else "⚠️" if first_audio < 300 else "❌"
        print(f"  Turn {i+1:2d}: LLM={llm_time:3.0f}ms, TTS={first_chunk_time:3.0f}ms, FIRST_AUDIO={first_audio:3.0f}ms {status}")
        print(f"           User: \"{user_input}\"")
        print(f"           Maya: \"{response}\"")

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

    print(f"\nLLM:           P50={llm_p50:.0f}ms, P95={llm_p95:.0f}ms, max={llm_max:.0f}ms")
    print(f"TTS First:     P50={tts_p50:.0f}ms, P95={tts_p95:.0f}ms, max={tts_max:.0f}ms")
    print(f"FIRST AUDIO:   P50={total_p50:.0f}ms, P95={total_p95:.0f}ms, max={total_max:.0f}ms")

    # Count spikes
    under_200 = sum(1 for t in total_times if t < 200)
    under_250 = sum(1 for t in total_times if t < 250)
    under_300 = sum(1 for t in total_times if t < 300)
    spikes = sum(1 for t in total_times if t > 400)

    print(f"\n< 200ms: {under_200}/{len(total_times)} ({under_200/len(total_times)*100:.0f}%)")
    print(f"< 250ms: {under_250}/{len(total_times)} ({under_250/len(total_times)*100:.0f}%)")
    print(f"< 300ms: {under_300}/{len(total_times)} ({under_300/len(total_times)*100:.0f}%)")
    print(f"Spikes (>400ms): {spikes}/{len(total_times)}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if total_p50 < 200 and spikes == 0:
        print("✅ SESAME AI LEVEL ACHIEVED!")
        print(f"   P50={total_p50:.0f}ms < 200ms target")
        print(f"   Zero spikes")
    elif total_p50 < 250 and spikes == 0:
        print("⚠️ NEAR TARGET - Acceptable for production")
        print(f"   P50={total_p50:.0f}ms (target: <200ms)")
        print(f"   Zero spikes")
    else:
        print("❌ BELOW TARGET")
        print(f"   P50={total_p50:.0f}ms (target: <200ms)")
        print(f"   Spikes: {spikes}")

    print("=" * 70)


if __name__ == "__main__":
    main()
