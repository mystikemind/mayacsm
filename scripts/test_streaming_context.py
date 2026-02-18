#!/usr/bin/env python3
"""
Test Streaming TTS with Conversation Context

This script tests the key improvements:
1. Streaming audio generation
2. Conversation context for prosodic consistency
3. Time to first audio chunk

Run: python scripts/test_streaming_context.py
"""

import sys
import time
import asyncio
import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from maya.engine.tts_streaming import StreamingTTSEngine


async def test_streaming():
    """Test streaming generation with context."""
    print("=" * 60)
    print("TESTING STREAMING TTS WITH CONVERSATION CONTEXT")
    print("=" * 60)

    # Initialize
    print("\n[1] Initializing StreamingTTSEngine...")
    tts = StreamingTTSEngine()
    init_start = time.time()
    tts.initialize()
    init_time = time.time() - init_start
    print(f"    Initialized in {init_time:.1f}s")
    print(f"    Voice prompt loaded: {tts._voice_prompt is not None}")

    # Test 1: Streaming without context
    print("\n[2] Testing streaming WITHOUT context...")
    test_text = "Hello, how are you doing today?"

    start = time.time()
    first_chunk_time = None
    chunk_count = 0
    total_samples = 0

    async for chunk in tts.generate_stream(test_text, use_context=False):
        if first_chunk_time is None:
            first_chunk_time = (time.time() - start) * 1000
        chunk_count += 1
        total_samples += len(chunk)

    total_time = (time.time() - start) * 1000
    audio_duration = total_samples / 24000

    print(f"    First chunk: {first_chunk_time:.0f}ms")
    print(f"    Total time: {total_time:.0f}ms")
    print(f"    Chunks: {chunk_count}")
    print(f"    Audio: {audio_duration:.2f}s")
    print(f"    RTF: {total_time / 1000 / audio_duration:.2f}x")

    # Test 2: Add context (simulating conversation)
    print("\n[3] Adding conversation context...")

    # Simulate user saying something
    user_text = "What's the weather like?"
    user_audio = torch.randn(24000 * 2)  # 2 seconds fake audio
    tts.add_context(user_text, user_audio, is_user=True)
    print(f"    Added user turn: '{user_text}'")

    # Simulate Maya responding
    maya_text = "It's a beautiful sunny day"
    maya_audio = torch.randn(24000 * 2)  # 2 seconds fake audio
    tts.add_context(maya_text, maya_audio, is_user=False)
    print(f"    Added Maya turn: '{maya_text}'")

    print(f"    Context turns: {len(tts._context)}")

    # Test 3: Streaming WITH context
    print("\n[4] Testing streaming WITH context...")
    test_text2 = "That sounds wonderful!"

    start = time.time()
    first_chunk_time = None
    chunk_count = 0
    total_samples = 0

    async for chunk in tts.generate_stream(test_text2, use_context=True):
        if first_chunk_time is None:
            first_chunk_time = (time.time() - start) * 1000
        chunk_count += 1
        total_samples += len(chunk)

    total_time = (time.time() - start) * 1000
    audio_duration = total_samples / 24000

    print(f"    First chunk: {first_chunk_time:.0f}ms")
    print(f"    Total time: {total_time:.0f}ms")
    print(f"    Chunks: {chunk_count}")
    print(f"    Audio: {audio_duration:.2f}s")
    print(f"    RTF: {total_time / 1000 / audio_duration:.2f}x")

    # Test 4: Multiple turns to see context effect
    print("\n[5] Testing multiple turns...")

    turns = [
        "Nice to meet you",
        "How can I help you today",
        "That's a great question",
    ]

    for i, text in enumerate(turns):
        start = time.time()
        first_chunk_time = None
        chunks = []

        async for chunk in tts.generate_stream(text, use_context=True):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - start) * 1000
            chunks.append(chunk)

        if chunks:
            audio = torch.cat(chunks)
            tts.add_context(text, audio, is_user=False)

        total_time = (time.time() - start) * 1000
        print(f"    Turn {i+1}: '{text[:30]}...' -> first_chunk={first_chunk_time:.0f}ms, total={total_time:.0f}ms")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    # Summary
    stats = tts.get_stats()
    print(f"\nStats:")
    print(f"  Total generations: {stats['total_generations']}")
    print(f"  Avg first chunk: {stats['avg_first_chunk_ms']:.0f}ms")
    print(f"  Total audio: {stats['total_audio_seconds']:.1f}s")
    print(f"  Context turns: {stats['context_turns']}")


if __name__ == "__main__":
    asyncio.run(test_streaming())
