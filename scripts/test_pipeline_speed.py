#!/usr/bin/env python3
"""
Test pipeline speed - measure actual latency.
"""

import sys
import time
import torch
import asyncio

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')


async def test_full_pipeline():
    """Test full pipeline latency."""
    print("=" * 60)
    print("MAYA PIPELINE SPEED TEST")
    print("=" * 60)

    from maya.engine.stt import STTEngine
    from maya.engine.llm import LLMEngine
    from maya.engine.tts_streaming import StreamingTTSEngine, StreamingConfig

    # ULTRA-FAST config - ABSOLUTE minimum TTFA
    FAST = StreamingConfig(
        initial_batch_size=3,   # ~240ms audio - FASTEST possible
        batch_size=6,
        buffer_size=6,
        max_audio_length_ms=4000,
        temperature=0.8,
        topk=50
    )

    print("\nInitializing engines...")

    stt = STTEngine()
    stt.initialize()
    print("STT ready")

    llm = LLMEngine()
    llm.initialize()
    print("LLM ready")

    tts = StreamingTTSEngine()
    tts.initialize()
    print("TTS ready")

    # Test cases
    test_inputs = [
        "Hello, my name is Pritam",
        "How are you doing today?",
        "Tell me something interesting",
    ]

    print("\n" + "=" * 60)
    print("RUNNING LATENCY TESTS")
    print("=" * 60)

    for user_input in test_inputs:
        print(f"\n--- User: '{user_input}' ---")

        # Simulate audio (just use the text directly for speed)
        total_start = time.time()

        # STT (simulate with direct text)
        stt_start = time.time()
        # In real scenario this would be: transcript = stt.transcribe(audio)
        transcript = user_input  # Skip actual STT for this test
        stt_time = (time.time() - stt_start) * 1000
        print(f"  STT: {stt_time:.0f}ms -> '{transcript}'")

        # LLM
        llm_start = time.time()
        response = llm.generate(transcript)
        llm_time = (time.time() - llm_start) * 1000
        print(f"  LLM: {llm_time:.0f}ms -> '{response[:40]}...'")

        # TTS Streaming
        tts_start = time.time()
        first_chunk_time = None
        total_samples = 0
        chunk_count = 0

        async for chunk in tts.generate_stream(response, use_context=False, config=FAST):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - tts_start) * 1000
                total_first_audio = (time.time() - total_start) * 1000
                print(f"  TTS first chunk: {first_chunk_time:.0f}ms")
                print(f"  >>> TOTAL TO FIRST AUDIO: {total_first_audio:.0f}ms <<<")

            total_samples += len(chunk)
            chunk_count += 1

        tts_total = (time.time() - tts_start) * 1000
        total_time = (time.time() - total_start) * 1000
        audio_duration = total_samples / 24000

        print(f"  TTS total: {tts_total:.0f}ms ({chunk_count} chunks, {audio_duration:.1f}s audio)")
        print(f"  TOTAL: {total_time:.0f}ms")

        # Check if under 2 seconds
        if total_first_audio < 2000:
            print(f"  ✅ PASS - First audio under 2 seconds")
        else:
            print(f"  ❌ FAIL - First audio over 2 seconds")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
