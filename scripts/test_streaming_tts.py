#!/usr/bin/env python3
"""
Test the streaming TTS engine.

This tests the lightning-fast approach:
- Stream audio chunks as they're generated
- Measure time to first chunk
"""

import torch
import torchaudio
import asyncio
import time
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

from maya.engine.tts_streaming import StreamingTTSEngine


async def test_streaming():
    """Test streaming TTS generation."""
    print("=" * 60)
    print("TESTING STREAMING TTS ENGINE")
    print("=" * 60)

    # Initialize
    print("\nInitializing streaming TTS...")
    engine = StreamingTTSEngine()
    engine.initialize()

    # Test phrases
    test_phrases = [
        "Hello! I'm Maya.",
        "That's a really interesting question. Let me think about that.",
        "I understand how you feel. It can be really difficult sometimes, but I'm here to help you through it.",
    ]

    for phrase in test_phrases:
        print(f"\n{'='*60}")
        print(f"Testing: '{phrase[:50]}...'")
        print("=" * 60)

        start = time.time()
        first_chunk_time = None
        chunks = []
        chunk_times = []

        async for chunk in engine.generate_stream(phrase, use_context=False):
            chunk_time = time.time() - start
            if first_chunk_time is None:
                first_chunk_time = chunk_time
                print(f"\n>>> FIRST CHUNK in {first_chunk_time*1000:.0f}ms! <<<\n")

            chunks.append(chunk)
            chunk_times.append(chunk_time)
            chunk_duration = len(chunk) / 24000
            print(f"  Chunk {len(chunks)}: {len(chunk)} samples ({chunk_duration:.2f}s) at {chunk_time*1000:.0f}ms")

        total_time = time.time() - start
        full_audio = torch.cat(chunks) if chunks else torch.zeros(24000)
        audio_duration = len(full_audio) / 24000

        print(f"\nResults:")
        print(f"  First chunk: {first_chunk_time*1000:.0f}ms")
        print(f"  Total time: {total_time*1000:.0f}ms")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Chunks: {len(chunks)}")
        print(f"  RTF: {total_time/audio_duration:.2f}x")

        # Save audio
        output_path = f"/home/ec2-user/SageMaker/project_maya/tests/outputs/streaming_test_{len(test_phrases)}.wav"
        torchaudio.save(output_path, full_audio.unsqueeze(0).cpu(), 24000)
        print(f"  Saved: {output_path}")

    print("\n" + "=" * 60)
    print("STREAMING TTS TEST COMPLETE")
    print("=" * 60)


async def test_comparison():
    """Compare streaming vs non-streaming."""
    print("\n" + "=" * 60)
    print("COMPARING STREAMING VS NON-STREAMING")
    print("=" * 60)

    from maya.engine.tts import TTSEngine

    phrase = "Hello! I'm Maya. How can I help you today?"

    # Streaming
    print("\n[STREAMING]")
    streaming_engine = StreamingTTSEngine()
    streaming_engine.initialize()

    start = time.time()
    first_chunk_time = None
    chunks = []

    async for chunk in streaming_engine.generate_stream(phrase, use_context=False):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
        chunks.append(chunk)

    streaming_total = time.time() - start
    streaming_audio = torch.cat(chunks) if chunks else torch.zeros(24000)

    print(f"  First chunk: {first_chunk_time*1000:.0f}ms")
    print(f"  Total time: {streaming_total*1000:.0f}ms")

    # Non-streaming
    print("\n[NON-STREAMING]")
    standard_engine = TTSEngine()
    standard_engine.initialize()

    start = time.time()
    standard_audio = standard_engine.generate(phrase, use_context=False)
    standard_total = time.time() - start

    print(f"  Total time: {standard_total*1000:.0f}ms (no first chunk)")

    # Compare
    print(f"\nIMPROVEMENT:")
    print(f"  First audio: {first_chunk_time*1000:.0f}ms (streaming) vs {standard_total*1000:.0f}ms (standard)")
    print(f"  Speedup to first audio: {standard_total/first_chunk_time:.1f}x faster!")


if __name__ == "__main__":
    asyncio.run(test_streaming())
    # Uncomment to compare:
    # asyncio.run(test_comparison())
