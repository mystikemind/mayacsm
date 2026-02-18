#!/usr/bin/env python3
"""
Test FULL pipeline latency with REAL STT.
"""

import sys
import time
import torch
import asyncio
import torchaudio

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')


async def test_with_real_stt():
    """Test full pipeline with real STT."""
    print("=" * 60)
    print("FULL LATENCY TEST (WITH REAL STT)")
    print("=" * 60)

    from maya.engine.stt import STTEngine
    from maya.engine.llm import LLMEngine
    from maya.engine.tts_streaming import StreamingTTSEngine, StreamingConfig

    FAST = StreamingConfig(
        initial_batch_size=3,
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

    # Generate test audio using TTS (simulating user speech)
    print("\nGenerating test audio...")
    test_text = "Hello my name is Pritam"
    test_audio_chunks = []
    async for chunk in tts.generate_stream(test_text, use_context=False, config=FAST):
        test_audio_chunks.append(chunk)
    test_audio = torch.cat(test_audio_chunks)
    print(f"Test audio: {len(test_audio)/24000:.1f}s")

    print("\n" + "=" * 60)
    print("RUNNING FULL PIPELINE TEST")
    print("=" * 60)

    # Run full pipeline
    total_start = time.time()

    # STEP 1: STT
    stt_start = time.time()
    transcript = stt.transcribe(test_audio)
    stt_time = (time.time() - stt_start) * 1000
    print(f"\n[{stt_time:.0f}ms] STT: '{transcript}'")

    # STEP 2: LLM
    llm_start = time.time()
    response = llm.generate(transcript)
    llm_time = (time.time() - llm_start) * 1000
    print(f"[{llm_time:.0f}ms] LLM: '{response}'")

    # STEP 3: TTS Streaming
    tts_start = time.time()
    first_chunk_time = None
    total_samples = 0

    async for chunk in tts.generate_stream(response, use_context=False, config=FAST):
        if first_chunk_time is None:
            first_chunk_time = (time.time() - tts_start) * 1000
            total_to_first = (time.time() - total_start) * 1000
            print(f"[{first_chunk_time:.0f}ms] TTS first chunk")
            print(f"\n>>> TOTAL TO FIRST AUDIO: {total_to_first:.0f}ms <<<")
        total_samples += len(chunk)

    tts_time = (time.time() - tts_start) * 1000
    total_time = (time.time() - total_start) * 1000

    print(f"\n--- RESULTS ---")
    print(f"STT: {stt_time:.0f}ms")
    print(f"LLM: {llm_time:.0f}ms")
    print(f"TTS first chunk: {first_chunk_time:.0f}ms")
    print(f"TTS total: {tts_time:.0f}ms")
    print(f"TOTAL TO FIRST AUDIO: {total_to_first:.0f}ms")
    print(f"TOTAL: {total_time:.0f}ms")

    if total_to_first < 2000:
        print(f"\n✅ PASS - Under 2 seconds!")
    else:
        print(f"\n❌ FAIL - Over 2 seconds")

    return total_to_first


if __name__ == "__main__":
    asyncio.run(test_with_real_stt())
