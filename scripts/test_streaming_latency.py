#!/usr/bin/env python3
"""
Test streaming TTS latency.

Measures time to first audio chunk with true streaming.
Target: ~320ms for first chunk after TTS starts.
"""

import sys
import time
import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def test_streaming_tts():
    """Test streaming TTS latency."""
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    logger.info("=" * 60)
    logger.info("STREAMING TTS LATENCY TEST")
    logger.info("=" * 60)

    # Initialize
    logger.info("Initializing TTS engine...")
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Test phrases
    test_phrases = [
        "hi, how are you",
        "that's a great question, let me think about that",
        "yeah i totally understand what you mean",
    ]

    results = []

    for phrase in test_phrases:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Testing: '{phrase}'")
        logger.info(f"{'=' * 40}")

        start_time = time.time()
        first_chunk_time = None
        total_audio_samples = 0
        chunk_count = 0

        for chunk in tts.generate_stream(phrase, use_context=False):
            chunk_count += 1
            total_audio_samples += len(chunk)

            if first_chunk_time is None:
                first_chunk_time = (time.time() - start_time) * 1000
                logger.info(f">>> FIRST CHUNK: {first_chunk_time:.0f}ms ({len(chunk)/24000*1000:.0f}ms audio)")

        total_time = (time.time() - start_time) * 1000
        total_audio_ms = total_audio_samples / 24000 * 1000

        logger.info(f"Total: {total_time:.0f}ms for {total_audio_ms:.0f}ms audio ({chunk_count} chunks)")

        results.append({
            'phrase': phrase,
            'first_chunk_ms': first_chunk_time,
            'total_time_ms': total_time,
            'audio_ms': total_audio_ms,
            'chunks': chunk_count,
        })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    avg_first_chunk = sum(r['first_chunk_ms'] for r in results) / len(results)

    for r in results:
        logger.info(f"'{r['phrase'][:30]}...' - First chunk: {r['first_chunk_ms']:.0f}ms")

    logger.info(f"\nAverage first chunk latency: {avg_first_chunk:.0f}ms")
    logger.info(f"Target: <500ms to first audio")

    if avg_first_chunk < 500:
        logger.info("PASS - First chunk under 500ms")
    else:
        logger.info("FAIL - First chunk over 500ms")

    return results


def test_full_pipeline_latency():
    """Test full pipeline latency (STT + LLM + streaming TTS)."""
    from maya.engine.stt import STTEngine
    from maya.engine.llm_optimized import OptimizedLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE LATENCY TEST")
    logger.info("=" * 60)

    # Initialize components
    logger.info("Initializing STT...")
    stt = STTEngine()
    stt.initialize()

    logger.info("Initializing LLM...")
    llm = OptimizedLLMEngine()
    llm.initialize()

    logger.info("Initializing streaming TTS...")
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Simulate user audio (pre-recorded or synthetic)
    # For this test, we'll just use direct text
    test_inputs = [
        "how are you doing today",
        "what do you think about artificial intelligence",
        "tell me a joke",
    ]

    results = []

    for user_text in test_inputs:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"User: '{user_text}'")
        logger.info(f"{'=' * 40}")

        pipeline_start = time.time()

        # LLM
        llm_start = time.time()
        response = llm.generate(user_text)
        llm_time = (time.time() - llm_start) * 1000
        logger.info(f"[{llm_time:.0f}ms] LLM: '{response}'")

        # TTS (streaming)
        tts_start = time.time()
        first_chunk_time = None

        for chunk in tts.generate_stream(response, use_context=False):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - tts_start) * 1000
                total_first_audio = (time.time() - pipeline_start) * 1000
                logger.info(f">>> FIRST AUDIO at {total_first_audio:.0f}ms (TTS: {first_chunk_time:.0f}ms)")

        total_time = (time.time() - pipeline_start) * 1000

        results.append({
            'user_text': user_text,
            'response': response,
            'llm_ms': llm_time,
            'tts_first_chunk_ms': first_chunk_time,
            'total_first_audio_ms': llm_time + first_chunk_time,
            'total_time_ms': total_time,
        })

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY (without STT)")
    logger.info("=" * 60)

    avg_first_audio = sum(r['total_first_audio_ms'] for r in results) / len(results)

    for r in results:
        logger.info(f"LLM: {r['llm_ms']:.0f}ms + TTS first: {r['tts_first_chunk_ms']:.0f}ms = {r['total_first_audio_ms']:.0f}ms")

    logger.info(f"\nAverage time to first audio: {avg_first_audio:.0f}ms")
    logger.info(f"Add ~100-150ms for STT")
    logger.info(f"Expected total: {avg_first_audio + 125:.0f}ms")
    logger.info(f"Target: <800ms to first audio")

    if avg_first_audio + 125 < 800:
        logger.info("PASS - Expected total under 800ms")
    else:
        logger.info("NEEDS IMPROVEMENT - Expected total over 800ms")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MAYA STREAMING LATENCY TEST")
    print("=" * 60 + "\n")

    # Test streaming TTS alone
    tts_results = test_streaming_tts()

    # Test full pipeline
    pipeline_results = test_full_pipeline_latency()

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
