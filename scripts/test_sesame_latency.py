#!/usr/bin/env python3
"""
Sesame-Level Latency Test

Tests the full optimized pipeline:
- STT: faster-whisper in Docker (~60ms)
- LLM: vLLM in Docker (~80ms)
- TTS: CSM with 2-frame first chunk (~140ms)

Target: ~280-320ms total
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def create_test_audio(duration_sec: float = 2.0, sample_rate: int = 24000) -> torch.Tensor:
    """Create test audio that sounds like speech."""
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples)

    # Mix frequencies to simulate speech
    audio = np.zeros(samples, dtype=np.float32)
    for freq in [200, 400, 600, 800]:
        audio += 0.1 * np.sin(2 * np.pi * freq * t)

    # Add envelope
    envelope = np.exp(-0.5 * ((t - duration_sec/2) / (duration_sec/4))**2)
    audio = audio * envelope

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.3

    return torch.tensor(audio, dtype=torch.float32)


def test_stt_latency():
    """Test faster-whisper Docker latency."""
    from maya.engine.stt_fast import FastSTTEngine

    logger.info("\n" + "="*60)
    logger.info("STT LATENCY TEST (faster-whisper Docker)")
    logger.info("="*60)

    stt = FastSTTEngine()
    stt.initialize()

    test_audio = create_test_audio(2.0)

    times = []
    for i in range(5):
        start = time.time()
        result = stt.transcribe(test_audio)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        logger.info(f"  Run {i+1}: {elapsed:.0f}ms -> '{result}'")

    avg = np.mean(times[1:])  # Skip first (might be cold)
    logger.info(f"\nSTT Average: {avg:.0f}ms (target: ~60ms)")
    return avg


def test_llm_latency():
    """Test vLLM Docker latency."""
    from maya.engine.llm_vllm import VLLMEngine

    logger.info("\n" + "="*60)
    logger.info("LLM LATENCY TEST (vLLM Docker)")
    logger.info("="*60)

    llm = VLLMEngine()
    llm.initialize()

    prompts = [
        "hello how are you",
        "what is your name",
        "tell me a joke",
        "how is the weather",
        "what do you like to do"
    ]

    times = []
    for i, prompt in enumerate(prompts):
        start = time.time()
        response = llm.generate(prompt)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        logger.info(f"  Run {i+1}: {elapsed:.0f}ms -> '{response}'")

    avg = np.mean(times[1:])  # Skip first
    logger.info(f"\nLLM Average: {avg:.0f}ms (target: ~80ms)")
    return avg


def test_tts_latency():
    """Test streaming TTS first chunk latency."""
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    logger.info("\n" + "="*60)
    logger.info("TTS LATENCY TEST (Streaming CSM)")
    logger.info("="*60)

    tts = RealStreamingTTSEngine()
    tts.initialize()

    phrases = [
        "hi there",
        "im doing great thanks",
        "thats really interesting",
        "tell me more about that",
        "i think thats a good idea"
    ]

    first_chunk_times = []

    for i, phrase in enumerate(phrases):
        start = time.time()
        first_chunk_time = None
        chunks = []

        for chunk in tts.generate_stream(phrase, use_context=False):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - start) * 1000
            chunks.append(chunk)

        total_time = (time.time() - start) * 1000
        total_samples = sum(len(c) for c in chunks)
        audio_duration = total_samples / 24000 * 1000

        first_chunk_times.append(first_chunk_time)
        logger.info(f"  Run {i+1}: First chunk {first_chunk_time:.0f}ms, Total {total_time:.0f}ms, Audio {audio_duration:.0f}ms -> '{phrase}'")

    avg = np.mean(first_chunk_times[1:])  # Skip first
    logger.info(f"\nTTS First Chunk Average: {avg:.0f}ms (target: ~140ms)")
    return avg


def test_full_pipeline():
    """Test full pipeline: STT + LLM + TTS first chunk."""
    from maya.engine.stt_fast import FastSTTEngine
    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    logger.info("\n" + "="*60)
    logger.info("FULL PIPELINE TEST (STT + LLM + TTS)")
    logger.info("="*60)

    # Initialize all engines
    logger.info("Initializing engines...")
    stt = FastSTTEngine()
    stt.initialize()

    llm = VLLMEngine()
    llm.initialize()

    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Create test audio
    test_audio = create_test_audio(2.0)

    times = []
    for i in range(5):
        logger.info(f"\n--- Pipeline Run {i+1} ---")

        pipeline_start = time.time()

        # STT
        stt_start = time.time()
        transcript = stt.transcribe(test_audio)
        stt_time = (time.time() - stt_start) * 1000

        # Use a fixed transcript for consistent testing
        transcript = "hello how are you today"
        logger.info(f"  STT: {stt_time:.0f}ms -> '{transcript}'")

        # LLM
        llm_start = time.time()
        response = llm.generate(transcript)
        llm_time = (time.time() - llm_start) * 1000
        logger.info(f"  LLM: {llm_time:.0f}ms -> '{response}'")

        # TTS (first chunk only)
        tts_start = time.time()
        first_chunk_time = None
        for chunk in tts.generate_stream(response, use_context=False):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - tts_start) * 1000
                break  # Only need first chunk

        total_time = (time.time() - pipeline_start) * 1000
        times.append({
            'stt': stt_time,
            'llm': llm_time,
            'tts': first_chunk_time,
            'total': total_time
        })

        logger.info(f"  TTS First Chunk: {first_chunk_time:.0f}ms")
        logger.info(f"  >>> TOTAL: {total_time:.0f}ms <<<")

    # Calculate averages (skip first run)
    avg_stt = np.mean([t['stt'] for t in times[1:]])
    avg_llm = np.mean([t['llm'] for t in times[1:]])
    avg_tts = np.mean([t['tts'] for t in times[1:]])
    avg_total = np.mean([t['total'] for t in times[1:]])

    return {
        'stt': avg_stt,
        'llm': avg_llm,
        'tts': avg_tts,
        'total': avg_total
    }


def main():
    logger.info("\n" + "="*70)
    logger.info("  SESAME-LEVEL LATENCY VERIFICATION")
    logger.info("  Target: ~280-320ms to first audio")
    logger.info("="*70)

    # Test individual components
    stt_avg = test_stt_latency()
    llm_avg = test_llm_latency()
    tts_avg = test_tts_latency()

    # Test full pipeline
    pipeline_results = test_full_pipeline()

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("  FINAL RESULTS - SESAME LEVEL COMPARISON")
    logger.info("="*70)
    logger.info("")
    logger.info("Individual Component Averages:")
    logger.info(f"  STT (faster-whisper Docker): {stt_avg:.0f}ms   (target: ~60ms)")
    logger.info(f"  LLM (vLLM Docker):           {llm_avg:.0f}ms   (target: ~80ms)")
    logger.info(f"  TTS First Chunk (CSM):       {tts_avg:.0f}ms  (target: ~140ms)")
    logger.info("")
    logger.info("Full Pipeline Averages:")
    logger.info(f"  STT: {pipeline_results['stt']:.0f}ms")
    logger.info(f"  LLM: {pipeline_results['llm']:.0f}ms")
    logger.info(f"  TTS: {pipeline_results['tts']:.0f}ms")
    logger.info(f"  ─────────────────────")
    logger.info(f"  TOTAL: {pipeline_results['total']:.0f}ms")
    logger.info("")

    # Compare with Sesame
    sesame_target = 280
    our_total = pipeline_results['total']
    gap = our_total - sesame_target

    if our_total <= 320:
        logger.info("  ✅ SESAME LEVEL ACHIEVED!")
        logger.info(f"  We are within target range (280-320ms)")
    elif our_total <= 400:
        logger.info(f"  ⚠️  Close to Sesame level ({gap:.0f}ms over target)")
    else:
        logger.info(f"  ❌ Not yet at Sesame level ({gap:.0f}ms over target)")

    logger.info("")
    logger.info("  Sesame Maya Reference: ~200-300ms")
    logger.info(f"  Our Implementation:    ~{our_total:.0f}ms")
    logger.info("")
    logger.info("="*70)

    return pipeline_results


if __name__ == "__main__":
    main()
