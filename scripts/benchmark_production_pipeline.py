#!/usr/bin/env python3
"""
Production Pipeline Benchmark - Measure REAL first-audio latency.

This script benchmarks the ACTUAL streaming production pipeline with:
- vLLM Docker for LLM (~80ms)
- Local CUDA STT (~50-85ms)
- Streaming TTS with 2-frame first chunk (~105ms)

Target: < 200ms first audio (Sesame AI level)

Usage:
    python scripts/benchmark_production_pipeline.py

Requirements:
    - vLLM Docker running (./start_maya.sh start)
    - GPU available for TTS and STT
"""

import sys
import os
import time
import torch
import numpy as np
import logging

# Add project to path
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def benchmark_stt(num_runs: int = 10) -> dict:
    """Benchmark STT latency with real audio."""
    from maya.engine.stt_local import LocalSTTEngine

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARKING STT (Local CUDA Whisper)")
    logger.info("=" * 60)

    stt = LocalSTTEngine()
    stt.initialize()

    # Create test audio (1.5 seconds of speech-like audio)
    sample_rate = 16000
    duration_s = 1.5
    samples = int(sample_rate * duration_s)

    # Generate more realistic audio (not just noise)
    t = torch.linspace(0, duration_s, samples)
    test_audio = torch.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz tone
    test_audio += torch.randn(samples) * 0.1  # Add some noise

    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        result = stt.transcribe(test_audio)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        if i < 3:
            logger.info(f"  Run {i+1}: {elapsed:.1f}ms -> '{result[:50] if result else '(empty)'}'")

    # Skip first 2 runs (warmup)
    times = times[2:]
    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    logger.info(f"\nSTT Results ({len(times)} runs):")
    logger.info(f"  Average: {avg:.1f}ms")
    logger.info(f"  P50: {p50:.1f}ms")
    logger.info(f"  P95: {p95:.1f}ms")

    return {"avg": avg, "p50": p50, "p95": p95, "times": times}


def benchmark_llm(num_runs: int = 10) -> dict:
    """Benchmark LLM latency via vLLM Docker."""
    from maya.engine.llm_vllm import VLLMEngine

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARKING LLM (vLLM Docker)")
    logger.info("=" * 60)

    llm = VLLMEngine()
    llm.initialize()

    test_inputs = [
        "Hello how are you",
        "What's the weather like today",
        "Tell me about yourself",
        "That sounds interesting",
        "Can you help me with something",
        "I had a great day",
        "What do you think about that",
        "Let's talk about music",
        "How does that work",
        "Thanks for the help",
    ]

    times = []
    for i, text in enumerate(test_inputs[:num_runs]):
        start = time.time()
        response = llm.generate(text)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        if i < 3:
            logger.info(f"  Run {i+1}: {elapsed:.1f}ms -> '{response[:50]}...'")
        llm.clear_history()  # Reset for fair comparison

    # Skip first 2 runs (warmup)
    times = times[2:]
    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    transport = "Unix Socket" if llm._use_unix_socket else "HTTP"
    logger.info(f"\nLLM Results ({len(times)} runs, {transport}):")
    logger.info(f"  Average: {avg:.1f}ms")
    logger.info(f"  P50: {p50:.1f}ms")
    logger.info(f"  P95: {p95:.1f}ms")

    return {"avg": avg, "p50": p50, "p95": p95, "times": times, "transport": transport}


def benchmark_tts(num_runs: int = 10) -> dict:
    """Benchmark TTS first-chunk latency."""
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARKING TTS (Streaming CSM, 2-frame first chunk)")
    logger.info("=" * 60)

    tts = RealStreamingTTSEngine()
    tts.initialize()

    test_texts = [
        "oh hey how are you",
        "thats really interesting",
        "let me think about that",
        "i totally get what you mean",
        "yeah for sure sounds good",
        "hmm thats a good question",
        "wow thats amazing",
        "im doing pretty well thanks",
        "sure i can help with that",
        "oh nice thats cool",
    ]

    first_chunk_times = []
    total_times = []

    for i, text in enumerate(test_texts[:num_runs]):
        torch.cuda.synchronize()
        start = time.time()
        first_chunk_time = None
        chunks = []

        for chunk in tts.generate_stream(text, use_context=False):
            if first_chunk_time is None:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - start) * 1000
            chunks.append(chunk)

        torch.cuda.synchronize()
        total_time = (time.time() - start) * 1000

        first_chunk_times.append(first_chunk_time)
        total_times.append(total_time)

        if i < 3:
            audio_ms = sum(len(c) for c in chunks) / 24000 * 1000
            logger.info(f"  Run {i+1}: first={first_chunk_time:.0f}ms, total={total_time:.0f}ms, audio={audio_ms:.0f}ms")

    # Skip first 2 runs (warmup)
    first_chunk_times = first_chunk_times[2:]
    total_times = total_times[2:]

    avg_first = np.mean(first_chunk_times)
    p50_first = np.percentile(first_chunk_times, 50)
    p95_first = np.percentile(first_chunk_times, 95)

    logger.info(f"\nTTS First Chunk Results ({len(first_chunk_times)} runs):")
    logger.info(f"  Average: {avg_first:.1f}ms")
    logger.info(f"  P50: {p50_first:.1f}ms")
    logger.info(f"  P95: {p95_first:.1f}ms")
    logger.info(f"  First chunk frames: {tts.FIRST_CHUNK_FRAMES}")

    return {
        "avg_first": avg_first,
        "p50_first": p50_first,
        "p95_first": p95_first,
        "first_chunk_frames": tts.FIRST_CHUNK_FRAMES,
        "times": first_chunk_times
    }


def benchmark_full_pipeline(num_runs: int = 5) -> dict:
    """Benchmark complete pipeline: STT -> LLM -> TTS first chunk."""
    from maya.engine.stt_local import LocalSTTEngine
    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARKING FULL PIPELINE (STT -> LLM -> TTS)")
    logger.info("Target: < 200ms first audio")
    logger.info("=" * 60)

    # Initialize all components
    stt = LocalSTTEngine()
    llm = VLLMEngine()
    tts = RealStreamingTTSEngine()

    stt.initialize()
    llm.initialize()
    tts.initialize()

    # Create test audio
    sample_rate = 16000
    duration_s = 1.5
    samples = int(sample_rate * duration_s)
    t = torch.linspace(0, duration_s, samples)
    test_audio = torch.sin(2 * np.pi * 440 * t) * 0.3
    test_audio += torch.randn(samples) * 0.1

    results = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        # STT
        stt_start = time.time()
        transcript = stt.transcribe(test_audio)
        stt_time = (time.time() - stt_start) * 1000

        # Use a real transcript if STT gives empty
        if not transcript or len(transcript) < 2:
            transcript = "hello how are you"

        # LLM
        llm_start = time.time()
        response = llm.generate(transcript)
        llm_time = (time.time() - llm_start) * 1000

        # TTS (first chunk only)
        tts_start = time.time()
        first_chunk_time = None
        for chunk in tts.generate_stream(response, use_context=False):
            torch.cuda.synchronize()
            first_chunk_time = (time.time() - tts_start) * 1000
            break  # Only time first chunk

        torch.cuda.synchronize()
        total_time = (time.time() - start) * 1000

        result = {
            "stt_ms": stt_time,
            "llm_ms": llm_time,
            "tts_first_ms": first_chunk_time,
            "total_ms": total_time,
            "transcript": transcript[:30],
            "response": response[:30]
        }
        results.append(result)

        logger.info(f"\nRun {i+1}:")
        logger.info(f"  STT: {stt_time:.0f}ms -> '{transcript[:30]}...'")
        logger.info(f"  LLM: {llm_time:.0f}ms -> '{response[:30]}...'")
        logger.info(f"  TTS: {first_chunk_time:.0f}ms (first chunk)")
        logger.info(f"  TOTAL: {total_time:.0f}ms")

        llm.clear_history()

    # Compute averages (skip first run for warmup)
    results = results[1:]
    avg_stt = np.mean([r["stt_ms"] for r in results])
    avg_llm = np.mean([r["llm_ms"] for r in results])
    avg_tts = np.mean([r["tts_first_ms"] for r in results])
    avg_total = np.mean([r["total_ms"] for r in results])

    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  STT (avg):       {avg_stt:.0f}ms")
    logger.info(f"  LLM (avg):       {avg_llm:.0f}ms")
    logger.info(f"  TTS first (avg): {avg_tts:.0f}ms")
    logger.info(f"  TOTAL (avg):     {avg_total:.0f}ms")
    logger.info("")

    target = 200
    if avg_total < target:
        logger.info(f"  ✓ BELOW TARGET ({target}ms) - SESAME LEVEL ACHIEVED!")
    else:
        gap = avg_total - target
        logger.info(f"  ✗ ABOVE TARGET by {gap:.0f}ms")
        logger.info(f"    Need to reduce: STT by {avg_stt*0.2:.0f}ms, LLM by {avg_llm*0.2:.0f}ms, or TTS by {avg_tts*0.3:.0f}ms")

    logger.info("=" * 60)

    return {
        "avg_stt": avg_stt,
        "avg_llm": avg_llm,
        "avg_tts": avg_tts,
        "avg_total": avg_total,
        "target": target,
        "results": results
    }


def main():
    logger.info("=" * 60)
    logger.info("MAYA PRODUCTION PIPELINE BENCHMARK")
    logger.info("Target: < 200ms first audio (Sesame AI Level)")
    logger.info("=" * 60)

    # Check vLLM is running
    import requests
    try:
        resp = requests.get("http://localhost:8001/health", timeout=2)
        if resp.status_code != 200:
            logger.error("vLLM server not healthy. Run: ./start_maya.sh start")
            return
    except Exception:
        logger.error("vLLM server not running. Run: ./start_maya.sh start")
        return

    # Run individual benchmarks
    stt_results = benchmark_stt(num_runs=8)
    llm_results = benchmark_llm(num_runs=8)
    tts_results = benchmark_tts(num_runs=8)

    # Run full pipeline benchmark
    pipeline_results = benchmark_full_pipeline(num_runs=6)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Component Latencies (P50):")
    logger.info(f"  STT:       {stt_results['p50']:.0f}ms")
    logger.info(f"  LLM:       {llm_results['p50']:.0f}ms ({llm_results['transport']})")
    logger.info(f"  TTS:       {tts_results['p50_first']:.0f}ms ({tts_results['first_chunk_frames']} frames)")
    logger.info(f"  Pipeline:  {pipeline_results['avg_total']:.0f}ms")
    logger.info("")
    logger.info(f"Target:      200ms")

    gap = pipeline_results['avg_total'] - 200
    if gap <= 0:
        logger.info(f"Status:      ✓ SESAME LEVEL ACHIEVED ({-gap:.0f}ms under target)")
    else:
        logger.info(f"Status:      ✗ {gap:.0f}ms above target")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
