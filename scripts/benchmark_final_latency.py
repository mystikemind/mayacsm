#!/usr/bin/env python3
"""
Final Latency Benchmark - Measure actual production latency.

This script measures the real-world latency of the Maya pipeline
after all warmup and initialization is complete.

Measures:
1. STT latency (via Docker faster-whisper)
2. LLM latency (via vLLM Docker)
3. TTS first chunk latency (local CSM)
4. Total pipeline latency

All measurements are done AFTER warmup to show steady-state performance.
"""

import sys
import os
import time
import torch
import numpy as np
import logging

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def benchmark_stt():
    """Benchmark STT latency."""
    logger.info("\n" + "=" * 60)
    logger.info("STT BENCHMARK (Docker faster-whisper)")
    logger.info("=" * 60)

    from maya.engine.stt_true_streaming import TrueStreamingSTTEngine
    from maya.config import AUDIO

    stt = TrueStreamingSTTEngine()
    stt.initialize()

    # Generate 1.5s test audio
    sample_rate = AUDIO.sample_rate
    duration = 1.5
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * 440 * t) * 0.3 + torch.randn(len(t)) * 0.1

    # Warmup (5 runs)
    logger.info("Warming up...")
    for _ in range(5):
        stt.reset()
        chunk_samples = int(sample_rate * 0.2)
        for i in range(0, len(audio), chunk_samples):
            stt.add_audio(audio[i:i+chunk_samples])
        stt.finalize()

    # Benchmark (10 runs)
    logger.info("Benchmarking...")
    times = []
    for run in range(10):
        stt.reset()
        chunk_samples = int(sample_rate * 0.2)

        start = time.time()
        for i in range(0, len(audio), chunk_samples):
            stt.add_audio(audio[i:i+chunk_samples])
        result = stt.finalize()
        elapsed = (time.time() - start) * 1000

        times.append(elapsed)
        logger.info(f"  Run {run+1}: {elapsed:.0f}ms - '{result.text[:30]}...'")

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    logger.info(f"\nSTT Results:")
    logger.info(f"  Average: {avg:.0f}ms")
    logger.info(f"  P50: {p50:.0f}ms")
    logger.info(f"  P95: {p95:.0f}ms")

    return avg


def benchmark_llm():
    """Benchmark LLM latency."""
    logger.info("\n" + "=" * 60)
    logger.info("LLM BENCHMARK (vLLM Docker)")
    logger.info("=" * 60)

    from maya.engine.llm_vllm import VLLMEngine

    llm = VLLMEngine()
    llm.initialize()

    prompts = [
        "Hello",
        "How are you doing today",
        "Tell me about yourself",
        "What's your favorite color",
        "Nice to meet you",
    ]

    # Warmup (5 runs)
    logger.info("Warming up...")
    for _ in range(5):
        llm.generate("warmup")
        llm.clear_history()

    # Benchmark (10 runs)
    logger.info("Benchmarking...")
    times = []
    for run, prompt in enumerate(prompts * 2):
        start = time.time()
        response = llm.generate(prompt)
        elapsed = (time.time() - start) * 1000

        times.append(elapsed)
        logger.info(f"  Run {run+1}: {elapsed:.0f}ms - '{response[:30]}...'")
        llm.clear_history()

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    logger.info(f"\nLLM Results:")
    logger.info(f"  Average: {avg:.0f}ms")
    logger.info(f"  P50: {p50:.0f}ms")
    logger.info(f"  P95: {p95:.0f}ms")

    return avg


def benchmark_tts():
    """Benchmark TTS first chunk latency."""
    logger.info("\n" + "=" * 60)
    logger.info("TTS BENCHMARK (CSM Streaming)")
    logger.info("=" * 60)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    tts = RealStreamingTTSEngine()
    tts.initialize()

    phrases = [
        "oh hey there",
        "thats interesting",
        "let me think about that",
        "sure no problem",
        "sounds good to me",
    ]

    # Warmup already done in initialize()

    # Benchmark (10 runs)
    logger.info("Benchmarking first chunk latency...")
    times = []
    for run, phrase in enumerate(phrases * 2):
        torch.cuda.synchronize()
        start = time.time()

        for chunk in tts.generate_stream(phrase, use_context=False):
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            break  # Only measure first chunk

        times.append(elapsed)
        logger.info(f"  Run {run+1}: {elapsed:.0f}ms first chunk for '{phrase}'")

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    logger.info(f"\nTTS First Chunk Results:")
    logger.info(f"  Average: {avg:.0f}ms")
    logger.info(f"  P50: {p50:.0f}ms")
    logger.info(f"  P95: {p95:.0f}ms")

    return avg


def benchmark_full_pipeline():
    """Benchmark full pipeline latency."""
    logger.info("\n" + "=" * 60)
    logger.info("FULL PIPELINE BENCHMARK")
    logger.info("=" * 60)

    from maya.engine.stt_true_streaming import TrueStreamingSTTEngine
    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    from maya.config import AUDIO

    stt = TrueStreamingSTTEngine()
    llm = VLLMEngine()
    tts = RealStreamingTTSEngine()

    stt.initialize()
    llm.initialize()
    tts.initialize()

    # Generate test audio
    sample_rate = AUDIO.sample_rate
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * 440 * t) * 0.3 + torch.randn(len(t)) * 0.1

    # Warmup (5 runs)
    logger.info("Warming up full pipeline...")
    for _ in range(5):
        stt.reset()
        chunk_samples = int(sample_rate * 0.2)
        for i in range(0, len(audio), chunk_samples):
            stt.add_audio(audio[i:i+chunk_samples])
        result = stt.finalize()
        transcript = result.text if result.text else "hello"

        response = llm.generate(transcript)
        llm.clear_history()

        for chunk in tts.generate_stream(response, use_context=False):
            break

    # Benchmark (10 runs)
    logger.info("Benchmarking full pipeline...")
    times = []
    component_times = {'stt': [], 'llm': [], 'tts': []}

    for run in range(10):
        # STT
        stt.reset()
        chunk_samples = int(sample_rate * 0.2)
        for i in range(0, len(audio), chunk_samples):
            stt.add_audio(audio[i:i+chunk_samples])

        torch.cuda.synchronize()
        pipeline_start = time.time()

        stt_start = time.time()
        result = stt.finalize()
        transcript = result.text if result.text else "hello"
        stt_time = (time.time() - stt_start) * 1000
        component_times['stt'].append(stt_time)

        # LLM
        llm_start = time.time()
        response = llm.generate(transcript)
        llm_time = (time.time() - llm_start) * 1000
        component_times['llm'].append(llm_time)
        llm.clear_history()

        # TTS first chunk
        tts_start = time.time()
        for chunk in tts.generate_stream(response, use_context=False):
            torch.cuda.synchronize()
            tts_time = (time.time() - tts_start) * 1000
            break
        component_times['tts'].append(tts_time)

        total = (time.time() - pipeline_start) * 1000
        times.append(total)

        logger.info(f"  Run {run+1}: STT={stt_time:.0f}ms LLM={llm_time:.0f}ms TTS={tts_time:.0f}ms TOTAL={total:.0f}ms")

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    logger.info(f"\nFull Pipeline Results:")
    logger.info(f"  Average: {avg:.0f}ms")
    logger.info(f"  P50: {p50:.0f}ms")
    logger.info(f"  P95: {p95:.0f}ms")

    logger.info(f"\nComponent Breakdown (averages):")
    for component, times_list in component_times.items():
        logger.info(f"  {component.upper()}: {np.mean(times_list):.0f}ms")

    return avg


def main():
    logger.info("=" * 60)
    logger.info("MAYA FINAL LATENCY BENCHMARK")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This benchmark measures steady-state latency AFTER warmup.")
    logger.info("Results represent real-world production performance.")
    logger.info("")

    stt_latency = benchmark_stt()
    llm_latency = benchmark_llm()
    tts_latency = benchmark_tts()
    pipeline_latency = benchmark_full_pipeline()

    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Component Latencies (after warmup):")
    logger.info(f"  STT (Docker):    {stt_latency:.0f}ms")
    logger.info(f"  LLM (vLLM):      {llm_latency:.0f}ms")
    logger.info(f"  TTS (CSM):       {tts_latency:.0f}ms")
    logger.info(f"  Full Pipeline:   {pipeline_latency:.0f}ms")
    logger.info("")

    # Calculate theoretical minimum
    theoretical_min = llm_latency + tts_latency  # STT can overlap with user speech
    logger.info(f"Theoretical minimum (LLM + TTS): {theoretical_min:.0f}ms")
    logger.info("")

    # Compare to Sesame target
    if pipeline_latency < 300:
        logger.info("STATUS: EXCELLENT - Near Sesame-level latency!")
    elif pipeline_latency < 400:
        logger.info("STATUS: GOOD - Competitive latency")
    elif pipeline_latency < 500:
        logger.info("STATUS: ACCEPTABLE - Production-ready")
    else:
        logger.info("STATUS: NEEDS IMPROVEMENT")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
