#!/usr/bin/env python3
"""
Test TTS first-chunk latency variance.

Measures consistency of first-chunk timing after warmup.
Target: < 150ms with < 50ms variance (p50 to p90)
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_tts_variance():
    """Test TTS first-chunk latency over multiple runs."""
    logger.info("=" * 60)
    logger.info("TTS FIRST-CHUNK LATENCY VARIANCE TEST")
    logger.info("=" * 60)

    # Load TTS
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Test phrases - mix of lengths
    test_phrases = [
        "Hello, how can I help you today?",
        "Sure, I understand.",
        "That's a great question.",
        "Let me think about that for a moment.",
        "I'm happy to help you with that.",
    ]

    # Run multiple tests
    NUM_RUNS = 10
    times = []

    logger.info(f"\nRunning {NUM_RUNS} generation tests...")
    logger.info("-" * 60)

    for i in range(NUM_RUNS):
        phrase = test_phrases[i % len(test_phrases)]

        torch.cuda.synchronize()
        start = time.time()

        first_chunk_time = None
        for chunk in tts.generate_stream(phrase):
            if first_chunk_time is None:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - start) * 1000
                break

        times.append(first_chunk_time)
        logger.info(f"  Run {i+1:2d}: {first_chunk_time:6.0f}ms - '{phrase[:30]}...'")

    # Statistics
    logger.info("-" * 60)
    logger.info("STATISTICS")
    logger.info("-" * 60)

    times_arr = np.array(times)
    logger.info(f"  Min:     {np.min(times_arr):.0f}ms")
    logger.info(f"  Max:     {np.max(times_arr):.0f}ms")
    logger.info(f"  Mean:    {np.mean(times_arr):.0f}ms")
    logger.info(f"  Median:  {np.median(times_arr):.0f}ms")
    logger.info(f"  P10:     {np.percentile(times_arr, 10):.0f}ms")
    logger.info(f"  P90:     {np.percentile(times_arr, 90):.0f}ms")
    logger.info(f"  Std Dev: {np.std(times_arr):.0f}ms")

    variance = np.percentile(times_arr, 90) - np.percentile(times_arr, 10)
    logger.info(f"  P10-P90 Variance: {variance:.0f}ms")

    # Assessment
    logger.info("-" * 60)

    median = np.median(times_arr)
    if median < 150 and variance < 50:
        logger.info(f"✓ EXCELLENT: {median:.0f}ms median, {variance:.0f}ms variance")
    elif median < 200 and variance < 100:
        logger.info(f"△ GOOD: {median:.0f}ms median, {variance:.0f}ms variance")
    else:
        logger.info(f"○ NEEDS WORK: {median:.0f}ms median, {variance:.0f}ms variance")

    # Sesame comparison
    logger.info(f"\nSesame target: ~100ms first chunk")
    logger.info(f"Gap: {median - 100:.0f}ms")


if __name__ == "__main__":
    test_tts_variance()
