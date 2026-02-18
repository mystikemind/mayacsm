#!/usr/bin/env python3
"""Test local faster-whisper STT latency vs Docker."""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_test_audio(duration_sec: float = 1.5) -> torch.Tensor:
    """Create test audio that sounds like speech."""
    sample_rate = 24000
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples)

    # Create speech-like audio with formants
    audio = np.zeros(samples, dtype=np.float32)
    for freq in [200, 400, 600, 800]:  # Formant frequencies
        audio += 0.08 * np.sin(2 * np.pi * freq * t)

    # Add envelope
    envelope = np.exp(-0.5 * ((t - duration_sec/2) / (duration_sec/4))**2)
    audio = audio * envelope * 0.3

    return torch.tensor(audio, dtype=torch.float32)


def main():
    logger.info("="*60)
    logger.info("LOCAL vs DOCKER STT LATENCY TEST")
    logger.info("="*60)

    test_audio = create_test_audio(1.5)
    logger.info(f"Test audio: {len(test_audio)} samples ({len(test_audio)/24000:.1f}s)")

    # Test local STT
    logger.info("\n--- LOCAL FASTER-WHISPER ---")
    try:
        from maya.engine.stt_local import LocalSTTEngine
        local_stt = LocalSTTEngine()
        local_stt.initialize()

        local_times = []
        for i in range(5):
            start = time.time()
            result = local_stt.transcribe(test_audio)
            elapsed = (time.time() - start) * 1000
            local_times.append(elapsed)
            logger.info(f"  Run {i+1}: {elapsed:.0f}ms -> '{result}'")

        local_avg = np.mean(local_times[1:])  # Skip first (warmup)
        logger.info(f"\nLocal STT Average: {local_avg:.0f}ms")

    except Exception as e:
        logger.error(f"Local STT Error: {e}")
        import traceback
        traceback.print_exc()
        local_avg = float('inf')

    # Test Docker STT for comparison
    logger.info("\n--- DOCKER FASTER-WHISPER ---")
    try:
        from maya.engine.stt_fast import FastSTTEngine
        docker_stt = FastSTTEngine()
        docker_stt.initialize()

        docker_times = []
        for i in range(5):
            start = time.time()
            result = docker_stt.transcribe(test_audio)
            elapsed = (time.time() - start) * 1000
            docker_times.append(elapsed)
            logger.info(f"  Run {i+1}: {elapsed:.0f}ms -> '{result}'")

        docker_avg = np.mean(docker_times[1:])
        logger.info(f"\nDocker STT Average: {docker_avg:.0f}ms")

    except Exception as e:
        logger.error(f"Docker STT Error: {e}")
        docker_avg = float('inf')

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    if local_avg != float('inf'):
        logger.info(f"Local STT:  {local_avg:.0f}ms")
    if docker_avg != float('inf'):
        logger.info(f"Docker STT: {docker_avg:.0f}ms")
    if local_avg != float('inf') and docker_avg != float('inf'):
        improvement = docker_avg - local_avg
        logger.info(f"Improvement: {improvement:.0f}ms ({improvement/docker_avg*100:.0f}%)")

    logger.info("\nTarget: < 100ms")


if __name__ == "__main__":
    main()
