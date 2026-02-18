#!/usr/bin/env python3
"""Test local Whisper STT latency."""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_test_audio(duration_sec: float = 1.5) -> torch.Tensor:
    """Create test audio."""
    sample_rate = 24000
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples)
    audio = np.zeros(samples, dtype=np.float32)
    for freq in [200, 400, 600]:
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-0.5 * ((t - duration_sec/2) / (duration_sec/4))**2)
    audio = audio * envelope * 0.3
    return torch.tensor(audio, dtype=torch.float32)


def main():
    logger.info("="*60)
    logger.info("LOCAL WHISPER STT TEST")
    logger.info("="*60)

    try:
        from maya.engine.stt import STTEngine
        stt = STTEngine()
        stt.initialize()

        test_audio = create_test_audio(1.5)

        times = []
        for i in range(5):
            start = time.time()
            result = stt.transcribe(test_audio)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            logger.info(f"  Run {i+1}: {elapsed:.0f}ms -> '{result}'")

        avg = np.mean(times[1:])
        logger.info(f"\nLocal STT Average: {avg:.0f}ms")
        logger.info(f"Target: ~125ms")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
