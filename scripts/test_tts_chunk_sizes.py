#!/usr/bin/env python3
"""
TTS Chunk Size Stability Test - Verify different chunk sizes work without gaps.

This script tests:
- First chunk sizes: 2, 3, 4 frames
- Playback buffer simulation
- Audio continuity (no gaps/underruns)
- Latency vs stability tradeoff

Usage:
    python scripts/test_tts_chunk_sizes.py
"""

import sys
import os
import time
import torch
import numpy as np
import logging

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PlaybackSimulator:
    """Simulate real-time audio playback to detect buffer underruns."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.buffer = []
        self.samples_played = 0
        self.underruns = 0
        self.start_time = None
        self.first_audio_time = None

    def reset(self):
        self.buffer = []
        self.samples_played = 0
        self.underruns = 0
        self.start_time = None
        self.first_audio_time = None

    def add_chunk(self, audio: torch.Tensor):
        """Add audio chunk to buffer."""
        if self.first_audio_time is None:
            self.first_audio_time = time.time()
        self.buffer.extend(audio.cpu().numpy().tolist())

    def simulate_playback(self, duration_ms: float = 100) -> dict:
        """Simulate playing audio for duration_ms.

        Returns dict with underrun info.
        """
        samples_needed = int(self.sample_rate * duration_ms / 1000)

        if len(self.buffer) < samples_needed:
            self.underruns += 1
            gap_samples = samples_needed - len(self.buffer)
            gap_ms = gap_samples / self.sample_rate * 1000
            # Consume what we have
            self.samples_played += len(self.buffer)
            self.buffer = []
            return {"underrun": True, "gap_ms": gap_ms}
        else:
            self.buffer = self.buffer[samples_needed:]
            self.samples_played += samples_needed
            return {"underrun": False, "gap_ms": 0}


def test_chunk_size(first_chunk_frames: int, num_tests: int = 5) -> dict:
    """Test a specific first chunk size configuration."""
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Temporarily modify the engine's first chunk setting
    original_frames = RealStreamingTTSEngine.FIRST_CHUNK_FRAMES
    RealStreamingTTSEngine.FIRST_CHUNK_FRAMES = first_chunk_frames

    logger.info(f"\nTesting FIRST_CHUNK_FRAMES = {first_chunk_frames}")
    logger.info("-" * 40)

    tts = RealStreamingTTSEngine()
    tts.initialize()

    test_texts = [
        "oh hey how are you doing today",
        "thats really interesting tell me more",
        "let me think about that for a second",
        "yeah i totally understand what you mean",
        "hmm thats a good question actually",
    ]

    results = []
    playback = PlaybackSimulator()

    for i, text in enumerate(test_texts[:num_tests]):
        playback.reset()

        torch.cuda.synchronize()
        gen_start = time.time()
        first_chunk_time = None
        chunk_times = []
        chunk_sizes = []

        # Generate and simulate real-time playback
        for chunk in tts.generate_stream(text, use_context=False):
            chunk_time = time.time() - gen_start
            chunk_times.append(chunk_time * 1000)
            chunk_sizes.append(len(chunk))

            if first_chunk_time is None:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - gen_start) * 1000
                # Start playback timer when first chunk arrives
                playback.start_time = time.time()

            playback.add_chunk(chunk)

            # Simulate playback catching up
            if playback.start_time:
                elapsed_since_start = (time.time() - playback.start_time) * 1000
                # How much audio should have been played by now?
                expected_samples = int(24000 * elapsed_since_start / 1000)
                # Check if we have enough in buffer
                if playback.samples_played + len(playback.buffer) < expected_samples:
                    playback.underruns += 1

        total_time = (time.time() - gen_start) * 1000
        total_audio_ms = sum(chunk_sizes) / 24000 * 1000

        result = {
            "text": text[:30],
            "first_chunk_ms": first_chunk_time,
            "first_chunk_audio_ms": chunk_sizes[0] / 24000 * 1000 if chunk_sizes else 0,
            "total_time_ms": total_time,
            "total_audio_ms": total_audio_ms,
            "num_chunks": len(chunk_times),
            "underruns": playback.underruns,
            "chunk_sizes": chunk_sizes
        }
        results.append(result)

        status = "✗ UNDERRUN" if playback.underruns > 0 else "✓ OK"
        logger.info(f"  Test {i+1}: first={first_chunk_time:.0f}ms ({result['first_chunk_audio_ms']:.0f}ms audio), "
                   f"total={total_time:.0f}ms, chunks={len(chunk_times)}, {status}")

    # Skip first result (warmup)
    results = results[1:]

    avg_first_chunk = np.mean([r["first_chunk_ms"] for r in results])
    total_underruns = sum(r["underruns"] for r in results)
    success_rate = 1 - (total_underruns / len(results)) if results else 0

    # Restore original setting
    RealStreamingTTSEngine.FIRST_CHUNK_FRAMES = original_frames

    summary = {
        "first_chunk_frames": first_chunk_frames,
        "avg_first_chunk_ms": avg_first_chunk,
        "first_chunk_audio_ms": first_chunk_frames * 80,  # 80ms per frame
        "total_underruns": total_underruns,
        "success_rate": success_rate,
        "results": results
    }

    logger.info(f"\n  Summary for {first_chunk_frames} frames:")
    logger.info(f"    Avg first chunk latency: {avg_first_chunk:.0f}ms")
    logger.info(f"    First chunk audio: {first_chunk_frames * 80}ms")
    logger.info(f"    Underruns: {total_underruns}")
    logger.info(f"    Success rate: {success_rate*100:.0f}%")

    return summary


def test_audio_continuity(first_chunk_frames: int = 2) -> dict:
    """Test audio continuity - check for clicks/gaps at chunk boundaries."""
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    import torchaudio

    logger.info(f"\nTesting Audio Continuity (first_chunk_frames={first_chunk_frames})")
    logger.info("-" * 40)

    original_frames = RealStreamingTTSEngine.FIRST_CHUNK_FRAMES
    RealStreamingTTSEngine.FIRST_CHUNK_FRAMES = first_chunk_frames

    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Generate a longer utterance
    text = "hello there its really nice to meet you how are you doing today"

    torch.cuda.synchronize()
    chunks = []
    for chunk in tts.generate_stream(text, use_context=False):
        chunks.append(chunk.cpu())

    # Concatenate all chunks
    full_audio = torch.cat(chunks)

    # Check for discontinuities at chunk boundaries
    discontinuities = []
    pos = 0
    for i, chunk in enumerate(chunks[:-1]):
        pos += len(chunk)
        if pos < len(full_audio) - 1:
            # Check sample difference at boundary
            diff = abs(full_audio[pos].item() - full_audio[pos-1].item())
            if diff > 0.3:  # Threshold for discontinuity
                discontinuities.append({
                    "position": pos,
                    "time_ms": pos / 24000 * 1000,
                    "diff": diff
                })

    # Save audio for manual inspection
    output_path = f"/tmp/tts_continuity_test_{first_chunk_frames}frames.wav"
    torchaudio.save(output_path, full_audio.unsqueeze(0), 24000)

    RealStreamingTTSEngine.FIRST_CHUNK_FRAMES = original_frames

    result = {
        "first_chunk_frames": first_chunk_frames,
        "total_chunks": len(chunks),
        "total_audio_ms": len(full_audio) / 24000 * 1000,
        "discontinuities": len(discontinuities),
        "output_path": output_path
    }

    if discontinuities:
        logger.info(f"  WARNING: {len(discontinuities)} discontinuities detected")
        for d in discontinuities[:5]:
            logger.info(f"    At {d['time_ms']:.0f}ms: diff={d['diff']:.3f}")
    else:
        logger.info(f"  ✓ No discontinuities detected - audio is smooth")

    logger.info(f"  Audio saved to: {output_path}")

    return result


def main():
    logger.info("=" * 60)
    logger.info("TTS CHUNK SIZE STABILITY TEST")
    logger.info("Testing different first chunk configurations")
    logger.info("=" * 60)

    # Test different chunk sizes
    chunk_sizes_to_test = [2, 3, 4]
    all_results = {}

    for frames in chunk_sizes_to_test:
        all_results[frames] = test_chunk_size(frames, num_tests=5)

    # Test audio continuity with our target size (2 frames)
    continuity_result = test_audio_continuity(2)

    # Final comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Frames':<8} {'Latency':<12} {'Audio':<12} {'Underruns':<12} {'Status'}")
    logger.info("-" * 60)

    for frames in chunk_sizes_to_test:
        r = all_results[frames]
        status = "✓ STABLE" if r["success_rate"] == 1.0 else "✗ UNSTABLE"
        logger.info(f"{frames:<8} {r['avg_first_chunk_ms']:.0f}ms{'':<6} "
                   f"{r['first_chunk_audio_ms']}ms{'':<6} "
                   f"{r['total_underruns']:<12} {status}")

    logger.info("-" * 60)

    # Recommendation
    best_frames = 2  # Start with smallest
    for frames in [2, 3, 4]:
        if all_results[frames]["success_rate"] == 1.0:
            best_frames = frames
            break

    logger.info(f"\nRECOMMENDATION: Use FIRST_CHUNK_FRAMES = {best_frames}")
    logger.info(f"  - Latency: {all_results[best_frames]['avg_first_chunk_ms']:.0f}ms")
    logger.info(f"  - Audio buffer: {best_frames * 80}ms")
    logger.info(f"  - Stability: {all_results[best_frames]['success_rate']*100:.0f}%")

    if best_frames == 2:
        logger.info("\n  ✓ OPTIMAL: 2 frames provides Sesame-level latency with stability")
    else:
        logger.info(f"\n  NOTE: Had to use {best_frames} frames for stability")
        logger.info(f"        Consider optimizing TTS RTF for 2-frame stability")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
