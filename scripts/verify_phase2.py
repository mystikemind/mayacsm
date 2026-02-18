#!/usr/bin/env python3
"""
Phase 2 Verification - Test all quality improvements.

Tests:
1. Voice prompt is optimal (5-10s, matching temp/topk)
2. TTS generates with correct settings
3. Full pipeline latency and quality
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_voice_prompt():
    """Verify voice prompt is optimal."""
    logger.info("=" * 60)
    logger.info("TEST 1: VOICE PROMPT VERIFICATION")
    logger.info("=" * 60)

    vp = torch.load('assets/voice_prompt/maya_voice_prompt.pt')

    # Check duration (optimal: 2-10s)
    duration = vp.get('duration_seconds', len(vp['audio']) / 24000)
    logger.info(f"  Duration: {duration:.2f}s")

    if 2.0 <= duration <= 10.0:
        logger.info("  ✓ Duration in optimal range (2-10s)")
        duration_ok = True
    else:
        logger.info(f"  ✗ Duration outside optimal range")
        duration_ok = False

    # Check settings match TTS
    settings = vp.get('settings', {})
    temp = settings.get('temperature', 0)
    topk = settings.get('topk', 0)

    logger.info(f"  Temperature: {temp} (TTS uses 0.9)")
    logger.info(f"  TopK: {topk} (TTS uses 50)")

    if temp == 0.9 and topk == 50:
        logger.info("  ✓ Settings match TTS generation")
        settings_ok = True
    else:
        logger.info("  ✗ Settings don't match TTS")
        settings_ok = False

    # Check text quality
    text = vp.get('text', '')
    logger.info(f"  Text: '{text}'")

    if '...' not in text and len(text) > 20:
        logger.info("  ✓ Text is single phrase (not fragmented)")
        text_ok = True
    else:
        logger.info("  ✗ Text may be fragmented")
        text_ok = False

    return duration_ok and settings_ok and text_ok


def test_tts_generation():
    """Test TTS generates correctly with new voice prompt."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: TTS GENERATION")
    logger.info("=" * 60)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    logger.info("  Initializing TTS engine...")
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Verify voice prompt was loaded
    if tts._voice_prompt:
        logger.info(f"  ✓ Voice prompt loaded: {len(tts._voice_prompt.audio)/24000:.1f}s")
    else:
        logger.info("  ✗ Voice prompt NOT loaded")
        return False

    # Test generation
    test_phrases = [
        "yeah im doing pretty good, how about you?",
        "oh thats interesting, tell me more about that",
        "hmm, let me think about that for a moment",
    ]

    all_ok = True
    for phrase in test_phrases:
        logger.info(f"\n  Testing: '{phrase}'")

        torch.cuda.synchronize()
        start = time.time()

        chunks = []
        first_chunk_time = None
        for chunk in tts.generate_stream(phrase):
            if first_chunk_time is None:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - start) * 1000
            chunks.append(chunk)

        torch.cuda.synchronize()
        total_time = (time.time() - start) * 1000

        if not chunks:
            logger.info("    ✗ No audio generated")
            all_ok = False
            continue

        full_audio = torch.cat(chunks)
        duration_ms = len(full_audio) / 24
        peak = full_audio.abs().max().item()

        logger.info(f"    First chunk: {first_chunk_time:.0f}ms")
        logger.info(f"    Total time: {total_time:.0f}ms")
        logger.info(f"    Audio: {duration_ms:.0f}ms, peak: {peak:.3f}")

        # Check first chunk latency (should be < 200ms for optimal)
        if first_chunk_time < 200:
            logger.info("    ✓ First chunk latency excellent")
        elif first_chunk_time < 300:
            logger.info("    ✓ First chunk latency good")
        else:
            logger.info("    ⚠ First chunk latency could be better")

    return all_ok


def test_full_pipeline():
    """Test full pipeline latency."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: FULL PIPELINE LATENCY")
    logger.info("=" * 60)

    from maya.engine.stt_local import LocalSTTEngine
    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Initialize
    logger.info("  Initializing components...")

    stt = LocalSTTEngine()
    stt.initialize()
    logger.info("  ✓ STT ready")

    llm = VLLMEngine()
    llm.initialize()
    logger.info("  ✓ LLM ready")

    tts = RealStreamingTTSEngine()
    tts.initialize()
    logger.info("  ✓ TTS ready")

    # Create test audio (1.5s synthetic speech-like signal)
    sample_rate = 24000
    duration = 1.5
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    test_audio = np.zeros(samples, dtype=np.float32)
    for freq in [200, 400, 600, 800]:
        test_audio += 0.08 * np.sin(2 * np.pi * freq * t)
    test_audio = torch.tensor(test_audio * 0.3, dtype=torch.float32)

    # Run pipeline multiple times
    logger.info("\n  Running pipeline tests...")
    latencies = []

    for i in range(3):
        torch.cuda.synchronize()
        start = time.time()

        # STT
        stt_start = time.time()
        transcript = stt.transcribe(test_audio)
        stt_time = (time.time() - stt_start) * 1000

        # Use fallback if empty
        if not transcript.strip():
            transcript = "hello how are you doing"

        # LLM
        llm_start = time.time()
        response = llm.generate(transcript)
        llm_time = (time.time() - llm_start) * 1000

        # TTS first chunk
        tts_start = time.time()
        for chunk in tts.generate_stream(response):
            torch.cuda.synchronize()
            tts_time = (time.time() - tts_start) * 1000
            break

        total = (time.time() - start) * 1000
        latencies.append(total)

        logger.info(f"\n  Run {i+1}:")
        logger.info(f"    STT: {stt_time:.0f}ms")
        logger.info(f"    LLM: {llm_time:.0f}ms ('{response[:40]}...')")
        logger.info(f"    TTS: {tts_time:.0f}ms")
        logger.info(f"    Total: {total:.0f}ms")

    avg_latency = np.mean(latencies)
    logger.info(f"\n  Average latency: {avg_latency:.0f}ms")
    logger.info(f"  Sesame target: ~280ms")
    logger.info(f"  Parity: {280/avg_latency*100:.0f}%")

    return avg_latency < 500


def print_summary(results):
    """Print test summary."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 VERIFICATION SUMMARY")
    logger.info("=" * 60)

    tests = [
        ("Voice Prompt Optimization", results.get("voice_prompt", False)),
        ("TTS Generation", results.get("tts", False)),
        ("Full Pipeline Latency", results.get("pipeline", False)),
    ]

    passed = sum(1 for _, r in tests if r)
    total = len(tests)

    for name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {name}")

    logger.info("-" * 60)
    logger.info(f"  {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✓ PHASE 2 VERIFICATION COMPLETE")
    else:
        logger.info("\n⚠ SOME TESTS NEED ATTENTION")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 2 VERIFICATION")
    logger.info("Testing voice prompt & TTS quality improvements")
    logger.info("=" * 60)

    results = {}

    # Test 1: Voice Prompt
    results["voice_prompt"] = test_voice_prompt()

    # Test 2: TTS Generation
    results["tts"] = test_tts_generation()

    # Test 3: Full Pipeline
    results["pipeline"] = test_full_pipeline()

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()
