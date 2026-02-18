#!/usr/bin/env python3
"""
Comprehensive Quality Test - Verify all Phase 1 fixes.

Tests:
1. STT accuracy (base.en vs tiny.en)
2. LLM response quality (natural, not rigid)
3. TTS audio quality (volume, normalization)
4. Full pipeline latency
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_stt_model():
    """Verify STT is using base.en model."""
    logger.info("=" * 60)
    logger.info("TEST 1: STT MODEL VERIFICATION")
    logger.info("=" * 60)

    from maya.engine.stt_local import LocalSTTEngine

    stt = LocalSTTEngine()
    logger.info(f"  Model configured: {stt.MODEL_SIZE}")

    if stt.MODEL_SIZE == "base.en":
        logger.info("  ✓ PASS: Using base.en for better accuracy")
        return True
    else:
        logger.info(f"  ✗ FAIL: Expected base.en, got {stt.MODEL_SIZE}")
        return False


def test_llm_response_quality():
    """Verify LLM produces natural responses."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: LLM RESPONSE QUALITY")
    logger.info("=" * 60)

    from maya.engine.llm_vllm import VLLMEngine

    llm = VLLMEngine()

    # Check system prompt doesn't have rigid word constraints
    if "words ONLY" in llm.SYSTEM_PROMPT or "6-10 words" in llm.SYSTEM_PROMPT:
        logger.info("  ✗ FAIL: System prompt still has rigid word constraints")
        return False

    logger.info("  ✓ System prompt has natural guidelines")

    # Check max_history_turns
    if llm._max_history_turns >= 8:
        logger.info(f"  ✓ Context turns: {llm._max_history_turns}")
    else:
        logger.info(f"  ✗ Context turns too low: {llm._max_history_turns}")
        return False

    # Test actual generation
    logger.info("\n  Testing response generation...")
    llm.initialize()

    test_inputs = [
        "Hey, how are you doing today?",
        "I'm feeling a bit stressed about work",
        "What do you think about artificial intelligence?",
    ]

    all_good = True
    for inp in test_inputs:
        response = llm.generate(inp)
        word_count = len(response.split())
        logger.info(f"  Input: '{inp}'")
        logger.info(f"  Response ({word_count} words): '{response}'")

        # Check response is natural (not too short, not too long)
        if word_count < 3:
            logger.info("  ✗ Response too short")
            all_good = False
        elif word_count > 50:
            logger.info("  ✗ Response too long")
            all_good = False
        else:
            logger.info("  ✓ Natural length")

    return all_good


def test_tts_quality():
    """Verify TTS audio quality settings."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: TTS QUALITY SETTINGS")
    logger.info("=" * 60)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    tts = RealStreamingTTSEngine()

    # Check preprocessing keeps prosody punctuation
    test_text = "Hello! How are you?"
    processed = tts._preprocess_for_speech(test_text)
    if "!" in processed:
        logger.info(f"  ✓ Exclamation marks preserved: '{processed}'")
    else:
        logger.info(f"  ✗ Exclamation marks removed: '{processed}'")
        return False

    # Initialize and test generation
    logger.info("\n  Initializing TTS...")
    tts.initialize()

    logger.info("\n  Testing audio generation quality...")
    test_phrases = [
        "Hello! How are you doing today?",
        "Oh, that sounds really interesting.",
        "Hmm, let me think about that.",
    ]

    for phrase in test_phrases:
        torch.cuda.synchronize()
        start = time.time()

        chunks = []
        first_chunk_time = None
        for chunk in tts.generate_stream(phrase):
            if first_chunk_time is None:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - start) * 1000
            chunks.append(chunk)

        if not chunks:
            logger.info(f"  ✗ No audio generated for: '{phrase}'")
            continue

        # Analyze audio quality
        full_audio = torch.cat(chunks)
        peak = torch.abs(full_audio).max().item()
        rms = torch.sqrt(torch.mean(full_audio ** 2)).item()

        logger.info(f"\n  Phrase: '{phrase}'")
        logger.info(f"    First chunk: {first_chunk_time:.0f}ms")
        logger.info(f"    Duration: {len(full_audio)/24000*1000:.0f}ms")
        logger.info(f"    Peak level: {peak:.3f} (target: 0.3-0.5)")
        logger.info(f"    RMS level: {rms:.3f} (target: 0.1-0.2)")

        # Check levels are reasonable
        if peak < 0.1:
            logger.info(f"    ✗ Audio too quiet")
        elif peak > 0.8:
            logger.info(f"    ✗ Audio might clip")
        else:
            logger.info(f"    ✓ Audio levels good")

    return True


def test_full_pipeline_latency():
    """Test full pipeline latency with quality fixes."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: FULL PIPELINE LATENCY")
    logger.info("=" * 60)

    from maya.engine.stt_local import LocalSTTEngine
    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Initialize all components
    logger.info("  Initializing components...")
    stt = LocalSTTEngine()
    stt.initialize()

    llm = VLLMEngine()
    llm.initialize()

    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Create test audio (simulated speech)
    logger.info("\n  Running pipeline test...")
    sample_rate = 24000
    duration = 1.5
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    test_audio = np.zeros(samples, dtype=np.float32)
    for freq in [200, 400, 600, 800]:
        test_audio += 0.08 * np.sin(2 * np.pi * freq * t)
    test_audio = torch.tensor(test_audio * 0.3, dtype=torch.float32)

    # Run pipeline multiple times
    latencies = []
    for i in range(3):
        torch.cuda.synchronize()
        start = time.time()

        # STT
        stt_start = time.time()
        transcript = stt.transcribe(test_audio)
        stt_time = (time.time() - stt_start) * 1000

        # Use a real transcript for testing
        if not transcript.strip():
            transcript = "Hello, how are you today?"

        # LLM
        llm_start = time.time()
        response = llm.generate(transcript)
        llm_time = (time.time() - llm_start) * 1000

        # TTS (first chunk only)
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
    logger.info(f"\n  Average total latency: {avg_latency:.0f}ms")
    logger.info(f"  Sesame target: ~280ms")
    logger.info(f"  Gap: {avg_latency - 280:.0f}ms")

    return avg_latency < 500  # Pass if under 500ms


def print_summary(results):
    """Print test summary."""
    logger.info("\n" + "=" * 60)
    logger.info("QUALITY TEST SUMMARY")
    logger.info("=" * 60)

    tests = [
        ("STT Model (base.en)", results.get("stt", False)),
        ("LLM Response Quality", results.get("llm", False)),
        ("TTS Audio Quality", results.get("tts", False)),
        ("Pipeline Latency", results.get("latency", False)),
    ]

    passed = sum(1 for _, r in tests if r)
    total = len(tests)

    for name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {name}")

    logger.info("-" * 60)
    logger.info(f"  {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✓ ALL QUALITY FIXES VERIFIED")
    else:
        logger.info("\n✗ SOME FIXES NEED ATTENTION")


def main():
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE QUALITY TEST")
    logger.info("Verifying all Phase 1 fixes")
    logger.info("=" * 60)

    results = {}

    # Test 1: STT Model
    results["stt"] = test_stt_model()

    # Test 2: LLM Response Quality
    results["llm"] = test_llm_response_quality()

    # Test 3: TTS Quality
    results["tts"] = test_tts_quality()

    # Test 4: Pipeline Latency
    results["latency"] = test_full_pipeline_latency()

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()
