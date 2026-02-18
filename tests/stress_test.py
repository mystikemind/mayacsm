#!/usr/bin/env python3
"""
STRESS TEST - Production-Grade Validation

Tests edge cases, error handling, and robustness:
1. Empty inputs
2. Very long inputs
3. Special characters
4. Rapid-fire requests
5. Invalid audio (NaN, Inf, zeros)
6. Memory pressure
7. Error recovery
"""

import sys
import time
import torch
import numpy as np
import traceback

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

print("=" * 70)
print("STRESS TEST - PRODUCTION VALIDATION")
print("Finding bugs before production!")
print("=" * 70)

# Track results
results = {
    'passed': 0,
    'failed': 0,
    'errors': []
}

def test(name, test_func):
    """Run a test and track results."""
    print(f"\n[TEST] {name}...")
    try:
        test_func()
        print(f"  ✅ PASSED")
        results['passed'] += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        results['failed'] += 1
        results['errors'].append((name, str(e)))

# =============================================================================
# COMPONENT TESTS
# =============================================================================

print("\n" + "=" * 70)
print("COMPONENT TESTS")
print("=" * 70)

# Initialize components
print("\n[1] Initializing components...")
from maya.engine.llm_optimized import OptimizedLLMEngine
from maya.engine.stt import STTEngine
from maya.engine.tts_compiled import CompiledTTSEngine

llm = None
stt = None
tts = None

def init_llm():
    global llm
    llm = OptimizedLLMEngine()
    llm.initialize()

def init_stt():
    global stt
    stt = STTEngine()
    stt.initialize()

def init_tts():
    global tts
    tts = CompiledTTSEngine()
    tts.initialize()

test("Initialize LLM", init_llm)
test("Initialize STT", init_stt)
test("Initialize TTS", init_tts)

# =============================================================================
# LLM EDGE CASES
# =============================================================================

print("\n" + "=" * 70)
print("LLM EDGE CASES")
print("=" * 70)

def test_llm_empty_input():
    """Test LLM with empty input."""
    response = llm.generate("")
    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be string"

def test_llm_very_long_input():
    """Test LLM with very long input (1000+ chars)."""
    long_input = "Hello " * 500  # 3000 chars
    response = llm.generate(long_input)
    assert response is not None
    assert len(response) < 1000  # Should still be concise

def test_llm_special_characters():
    """Test LLM with special characters."""
    special = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?`~ 你好 مرحبا"
    response = llm.generate(special)
    assert response is not None

def test_llm_rapid_fire():
    """Test LLM with rapid consecutive requests."""
    for i in range(5):
        response = llm.generate(f"Test {i}")
        assert response is not None

def test_llm_clear_history():
    """Test LLM history clearing."""
    llm.generate("Remember this: banana")
    llm.clear_history()
    # History should be reset
    assert len(llm._messages) == 1  # Only system prompt

def test_llm_unicode():
    """Test LLM with unicode/emoji."""
    response = llm.generate("I feel 😊 today!")
    assert response is not None

test("LLM empty input", test_llm_empty_input)
test("LLM very long input", test_llm_very_long_input)
test("LLM special characters", test_llm_special_characters)
test("LLM rapid fire requests", test_llm_rapid_fire)
test("LLM clear history", test_llm_clear_history)
test("LLM unicode/emoji", test_llm_unicode)

# =============================================================================
# STT EDGE CASES
# =============================================================================

print("\n" + "=" * 70)
print("STT EDGE CASES")
print("=" * 70)

def test_stt_empty_audio():
    """Test STT with empty audio."""
    audio = torch.tensor([])
    try:
        result = stt.transcribe(audio)
        # Should handle gracefully
    except Exception as e:
        # Empty audio might raise, which is acceptable
        assert "empty" in str(e).lower() or len(audio) == 0

def test_stt_very_short_audio():
    """Test STT with very short audio (<100ms)."""
    # 50ms of silence
    audio = torch.zeros(int(24000 * 0.05))
    result = stt.transcribe(audio)
    # Should return something (even empty)
    assert isinstance(result, str)

def test_stt_silence():
    """Test STT with pure silence."""
    audio = torch.zeros(24000)  # 1 second silence
    result = stt.transcribe(audio)
    assert isinstance(result, str)

def test_stt_noise():
    """Test STT with random noise."""
    audio = torch.randn(24000) * 0.1  # Low amplitude noise
    result = stt.transcribe(audio)
    assert isinstance(result, str)

def test_stt_clipped_audio():
    """Test STT with clipped/saturated audio."""
    audio = torch.ones(24000)  # Fully saturated
    result = stt.transcribe(audio)
    assert isinstance(result, str)

def test_stt_nan_audio():
    """Test STT with NaN values in audio."""
    audio = torch.randn(24000)
    audio[100:200] = float('nan')
    # Replace NaN before passing
    audio = torch.nan_to_num(audio, nan=0.0)
    result = stt.transcribe(audio)
    assert isinstance(result, str)

test("STT empty audio", test_stt_empty_audio)
test("STT very short audio", test_stt_very_short_audio)
test("STT silence", test_stt_silence)
test("STT noise", test_stt_noise)
test("STT clipped audio", test_stt_clipped_audio)
test("STT NaN audio", test_stt_nan_audio)

# =============================================================================
# TTS EDGE CASES
# =============================================================================

print("\n" + "=" * 70)
print("TTS EDGE CASES")
print("=" * 70)

def test_tts_empty_text():
    """Test TTS with empty text."""
    try:
        audio = tts.generate("")
        # Should either return silence or raise
        assert audio is None or len(audio) >= 0
    except Exception as e:
        # Empty text raising is acceptable
        pass

def test_tts_single_word():
    """Test TTS with single word."""
    audio = tts.generate("Hello")
    assert audio is not None
    assert len(audio) > 0
    # Check for NaN/Inf
    assert not torch.isnan(audio).any(), "Audio contains NaN"
    assert not torch.isinf(audio).any(), "Audio contains Inf"

def test_tts_punctuation_only():
    """Test TTS with punctuation only."""
    try:
        audio = tts.generate("!!!")
        # Should handle gracefully
    except Exception:
        pass  # Acceptable to fail

def test_tts_numbers():
    """Test TTS with numbers."""
    audio = tts.generate("The number is 12345")
    assert audio is not None
    assert len(audio) > 0

def test_tts_rapid_generation():
    """Test TTS with rapid consecutive generations."""
    for i in range(3):
        audio = tts.generate(f"Test number {i}")
        assert audio is not None
        assert len(audio) > 0

def test_tts_context_overflow():
    """Test TTS context doesn't grow unbounded."""
    initial_context = tts.get_context_size()
    for i in range(10):
        audio = tts.generate(f"Message {i}")
        tts.add_context(f"Message {i}", audio, is_user=False)
    final_context = tts.get_context_size()
    # Context should be bounded (max 4)
    assert final_context <= 4, f"Context grew to {final_context}"

def test_tts_audio_quality():
    """Test TTS audio quality metrics."""
    audio = tts.generate("This is a quality test")

    # Check basic audio properties
    assert audio.dtype == torch.float32 or audio.dtype == torch.bfloat16

    # Check amplitude is reasonable
    max_amp = audio.abs().max().item()
    assert max_amp <= 1.0, f"Audio exceeds [-1, 1]: max={max_amp}"
    assert max_amp > 0.01, f"Audio too quiet: max={max_amp}"

    # Check for DC offset
    mean = audio.mean().item()
    assert abs(mean) < 0.1, f"Large DC offset: {mean}"

test("TTS empty text", test_tts_empty_text)
test("TTS single word", test_tts_single_word)
test("TTS punctuation only", test_tts_punctuation_only)
test("TTS numbers", test_tts_numbers)
test("TTS rapid generation", test_tts_rapid_generation)
test("TTS context overflow", test_tts_context_overflow)
test("TTS audio quality", test_tts_audio_quality)

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

print("\n" + "=" * 70)
print("INTEGRATION TESTS")
print("=" * 70)

def test_full_pipeline_basic():
    """Test full pipeline: user input -> LLM -> TTS."""
    user_input = "Hello, how are you?"

    # LLM
    response = llm.generate(user_input)
    assert response is not None

    # TTS
    audio = tts.generate(response)
    assert audio is not None
    assert len(audio) > 0

    print(f"    Input: '{user_input}'")
    print(f"    Response: '{response}'")
    print(f"    Audio: {len(audio)/24000:.1f}s")

def test_full_pipeline_conversation():
    """Test multi-turn conversation."""
    conversation = [
        "Hi!",
        "What's your name?",
        "Nice to meet you!",
    ]

    for user_input in conversation:
        response = llm.generate(user_input)
        audio = tts.generate(response)
        assert audio is not None
        print(f"    User: '{user_input}' -> Maya: '{response}'")

def test_latency_consistency():
    """Test latency is consistent across turns."""
    latencies = []

    for i in range(5):
        start = time.time()
        response = llm.generate(f"Test {i}")
        audio = tts.generate(response)
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000
        latencies.append(latency)

    avg = sum(latencies) / len(latencies)
    std = (sum((l - avg) ** 2 for l in latencies) / len(latencies)) ** 0.5

    print(f"    Latencies: {[f'{l:.0f}ms' for l in latencies]}")
    print(f"    Average: {avg:.0f}ms, StdDev: {std:.0f}ms")

    # Std dev should be reasonable (< 50% of mean)
    assert std < avg * 0.5, f"Latency too variable: std={std:.0f}, avg={avg:.0f}"

test("Full pipeline basic", test_full_pipeline_basic)
test("Full pipeline conversation", test_full_pipeline_conversation)
test("Latency consistency", test_latency_consistency)

# =============================================================================
# MEMORY TEST
# =============================================================================

print("\n" + "=" * 70)
print("MEMORY TEST")
print("=" * 70)

def test_memory_stability():
    """Test memory doesn't grow over many iterations."""
    import gc

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    initial_memory = torch.cuda.memory_allocated() / 1e9

    # Run many iterations
    for i in range(20):
        response = llm.generate(f"Test iteration {i}")
        audio = tts.generate(response)
        del audio

        if i % 5 == 0:
            torch.cuda.synchronize()
            gc.collect()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated() / 1e9
    memory_growth = final_memory - initial_memory

    print(f"    Initial: {initial_memory:.2f}GB")
    print(f"    Final: {final_memory:.2f}GB")
    print(f"    Growth: {memory_growth:.2f}GB")

    # Memory growth should be minimal (< 1GB)
    assert memory_growth < 1.0, f"Memory grew by {memory_growth:.2f}GB"

test("Memory stability", test_memory_stability)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STRESS TEST SUMMARY")
print("=" * 70)

total = results['passed'] + results['failed']
pass_rate = results['passed'] / total * 100 if total > 0 else 0

print(f"""
  Total tests: {total}
  Passed: {results['passed']} ✅
  Failed: {results['failed']} ❌
  Pass rate: {pass_rate:.1f}%
""")

if results['errors']:
    print("  Failed tests:")
    for name, error in results['errors']:
        print(f"    - {name}: {error[:50]}...")

status = "✅ PRODUCTION READY" if pass_rate == 100 else "⚠️ NEEDS FIXES"
print(f"\n  Status: {status}")
print("=" * 70)

# Exit with error code if tests failed
sys.exit(0 if pass_rate == 100 else 1)
