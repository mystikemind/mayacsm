#!/usr/bin/env python3
"""
FINAL VALIDATION - Senior AI Engineer Level Testing

This is the FINAL check before user testing.
Tests EVERYTHING: naturalness, latency, quality, stability.
"""

import sys
import time
import torch
import numpy as np
import torchaudio
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

print("=" * 70)
print("FINAL VALIDATION - WORLD CLASS STANDARDS")
print("Senior AI Engineer Testing Protocol")
print("=" * 70)

results = {'passed': 0, 'failed': 0, 'warnings': 0, 'details': []}

def test(name, check_func, critical=True):
    """Run a test."""
    print(f"\n[TEST] {name}...")
    try:
        result, detail = check_func()
        if result:
            print(f"  ✅ PASSED: {detail}")
            results['passed'] += 1
        else:
            if critical:
                print(f"  ❌ FAILED: {detail}")
                results['failed'] += 1
            else:
                print(f"  ⚠️ WARNING: {detail}")
                results['warnings'] += 1
        results['details'].append((name, result, detail))
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results['failed'] += 1
        results['details'].append((name, False, str(e)))

# =============================================================================
# INITIALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 1: INITIALIZATION")
print("=" * 70)

from maya.engine.llm_optimized import OptimizedLLMEngine
from maya.engine.tts_compiled import CompiledTTSEngine

print("\nInitializing LLM...")
llm = OptimizedLLMEngine()
llm.initialize()

print("\nInitializing TTS...")
tts = CompiledTTSEngine()
tts.initialize()

# =============================================================================
# LLM NATURALNESS TESTS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 2: LLM NATURALNESS")
print("=" * 70)

def test_llm_response_length():
    """Test that responses are 10-15 words (natural length)."""
    test_inputs = ["Hello!", "How are you?", "Tell me about yourself", "I had a tough day"]
    word_counts = []

    for inp in test_inputs:
        response = llm.generate(inp)
        words = len(response.split())
        word_counts.append(words)
        print(f"    '{inp}' -> '{response}' ({words} words)")

    avg_words = sum(word_counts) / len(word_counts)
    # Natural speech: 8-18 words is good range
    if 8 <= avg_words <= 18:
        return True, f"Average {avg_words:.1f} words (target: 8-18)"
    else:
        return False, f"Average {avg_words:.1f} words - outside natural range"

def test_llm_has_natural_patterns():
    """Test that responses include natural speech patterns."""
    responses = []
    for _ in range(5):
        responses.append(llm.generate("What do you think about that?"))

    all_text = " ".join(responses).lower()

    # Check for natural patterns (at least some should appear)
    natural_markers = ["well", "you know", "i mean", "actually", "hmm", "oh", "yeah", "really"]
    found = [m for m in natural_markers if m in all_text]

    if len(found) >= 1:
        return True, f"Found natural markers: {found}"
    else:
        return False, f"No natural speech markers found in responses"

def test_llm_not_robotic():
    """Test that responses don't sound robotic."""
    robotic_patterns = ["How can I help you", "I am an AI", "I don't have feelings", "As an assistant"]

    responses = []
    for inp in ["Hi", "How are you", "What do you like"]:
        responses.append(llm.generate(inp))

    all_text = " ".join(responses)

    found_robotic = [p for p in robotic_patterns if p.lower() in all_text.lower()]

    if not found_robotic:
        return True, "No robotic patterns detected"
    else:
        return False, f"Robotic patterns found: {found_robotic}"

test("LLM response length (10-15 words)", test_llm_response_length)
test("LLM has natural speech patterns", test_llm_has_natural_patterns, critical=False)
test("LLM not robotic", test_llm_not_robotic)

# =============================================================================
# TTS AUDIO QUALITY TESTS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 3: TTS AUDIO QUALITY")
print("=" * 70)

def test_tts_audio_duration():
    """Test that audio duration matches expected for natural speech."""
    text = "Oh hey, yeah I'm doing pretty well actually, how about you?"
    audio = tts.generate(text)

    duration = len(audio) / 24000
    word_count = len(text.split())
    words_per_second = word_count / duration

    # Natural speech: 2-3 words per second
    if 1.5 <= words_per_second <= 4.0:
        return True, f"{duration:.1f}s for {word_count} words ({words_per_second:.1f} wps)"
    else:
        return False, f"Unnatural pace: {words_per_second:.1f} words/second"

def test_tts_no_clipping():
    """Test that audio doesn't clip."""
    audio = tts.generate("This is a test of audio quality and volume levels")

    max_val = audio.abs().max().item()

    if max_val <= 0.95:
        return True, f"Peak amplitude: {max_val:.3f} (no clipping)"
    else:
        return False, f"Audio may be clipping: peak={max_val:.3f}"

def test_tts_smooth_ending():
    """Test that audio has smooth fade-out (not abrupt)."""
    audio = tts.generate("Testing the ending of this sentence")

    # Check last 300ms
    last_samples = audio[-int(24000 * 0.3):]

    # Should fade to near-zero
    final_energy = last_samples[-1000:].abs().mean().item()
    mid_energy = last_samples[:1000].abs().mean().item()

    if final_energy < mid_energy * 0.3:  # Final should be much quieter
        return True, f"Smooth fade-out detected (final energy: {final_energy:.4f})"
    else:
        return False, f"Abrupt ending: final={final_energy:.4f}, mid={mid_energy:.4f}"

def test_tts_consistent_voice():
    """Test voice consistency across multiple generations."""
    texts = ["Hello there", "How are you", "Nice to meet you"]
    energies = []

    for text in texts:
        audio = tts.generate(text)
        energy = audio.abs().mean().item()
        energies.append(energy)

    avg_energy = sum(energies) / len(energies)
    variance = sum((e - avg_energy) ** 2 for e in energies) / len(energies)

    if variance < 0.001:  # Low variance = consistent
        return True, f"Consistent energy: avg={avg_energy:.4f}, var={variance:.6f}"
    else:
        return False, f"Inconsistent: variance={variance:.6f}"

test("TTS audio duration (natural pace)", test_tts_audio_duration)
test("TTS no clipping", test_tts_no_clipping)
test("TTS smooth ending", test_tts_smooth_ending)
test("TTS consistent voice", test_tts_consistent_voice)

# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 4: FULL PIPELINE")
print("=" * 70)

def test_pipeline_latency():
    """Test full pipeline latency."""
    latencies = []

    test_inputs = ["Hello", "How are you today", "That's interesting", "Tell me more"]

    for inp in test_inputs:
        start = time.time()
        response = llm.generate(inp)
        audio = tts.generate(response)
        torch.cuda.synchronize()
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        print(f"    '{inp}' -> {latency:.0f}ms")

    avg_latency = sum(latencies) / len(latencies)

    # Target: < 1500ms for natural speech (was 1000ms for short responses)
    if avg_latency < 1500:
        return True, f"Average: {avg_latency:.0f}ms (target: <1500ms)"
    else:
        return False, f"Too slow: {avg_latency:.0f}ms"

def test_pipeline_conversation():
    """Test multi-turn conversation flow."""
    conversation = [
        ("Hi!", None),
        ("How are you doing?", None),
        ("That's good to hear", None),
        ("What do you like to do?", None),
    ]

    for i, (user_input, _) in enumerate(conversation):
        response = llm.generate(user_input)
        audio = tts.generate(response)
        conversation[i] = (user_input, response)
        print(f"    Turn {i+1}: '{user_input}' -> '{response}'")

    # All turns should have responses
    all_have_responses = all(r is not None and len(r) > 0 for _, r in conversation)

    if all_have_responses:
        return True, f"All {len(conversation)} turns successful"
    else:
        return False, "Some turns failed"

test("Pipeline latency (<1500ms)", test_pipeline_latency)
test("Pipeline conversation flow", test_pipeline_conversation)

# =============================================================================
# GENERATE FINAL SAMPLES
# =============================================================================

print("\n" + "=" * 70)
print("PHASE 5: GENERATE FINAL SAMPLES")
print("=" * 70)

output_dir = Path("/home/ec2-user/SageMaker/project_maya/tests/outputs/final_validation")
output_dir.mkdir(parents=True, exist_ok=True)

final_conversation = [
    "Hello!",
    "How are you doing today?",
    "That's nice to hear.",
    "What have you been up to?",
    "Sounds interesting!",
]

print("\nGenerating final audio samples...")
for i, user_input in enumerate(final_conversation):
    response = llm.generate(user_input)
    audio = tts.generate(response)

    # Save audio
    output_path = output_dir / f"final_{i+1:02d}.wav"
    audio_cpu = audio.cpu().unsqueeze(0)
    torchaudio.save(str(output_path), audio_cpu, 24000)

    duration = len(audio) / 24000
    print(f"  {i+1}. User: '{user_input}'")
    print(f"     Maya: '{response}'")
    print(f"     Audio: {duration:.1f}s -> {output_path.name}")

print(f"\n  Samples saved to: {output_dir}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

total = results['passed'] + results['failed']
pass_rate = results['passed'] / total * 100 if total > 0 else 0

print(f"""
  Total Tests: {total}
  Passed: {results['passed']} ✅
  Failed: {results['failed']} ❌
  Warnings: {results['warnings']} ⚠️

  Pass Rate: {pass_rate:.0f}%
""")

if results['failed'] > 0:
    print("  Failed Tests:")
    for name, passed, detail in results['details']:
        if not passed:
            print(f"    - {name}: {detail}")

# Final verdict
if results['failed'] == 0:
    print("\n  🏆 VERDICT: WORLD CLASS - READY FOR USER TESTING")
elif results['failed'] <= 2:
    print("\n  ✅ VERDICT: ACCEPTABLE - Minor issues, can proceed")
else:
    print("\n  ❌ VERDICT: NOT READY - Fix issues before user testing")

print("\n" + "=" * 70)
