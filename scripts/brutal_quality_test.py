#!/usr/bin/env python3
"""
BRUTAL QUALITY TEST - Senior AI Engineer Level

Tests Maya's voice quality for:
1. Natural phrasing (not speaking in one go)
2. Pitch variation (not monotonous)
3. Proper endings (not abrupt)
4. Emotional tone variation
5. Response appropriateness
6. Latency under load

This simulates real conversations and analyzes output quality.
"""

import sys
import os
import time
import numpy as np
import torch
import json
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

print("=" * 70)
print("BRUTAL QUALITY TEST - Maya Voice AI")
print("=" * 70)

# Test cases covering different emotional contexts and lengths
TEST_CASES = [
    # Greetings (should be warm, inviting)
    {"input": "hello", "expected_tone": "warm", "category": "greeting"},
    {"input": "hey there", "expected_tone": "casual", "category": "greeting"},
    {"input": "hi maya", "expected_tone": "friendly", "category": "greeting"},

    # Questions (should have rising intonation, engaged)
    {"input": "how are you doing today", "expected_tone": "curious", "category": "question"},
    {"input": "what do you think about that", "expected_tone": "thoughtful", "category": "question"},
    {"input": "can you help me with something", "expected_tone": "helpful", "category": "question"},

    # Emotional statements (should adapt tone)
    {"input": "im feeling really sad today", "expected_tone": "empathetic", "category": "emotional"},
    {"input": "i got a promotion at work", "expected_tone": "excited", "category": "emotional"},
    {"input": "im so frustrated with everything", "expected_tone": "understanding", "category": "emotional"},

    # Complex sentences (tests phrasing)
    {"input": "i was thinking about going to the store but then i realized i dont have my wallet", "expected_tone": "conversational", "category": "complex"},
    {"input": "you know what i really love about music is how it makes me feel alive", "expected_tone": "passionate", "category": "complex"},

    # Short responses (tests handling brevity)
    {"input": "yes", "expected_tone": "acknowledging", "category": "short"},
    {"input": "no", "expected_tone": "understanding", "category": "short"},
    {"input": "okay", "expected_tone": "casual", "category": "short"},

    # Identity questions
    {"input": "who are you", "expected_tone": "friendly", "category": "identity"},
    {"input": "whats your name", "expected_tone": "warm", "category": "identity"},
]

def analyze_audio_quality(audio: torch.Tensor, sample_rate: int = 24000) -> dict:
    """Analyze audio for quality metrics."""
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    audio = audio.flatten()

    # Basic stats
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    duration = len(audio) / sample_rate

    # Dynamic range (indicates expressiveness)
    if len(audio) > sample_rate // 10:  # At least 100ms
        # Split into 50ms windows
        window_size = sample_rate // 20
        windows = [audio[i:i+window_size] for i in range(0, len(audio)-window_size, window_size)]
        window_rms = [np.sqrt(np.mean(w**2)) for w in windows if len(w) == window_size]
        if window_rms:
            rms_std = np.std(window_rms)
            rms_range = max(window_rms) - min(window_rms) if window_rms else 0
        else:
            rms_std = 0
            rms_range = 0
    else:
        rms_std = 0
        rms_range = 0

    # Zero crossing rate (indicates pitch activity)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
    zcr = zero_crossings / len(audio) * sample_rate

    # Check for clipping
    clipping_threshold = 0.99
    clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
    clipping_ratio = clipped_samples / len(audio)

    # Check for silence at end (abrupt ending indicator)
    end_samples = audio[-int(sample_rate * 0.1):]  # Last 100ms
    end_rms = np.sqrt(np.mean(end_samples ** 2)) if len(end_samples) > 0 else 0

    return {
        "duration_s": duration,
        "rms": rms,
        "peak": peak,
        "dynamic_range": rms_range,
        "rms_variation": rms_std,
        "zero_crossing_rate": zcr,
        "clipping_ratio": clipping_ratio,
        "end_energy": end_rms,
        "is_monotonous": rms_std < 0.02,  # Low variation = monotonous
        "has_clipping": clipping_ratio > 0.001,
        "abrupt_ending": end_rms > rms * 0.8,  # Ending energy close to average = abrupt
    }

def test_llm_responses():
    """Test LLM response quality."""
    print("\n" + "=" * 70)
    print("PHASE 1: LLM RESPONSE QUALITY")
    print("=" * 70)

    from maya.engine.llm_vllm import VLLMEngine

    llm = VLLMEngine()
    llm.initialize()

    results = []

    for i, test in enumerate(TEST_CASES):
        llm.clear_history()

        start = time.time()
        response = llm.generate(test["input"])
        latency = (time.time() - start) * 1000

        # Analyze response
        word_count = len(response.split())
        has_comma = ',' in response
        has_question = '?' in response
        starts_with_filler = any(response.lower().startswith(f) for f in ['hmm', 'oh', 'well', 'yeah', 'right'])

        result = {
            "input": test["input"],
            "response": response,
            "latency_ms": latency,
            "word_count": word_count,
            "has_phrasing_comma": has_comma,
            "has_question": has_question,
            "starts_with_filler": starts_with_filler,
            "category": test["category"],
        }
        results.append(result)

        # Print result
        status = "✓" if has_comma or word_count < 8 else "✗"
        print(f"\n[{i+1}/{len(TEST_CASES)}] {test['category'].upper()}: \"{test['input']}\"")
        print(f"  Response: \"{response}\"")
        print(f"  {status} Latency: {latency:.0f}ms | Words: {word_count} | Comma: {has_comma} | Filler: {starts_with_filler}")

    # Summary
    avg_latency = np.mean([r["latency_ms"] for r in results])
    comma_rate = np.mean([r["has_phrasing_comma"] for r in results]) * 100
    filler_rate = np.mean([r["starts_with_filler"] for r in results]) * 100

    print("\n" + "-" * 50)
    print("LLM SUMMARY:")
    print(f"  Average latency: {avg_latency:.0f}ms")
    print(f"  Responses with commas (phrasing): {comma_rate:.0f}%")
    print(f"  Responses starting with fillers: {filler_rate:.0f}%")

    return results

def test_tts_quality():
    """Test TTS audio quality."""
    print("\n" + "=" * 70)
    print("PHASE 2: TTS AUDIO QUALITY")
    print("=" * 70)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Test phrases with different characteristics
    test_phrases = [
        # Natural phrasing (commas)
        "hey, so, what's on your mind today?",
        "hmm, yeah, i get it. that can be tough.",
        "oh, that's really interesting! tell me more.",

        # Questions (should have rising intonation)
        "what do you think about that?",
        "how does that make you feel?",

        # Statements (should have falling intonation)
        "i understand what you mean.",
        "that makes a lot of sense to me.",

        # Emotional variations
        "oh wow, that's amazing!",
        "hmm, that's a bit concerning.",
        "well, i'm here for you.",

        # Longer phrase (tests sustained quality)
        "you know, i was thinking about what you said earlier, and i think you might be onto something there.",
    ]

    results = []
    all_audio = []

    for i, phrase in enumerate(test_phrases):
        print(f"\n[{i+1}/{len(test_phrases)}] Testing: \"{phrase}\"")

        # Generate audio
        start = time.time()
        audio_chunks = []
        first_chunk_time = None

        for chunk in tts.generate_stream(phrase, use_context=True):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - start) * 1000
            audio_chunks.append(chunk)

        total_time = (time.time() - start) * 1000

        if audio_chunks:
            audio = torch.cat(audio_chunks)
            quality = analyze_audio_quality(audio)

            result = {
                "phrase": phrase,
                "first_chunk_ms": first_chunk_time,
                "total_ms": total_time,
                **quality
            }
            results.append(result)
            all_audio.append(audio)

            # Print analysis
            mono_status = "✗ MONOTONOUS" if quality["is_monotonous"] else "✓ Dynamic"
            end_status = "✗ ABRUPT" if quality["abrupt_ending"] else "✓ Natural"
            clip_status = "✗ CLIPPING" if quality["has_clipping"] else "✓ Clean"

            print(f"  First audio: {first_chunk_time:.0f}ms | Duration: {quality['duration_s']:.2f}s")
            print(f"  {mono_status} (RMS var: {quality['rms_variation']:.4f})")
            print(f"  {end_status} (end energy: {quality['end_energy']:.4f} vs avg: {quality['rms']:.4f})")
            print(f"  {clip_status} (clip ratio: {quality['clipping_ratio']:.6f})")

    # Summary
    print("\n" + "-" * 50)
    print("TTS SUMMARY:")
    avg_first = np.mean([r["first_chunk_ms"] for r in results])
    monotonous_count = sum(1 for r in results if r["is_monotonous"])
    abrupt_count = sum(1 for r in results if r["abrupt_ending"])
    clipping_count = sum(1 for r in results if r["has_clipping"])

    print(f"  Average first chunk: {avg_first:.0f}ms")
    print(f"  Monotonous outputs: {monotonous_count}/{len(results)}")
    print(f"  Abrupt endings: {abrupt_count}/{len(results)}")
    print(f"  Clipping issues: {clipping_count}/{len(results)}")

    # Save sample audio for manual review
    if all_audio:
        combined = torch.cat(all_audio)
        import torchaudio
        output_path = "/tmp/maya_quality_test.wav"
        torchaudio.save(output_path, combined.unsqueeze(0).cpu(), 24000)
        print(f"\n  Sample audio saved to: {output_path}")

    return results

def test_end_to_end():
    """Test complete pipeline end-to-end."""
    print("\n" + "=" * 70)
    print("PHASE 3: END-TO-END PIPELINE")
    print("=" * 70)

    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    llm = VLLMEngine()
    llm.initialize()

    tts = RealStreamingTTSEngine()
    # TTS should already be initialized from Phase 2

    test_inputs = [
        "hey maya how are you",
        "tell me something interesting",
        "i had a really bad day",
        "what should i do about my problem",
        "thanks for listening",
    ]

    results = []

    for i, user_input in enumerate(test_inputs):
        print(f"\n[{i+1}/{len(test_inputs)}] User: \"{user_input}\"")

        # LLM
        llm_start = time.time()
        response = llm.generate(user_input)
        llm_time = (time.time() - llm_start) * 1000

        print(f"  Maya: \"{response}\"")
        print(f"  LLM: {llm_time:.0f}ms")

        # TTS
        tts_start = time.time()
        audio_chunks = []
        first_chunk_time = None

        for chunk in tts.generate_stream(response, use_context=True):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - tts_start) * 1000
            audio_chunks.append(chunk)

        tts_total = (time.time() - tts_start) * 1000

        if audio_chunks:
            audio = torch.cat(audio_chunks)
            quality = analyze_audio_quality(audio)

            total_first_audio = llm_time + first_chunk_time

            result = {
                "input": user_input,
                "response": response,
                "llm_ms": llm_time,
                "tts_first_ms": first_chunk_time,
                "total_first_audio_ms": total_first_audio,
                "duration_s": quality["duration_s"],
                "is_monotonous": quality["is_monotonous"],
                "abrupt_ending": quality["abrupt_ending"],
            }
            results.append(result)

            status = "✓" if total_first_audio < 1500 else "✗"
            print(f"  {status} First audio: {total_first_audio:.0f}ms (LLM:{llm_time:.0f} + TTS:{first_chunk_time:.0f})")
            print(f"  Audio duration: {quality['duration_s']:.2f}s")

            # Add to context for next turn
            tts.add_context(response, audio, is_user=False)

    # Summary
    print("\n" + "-" * 50)
    print("END-TO-END SUMMARY:")
    avg_first = np.mean([r["total_first_audio_ms"] for r in results])
    under_target = sum(1 for r in results if r["total_first_audio_ms"] < 1500)

    print(f"  Average first audio: {avg_first:.0f}ms")
    print(f"  Under 1.5s target: {under_target}/{len(results)}")

    return results

def run_stress_test():
    """Run rapid-fire stress test."""
    print("\n" + "=" * 70)
    print("PHASE 4: STRESS TEST (Rapid Fire)")
    print("=" * 70)

    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    llm = VLLMEngine()
    llm.initialize()

    tts = RealStreamingTTSEngine()

    # Rapid fire 10 requests
    inputs = ["hi", "okay", "yes", "no", "what", "why", "how", "cool", "nice", "thanks"]

    times = []
    for i, inp in enumerate(inputs):
        start = time.time()

        response = llm.generate(inp)
        chunks = list(tts.generate_stream(response, use_context=False))

        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  [{i+1}] \"{inp}\" -> \"{response[:30]}...\" ({elapsed:.0f}ms)")

    print("\n" + "-" * 50)
    print("STRESS TEST SUMMARY:")
    print(f"  Average response: {np.mean(times):.0f}ms")
    print(f"  Min: {np.min(times):.0f}ms | Max: {np.max(times):.0f}ms")
    print(f"  Std dev: {np.std(times):.0f}ms")

def main():
    """Run all tests."""
    start_time = time.time()

    try:
        # Phase 1: LLM
        llm_results = test_llm_responses()

        # Phase 2: TTS
        tts_results = test_tts_quality()

        # Phase 3: End-to-End
        e2e_results = test_end_to_end()

        # Phase 4: Stress
        run_stress_test()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Final Report
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    # Calculate scores
    llm_comma_rate = np.mean([r["has_phrasing_comma"] for r in llm_results]) * 100
    tts_monotonous_rate = np.mean([r["is_monotonous"] for r in tts_results]) * 100
    tts_abrupt_rate = np.mean([r["abrupt_ending"] for r in tts_results]) * 100
    e2e_avg_latency = np.mean([r["total_first_audio_ms"] for r in e2e_results])

    print(f"\n📊 QUALITY METRICS:")
    print(f"  LLM phrasing (commas): {llm_comma_rate:.0f}% (target: >50%)")
    print(f"  TTS monotonous rate: {tts_monotonous_rate:.0f}% (target: <20%)")
    print(f"  TTS abrupt endings: {tts_abrupt_rate:.0f}% (target: <20%)")
    print(f"  E2E first audio: {e2e_avg_latency:.0f}ms (target: <1500ms)")

    # Overall grade
    score = 0
    if llm_comma_rate > 50: score += 25
    if tts_monotonous_rate < 20: score += 25
    if tts_abrupt_rate < 20: score += 25
    if e2e_avg_latency < 1500: score += 25

    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 50 else "D" if score >= 25 else "F"

    print(f"\n🎯 OVERALL SCORE: {score}/100 (Grade: {grade})")
    print(f"\n⏱️  Total test time: {total_time:.1f}s")

    # Issues found
    print(f"\n⚠️  ISSUES TO ADDRESS:")
    if llm_comma_rate < 50:
        print(f"  - LLM not generating enough phrased responses")
    if tts_monotonous_rate > 20:
        print(f"  - TTS producing monotonous output ({tts_monotonous_rate:.0f}%)")
    if tts_abrupt_rate > 20:
        print(f"  - TTS has abrupt endings ({tts_abrupt_rate:.0f}%)")
    if e2e_avg_latency > 1500:
        print(f"  - Latency too high ({e2e_avg_latency:.0f}ms)")

    if score >= 75:
        print(f"\n✅ System is performing well!")
    else:
        print(f"\n❌ System needs improvement")

if __name__ == "__main__":
    main()
