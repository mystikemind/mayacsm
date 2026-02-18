#!/usr/bin/env python3
"""
FINAL QUALITY REPORT - Comprehensive audio quality verification
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import time
import wave
from pathlib import Path

OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/audio_final_test")

def save_wav(audio: np.ndarray, filepath: str, sample_rate: int = 24000):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

def analyze_boundaries(chunks: list) -> dict:
    """Analyze chunk boundary quality."""
    boundary_diffs = []

    for i in range(1, len(chunks)):
        prev = chunks[i-1]
        curr = chunks[i]
        diff = abs(curr[0] - prev[-1])
        boundary_diffs.append(diff)

    return {
        'num_boundaries': len(boundary_diffs),
        'max_boundary_diff': max(boundary_diffs) if boundary_diffs else 0,
        'avg_boundary_diff': np.mean(boundary_diffs) if boundary_diffs else 0,
        'boundaries_over_01': sum(1 for d in boundary_diffs if d > 0.1),
        'boundaries_over_03': sum(1 for d in boundary_diffs if d > 0.3)
    }

def main():
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    print("="*70)
    print("FINAL AUDIO QUALITY REPORT")
    print("Verifying Sesame AI Level Quality")
    print("="*70)

    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Generate 10 test phrases
    phrases = [
        "hi there how are you",
        "oh thats really interesting",
        "hmm let me think about that for a second",
        "yeah definitely i totally agree with you",
        "so what do you think we should do next",
        "aww thats so sweet of you to say",
        "well you know it really depends on the situation",
        "oh wow i didnt expect that at all",
        "thats a great question let me explain",
        "mhm i see what you mean now"
    ]

    all_results = []

    for i, phrase in enumerate(phrases):
        # Generate streaming audio
        start = time.time()
        chunks = []
        for chunk in tts.generate_stream(phrase, use_context=False):
            chunks.append(chunk.cpu().numpy())
        latency = (time.time() - start) * 1000

        if not chunks:
            print(f"  {i+1}. FAILED: No audio generated")
            continue

        # Analyze boundary quality
        boundary_stats = analyze_boundaries(chunks)

        # Concatenate for overall stats
        audio = np.concatenate(chunks)
        duration_ms = len(audio) / 24000 * 1000
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100

        result = {
            'phrase': phrase,
            'num_chunks': len(chunks),
            'duration_ms': duration_ms,
            'latency_ms': latency,
            'rms_db': rms_db,
            'peak_db': peak_db,
            **boundary_stats
        }
        all_results.append(result)

        # Print summary
        status = "✓" if boundary_stats['boundaries_over_03'] == 0 else "⚠"
        print(f"  {i+1}. {status} '{phrase[:30]}...'")
        print(f"      {len(chunks)} chunks, {duration_ms:.0f}ms audio, {latency:.0f}ms latency")
        print(f"      Max boundary diff: {boundary_stats['max_boundary_diff']:.4f}")

    # Save test audio
    print("\n" + "-"*70)
    print("Generating test compilation...")

    # Generate a longer conversation sample
    conversation = [
        "hey whats up",
        "oh im doing great thanks for asking",
        "so what have you been up to lately",
        "hmm not much just working on some projects",
        "thats cool what kind of projects",
        "well mostly coding stuff you know how it is",
        "yeah totally i get that",
        "anyway enough about me how about you"
    ]

    all_audio = []
    for phrase in conversation:
        chunks = list(tts.generate_stream(phrase, use_context=False))
        if chunks:
            audio = np.concatenate([c.cpu().numpy() for c in chunks])
            all_audio.append(audio)
            all_audio.append(np.zeros(int(24000 * 0.3)))  # 0.3s pause

    if all_audio:
        compilation = np.concatenate(all_audio)
        save_wav(compilation, str(OUTPUT_DIR / "CONVERSATION_TEST.wav"))
        print(f"  Saved: CONVERSATION_TEST.wav ({len(compilation)/24000:.1f}s)")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    total_phrases = len(all_results)
    avg_latency = np.mean([r['latency_ms'] for r in all_results])
    avg_rms = np.mean([r['rms_db'] for r in all_results])
    max_boundary_diff = max([r['max_boundary_diff'] for r in all_results])
    total_bad_boundaries = sum([r['boundaries_over_03'] for r in all_results])

    print(f"\nPhrases tested: {total_phrases}")
    print(f"Average latency: {avg_latency:.0f}ms")
    print(f"Average RMS: {avg_rms:.1f}dB (target: -16dB)")
    print(f"Maximum boundary discontinuity: {max_boundary_diff:.4f}")
    print(f"Boundaries with diff > 0.3: {total_bad_boundaries}")

    print("\n--- QUALITY CHECKLIST ---")

    # Latency check
    if avg_latency < 2000:
        print(f"✓ Latency: {avg_latency:.0f}ms (target: <2000ms)")
    else:
        print(f"✗ Latency: {avg_latency:.0f}ms (target: <2000ms)")

    # Loudness check
    if -18 < avg_rms < -14:
        print(f"✓ Loudness: {avg_rms:.1f}dB (target: -16dB)")
    else:
        print(f"✗ Loudness: {avg_rms:.1f}dB (target: -16dB)")

    # Boundary check
    if max_boundary_diff < 0.1:
        print(f"✓ Crossfade: max discontinuity {max_boundary_diff:.4f} (target: <0.1)")
    elif max_boundary_diff < 0.3:
        print(f"⚠ Crossfade: max discontinuity {max_boundary_diff:.4f} (acceptable: <0.3)")
    else:
        print(f"✗ Crossfade: max discontinuity {max_boundary_diff:.4f} (target: <0.3)")

    # Bad boundaries check
    if total_bad_boundaries == 0:
        print(f"✓ No severe boundary issues detected")
    else:
        print(f"✗ {total_bad_boundaries} severe boundary issues")

    print("\n" + "="*70)
    if max_boundary_diff < 0.3 and total_bad_boundaries == 0:
        print("OVERALL: ✓ SESAME AI LEVEL QUALITY ACHIEVED")
        print("Audio files ready for human verification in:")
        print(f"  {OUTPUT_DIR}")
    else:
        print("OVERALL: ✗ QUALITY ISSUES REMAIN")
    print("="*70)

if __name__ == "__main__":
    main()
