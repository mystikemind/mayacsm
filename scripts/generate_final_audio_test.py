#!/usr/bin/env python3
"""
Generate comprehensive audio samples for final human verification.
Tests various phrase lengths, natural speech patterns, and edge cases.
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import wave
import time
from pathlib import Path

OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/audio_final_test")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_wav(audio: np.ndarray, filepath: str, sample_rate: int = 24000):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return filepath

def analyze_audio(audio: np.ndarray) -> dict:
    """Quick audio quality check."""
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    diff = np.abs(np.diff(audio))
    max_diff = np.max(diff)
    clicks_03 = np.sum(diff > 0.3)
    clicks_05 = np.sum(diff > 0.5)

    return {
        'rms_db': 20 * np.log10(rms) if rms > 0 else -100,
        'peak_db': 20 * np.log10(peak) if peak > 0 else -100,
        'max_diff': max_diff,
        'clicks_03': clicks_03,
        'clicks_05': clicks_05
    }

def main():
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    print("="*70)
    print("FINAL AUDIO QUALITY TEST")
    print("Generating samples for human verification")
    print("="*70)

    # Initialize TTS
    print("\nLoading TTS engine...")
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Test cases - diverse phrases to test various aspects
    test_cases = [
        # Quick responses
        ("01_quick_hi", "hi"),
        ("02_quick_yes", "yeah definitely"),
        ("03_quick_hmm", "hmm interesting"),

        # Natural conversational
        ("04_natural_greeting", "oh hey whats up"),
        ("05_natural_question", "so how are you doing today"),
        ("06_natural_response", "thats really cool tell me more"),

        # With fillers and pauses
        ("07_thinking", "hmm let me think about that"),
        ("08_filler", "oh yeah well i guess that makes sense"),
        ("09_hesitation", "umm i mean you know its kinda complicated"),

        # Longer responses
        ("10_longer", "oh thats a great question i think it really depends on what youre looking for"),
        ("11_story", "so the other day i was thinking about how things have changed over the years"),

        # Edge cases - plosives and fricatives
        ("12_plosives", "pick a perfect purple pepper"),
        ("13_sibilants", "she sells seashells by the seashore"),

        # Emotional content
        ("14_excited", "oh wow thats amazing i love it"),
        ("15_gentle", "aww im sorry to hear that how are you feeling"),
    ]

    all_audio = []
    results = []

    for name, phrase in test_cases:
        print(f"\n--- {name}: '{phrase}' ---")

        start = time.time()
        chunks = list(tts.generate_stream(phrase, use_context=False))
        elapsed_ms = (time.time() - start) * 1000

        if not chunks:
            print(f"  ERROR: No audio generated!")
            continue

        audio = np.concatenate([c.cpu().numpy() for c in chunks])
        duration_ms = len(audio) / 24000 * 1000

        # Analyze
        stats = analyze_audio(audio)

        print(f"  Generated: {duration_ms:.0f}ms audio in {elapsed_ms:.0f}ms")
        print(f"  Chunks: {len(chunks)}, Sizes: {[len(c) for c in chunks]}")
        print(f"  RMS: {stats['rms_db']:.1f}dB, Peak: {stats['peak_db']:.1f}dB")
        print(f"  Max sample diff: {stats['max_diff']:.4f}")
        print(f"  Clicks (>0.3): {stats['clicks_03']}, Clicks (>0.5): {stats['clicks_05']}")

        # Save individual file
        filepath = save_wav(audio, str(OUTPUT_DIR / f"{name}.wav"))

        # Add to compilation
        all_audio.append(audio)
        # Add 0.5s silence between clips
        all_audio.append(np.zeros(int(24000 * 0.5)))

        results.append({
            'name': name,
            'phrase': phrase,
            'duration_ms': duration_ms,
            'latency_ms': elapsed_ms,
            **stats
        })

    # Create compilation
    print("\n" + "="*70)
    print("Creating compilation...")
    compilation = np.concatenate(all_audio)
    comp_path = save_wav(compilation, str(OUTPUT_DIR / "00_FULL_COMPILATION.wav"))
    print(f"Compilation: {len(compilation)/24000:.1f}s, saved to {comp_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_clicks_03 = sum(r['clicks_03'] for r in results)
    total_clicks_05 = sum(r['clicks_05'] for r in results)
    avg_latency = np.mean([r['latency_ms'] for r in results])

    print(f"\nTotal samples: {len(results)}")
    print(f"Average latency: {avg_latency:.0f}ms")
    print(f"Total clicks (>0.3): {total_clicks_03}")
    print(f"Total clicks (>0.5): {total_clicks_05}")

    if total_clicks_05 == 0:
        print("\n✓ NO severe clicks detected (>0.5 threshold)")
    else:
        print(f"\n✗ {total_clicks_05} severe clicks detected")

    # Check all RMS levels are consistent
    rms_values = [r['rms_db'] for r in results]
    rms_std = np.std(rms_values)
    print(f"\nRMS consistency (std): {rms_std:.2f}dB")
    if rms_std < 2.0:
        print("✓ Consistent loudness across samples")
    else:
        print("✗ Loudness variation too high")

    print("\n" + "="*70)
    print(f"Audio files saved to: {OUTPUT_DIR}")
    print("Please listen to verify quality!")
    print("="*70)

if __name__ == "__main__":
    main()
