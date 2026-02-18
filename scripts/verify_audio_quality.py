#!/usr/bin/env python3
"""
BRUTAL Audio Quality Verification - Sesame AI Level Check

This script generates audio and performs comprehensive quality checks:
1. Click/pop detection at chunk boundaries
2. Sample duplication detection (the bug we just fixed)
3. Audio continuity analysis
4. Waveform visualization
5. Actually saves audio files for human listening

NO SHORTCUTS. Test everything.
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import time
import os
from pathlib import Path
import wave
import struct

# Create output directory
OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/audio_quality_test")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_wav(audio: np.ndarray, filepath: str, sample_rate: int = 24000):
    """Save audio as WAV file for human listening."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"  Saved: {filepath}")

def detect_clicks(audio: np.ndarray, threshold: float = 0.3) -> list:
    """Detect clicks/pops in audio by looking for sudden jumps."""
    diff = np.abs(np.diff(audio))
    click_indices = np.where(diff > threshold)[0]
    return click_indices.tolist()

def detect_sample_duplication(audio: np.ndarray, window_size: int = 240) -> list:
    """Detect if any audio segments are duplicated (the bug we fixed)."""
    duplications = []

    # Check for near-identical consecutive windows
    for i in range(0, len(audio) - window_size * 2, window_size):
        window1 = audio[i:i+window_size]
        window2 = audio[i+window_size:i+window_size*2]

        # Correlation check
        if len(window1) == len(window2):
            correlation = np.corrcoef(window1, window2)[0, 1]
            if not np.isnan(correlation) and correlation > 0.95:
                duplications.append((i, i + window_size, correlation))

    return duplications

def analyze_audio_continuity(audio: np.ndarray, chunk_size_samples: int = 15360) -> dict:
    """Analyze audio continuity at expected chunk boundaries."""
    results = {
        "boundary_discontinuities": [],
        "dc_offset_jumps": [],
        "energy_drops": []
    }

    # Check at expected chunk boundaries
    for boundary in range(chunk_size_samples, len(audio) - 100, chunk_size_samples):
        # Get samples around boundary
        before = audio[boundary-100:boundary]
        after = audio[boundary:boundary+100]

        # Check for DC offset jump
        dc_before = np.mean(before)
        dc_after = np.mean(after)
        dc_jump = abs(dc_after - dc_before)
        if dc_jump > 0.05:
            results["dc_offset_jumps"].append((boundary, dc_jump))

        # Check for sample discontinuity (click)
        sample_before = audio[boundary-1]
        sample_after = audio[boundary]
        discontinuity = abs(sample_after - sample_before)
        if discontinuity > 0.3:
            results["boundary_discontinuities"].append((boundary, discontinuity))

        # Check for energy drop (silence at boundary)
        energy_before = np.sqrt(np.mean(before**2))
        energy_after = np.sqrt(np.mean(after**2))
        if energy_before > 0.05 and energy_after < 0.01:
            results["energy_drops"].append((boundary, energy_before, energy_after))

    return results

def test_streaming_chunks():
    """Test that streaming chunks are properly crossfaded."""
    print("\n" + "="*70)
    print("TEST: Streaming Chunk Crossfade Verification")
    print("="*70)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Initialize TTS
    print("\nInitializing TTS engine...")
    tts = RealStreamingTTSEngine()
    tts.initialize()

    test_phrases = [
        ("short_hi", "hi there"),
        ("medium_question", "how are you doing today"),
        ("long_response", "thats really interesting tell me more about what happened next"),
        ("with_pauses", "hmm let me think about that for a moment"),
    ]

    all_results = {}

    for name, phrase in test_phrases:
        print(f"\n--- Testing: '{phrase}' ---")

        # Generate with streaming
        chunks = []
        chunk_boundaries = [0]

        start = time.time()
        for chunk in tts.generate_stream(phrase, use_context=False):
            chunk_np = chunk.cpu().numpy()
            chunks.append(chunk_np)
            chunk_boundaries.append(chunk_boundaries[-1] + len(chunk_np))
        elapsed = (time.time() - start) * 1000

        if not chunks:
            print(f"  ERROR: No audio generated!")
            continue

        # Concatenate all chunks
        full_audio = np.concatenate(chunks)
        duration_ms = len(full_audio) / 24000 * 1000

        print(f"  Generated {len(chunks)} chunks, {duration_ms:.0f}ms audio in {elapsed:.0f}ms")
        print(f"  Chunk sizes: {[len(c) for c in chunks]}")

        # Save audio for listening
        wav_path = str(OUTPUT_DIR / f"{name}.wav")
        save_wav(full_audio, wav_path)

        # Run quality checks
        print(f"\n  Quality Analysis:")

        # 1. Click detection
        clicks = detect_clicks(full_audio)
        click_at_boundaries = []
        for click in clicks:
            for boundary in chunk_boundaries[1:-1]:  # Skip start and end
                if abs(click - boundary) < 50:  # Within 50 samples of boundary
                    click_at_boundaries.append((click, boundary))
                    break

        print(f"    Clicks detected: {len(clicks)}")
        print(f"    Clicks at chunk boundaries: {len(click_at_boundaries)}")
        if click_at_boundaries:
            print(f"      WARNING: Clicks at boundaries indicate crossfade issue!")
            for click, boundary in click_at_boundaries[:5]:
                print(f"        Click at {click}, boundary at {boundary}")

        # 2. Duplication detection
        duplications = detect_sample_duplication(full_audio)
        print(f"    Sample duplications detected: {len(duplications)}")
        if duplications:
            print(f"      WARNING: Sample duplication indicates the bug is back!")
            for start, end, corr in duplications[:3]:
                print(f"        Samples {start}-{end} duplicated (correlation: {corr:.3f})")

        # 3. Continuity analysis
        continuity = analyze_audio_continuity(full_audio)
        print(f"    DC offset jumps: {len(continuity['dc_offset_jumps'])}")
        print(f"    Boundary discontinuities: {len(continuity['boundary_discontinuities'])}")
        print(f"    Energy drops: {len(continuity['energy_drops'])}")

        # 4. Overall audio quality
        rms = np.sqrt(np.mean(full_audio**2))
        peak = np.max(np.abs(full_audio))
        crest_factor = peak / rms if rms > 0 else 0

        print(f"\n  Audio Metrics:")
        print(f"    RMS: {rms:.4f} ({20*np.log10(rms):.1f} dBFS)")
        print(f"    Peak: {peak:.4f} ({20*np.log10(peak):.1f} dBFS)")
        print(f"    Crest factor: {crest_factor:.2f}")

        # Store results
        all_results[name] = {
            "duration_ms": duration_ms,
            "latency_ms": elapsed,
            "num_chunks": len(chunks),
            "clicks_total": len(clicks),
            "clicks_at_boundaries": len(click_at_boundaries),
            "duplications": len(duplications),
            "dc_jumps": len(continuity["dc_offset_jumps"]),
            "discontinuities": len(continuity["boundary_discontinuities"]),
            "energy_drops": len(continuity["energy_drops"]),
            "rms": rms,
            "peak": peak
        }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_boundary_clicks = sum(r["clicks_at_boundaries"] for r in all_results.values())
    total_duplications = sum(r["duplications"] for r in all_results.values())
    total_discontinuities = sum(r["discontinuities"] for r in all_results.values())

    print(f"\nTotal clicks at chunk boundaries: {total_boundary_clicks}")
    print(f"Total sample duplications: {total_duplications}")
    print(f"Total boundary discontinuities: {total_discontinuities}")

    if total_boundary_clicks == 0 and total_duplications == 0 and total_discontinuities == 0:
        print("\n✓ PASS: No audio artifacts detected!")
        print("  Audio files saved to:", OUTPUT_DIR)
        print("  Please listen to verify quality.")
    else:
        print("\n✗ FAIL: Audio artifacts detected!")
        if total_boundary_clicks > 0:
            print("  - Clicks at chunk boundaries (crossfade issue)")
        if total_duplications > 0:
            print("  - Sample duplications (duplication bug)")
        if total_discontinuities > 0:
            print("  - Boundary discontinuities (stitching issue)")

    return all_results

def test_individual_chunks():
    """Test individual chunk quality (before concatenation)."""
    print("\n" + "="*70)
    print("TEST: Individual Chunk Quality")
    print("="*70)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    tts = RealStreamingTTSEngine()
    tts.initialize()

    phrase = "hello how are you doing today i hope everything is going well"

    print(f"\nGenerating: '{phrase}'")

    chunks = []
    for i, chunk in enumerate(tts.generate_stream(phrase, use_context=False)):
        chunk_np = chunk.cpu().numpy()
        chunks.append(chunk_np)

        # Save individual chunk
        wav_path = str(OUTPUT_DIR / f"chunk_{i:02d}.wav")
        save_wav(chunk_np, wav_path)

        # Analyze chunk
        rms = np.sqrt(np.mean(chunk_np**2))
        peak = np.max(np.abs(chunk_np))
        duration = len(chunk_np) / 24000 * 1000

        # Check for clicks at start/end
        start_diff = np.max(np.abs(np.diff(chunk_np[:50])))
        end_diff = np.max(np.abs(np.diff(chunk_np[-50:])))

        print(f"  Chunk {i}: {duration:.0f}ms, RMS={rms:.3f}, Peak={peak:.3f}")
        print(f"           Start discontinuity: {start_diff:.4f}, End discontinuity: {end_diff:.4f}")

        if start_diff > 0.3:
            print(f"           WARNING: High discontinuity at chunk start!")
        if end_diff > 0.3:
            print(f"           WARNING: High discontinuity at chunk end!")

    # Save concatenated version
    if chunks:
        full_audio = np.concatenate(chunks)
        save_wav(full_audio, str(OUTPUT_DIR / "full_concatenated.wav"))
        print(f"\n  Full audio: {len(full_audio)/24000*1000:.0f}ms")

if __name__ == "__main__":
    print("="*70)
    print("BRUTAL AUDIO QUALITY VERIFICATION")
    print("Sesame AI Level Check")
    print("="*70)

    try:
        # Test 1: Streaming chunk quality
        results = test_streaming_chunks()

        # Test 2: Individual chunk quality
        test_individual_chunks()

        print("\n" + "="*70)
        print(f"Audio files saved to: {OUTPUT_DIR}")
        print("LISTEN TO THEM to verify quality!")
        print("="*70)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
