#!/usr/bin/env python3
"""
Deep Crossfade Diagnostic - Find the exact issue
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import wave
from pathlib import Path

OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/audio_quality_test")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_wav(audio: np.ndarray, filepath: str, sample_rate: int = 24000):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    print(f"Saved: {filepath}")

def test_crossfade_implementation():
    """Test the crossfade function directly with synthetic audio."""
    print("\n" + "="*70)
    print("TEST: Crossfade Implementation with Synthetic Audio")
    print("="*70)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Create synthetic test signals
    sr = 24000
    freq = 440  # Hz
    duration = 0.5  # seconds

    # Create two sine waves at slightly different phases
    t1 = np.linspace(0, duration, int(sr * duration))
    t2 = np.linspace(duration, duration * 2, int(sr * duration))

    chunk1 = torch.tensor(np.sin(2 * np.pi * freq * t1), dtype=torch.float32, device='cuda')
    chunk2 = torch.tensor(np.sin(2 * np.pi * freq * t2), dtype=torch.float32, device='cuda')

    print(f"\nChunk 1: {len(chunk1)} samples, sine wave at {freq}Hz")
    print(f"Chunk 2: {len(chunk2)} samples, sine wave at {freq}Hz (phase continued)")

    # Initialize TTS just to use its crossfade function
    tts = RealStreamingTTSEngine()

    # Pre-compute crossfade tensors (mimicking initialization)
    crossfade_samples = 240
    tts.CROSSFADE_SAMPLES = crossfade_samples
    tts._crossfade_t = torch.linspace(0, 1, crossfade_samples, device="cuda")
    tts._crossfade_fade_out = torch.cos(tts._crossfade_t * 3.14159 / 2)
    tts._crossfade_fade_in = torch.sin(tts._crossfade_t * 3.14159 / 2)
    tts._initialized = True  # Skip full init

    # Test crossfade
    print("\n--- Testing Crossfade ---")

    # First chunk (no previous tail)
    output1, tail1 = tts._crossfade_chunks(None, chunk1)
    print(f"Chunk 1 output: {len(output1)} samples, tail: {len(tail1)} samples")

    # Second chunk (with previous tail)
    output2, tail2 = tts._crossfade_chunks(tail1, chunk2)
    print(f"Chunk 2 output: {len(output2)} samples, tail: {len(tail2)} samples")

    # Concatenate
    full_audio = torch.cat([output1, output2]).cpu().numpy()
    full_audio = np.concatenate([full_audio, tail2.cpu().numpy()])  # Add final tail

    print(f"\nFull audio: {len(full_audio)} samples")

    # Also create a reference without crossfade (just concatenate)
    ref_no_crossfade = np.concatenate([
        chunk1.cpu().numpy(),
        chunk2.cpu().numpy()
    ])

    # Check for discontinuity at crossfade point
    crossfade_point = len(output1.cpu().numpy())
    region_before = full_audio[crossfade_point-50:crossfade_point]
    region_after = full_audio[crossfade_point:crossfade_point+50]

    print(f"\nAt crossfade point {crossfade_point}:")
    print(f"  Last sample before: {full_audio[crossfade_point-1]:.6f}")
    print(f"  First sample after: {full_audio[crossfade_point]:.6f}")
    print(f"  Difference: {abs(full_audio[crossfade_point] - full_audio[crossfade_point-1]):.6f}")

    # Check the original boundary without crossfade
    orig_boundary = len(chunk1.cpu().numpy())
    print(f"\nAt original boundary (no crossfade):")
    print(f"  Last sample chunk1: {ref_no_crossfade[orig_boundary-1]:.6f}")
    print(f"  First sample chunk2: {ref_no_crossfade[orig_boundary]:.6f}")
    print(f"  Difference: {abs(ref_no_crossfade[orig_boundary] - ref_no_crossfade[orig_boundary-1]):.6f}")

    # Save both versions
    save_wav(full_audio, str(OUTPUT_DIR / "synthetic_with_crossfade.wav"))
    save_wav(ref_no_crossfade, str(OUTPUT_DIR / "synthetic_no_crossfade.wav"))

    # Analyze for clicks
    diff_crossfade = np.abs(np.diff(full_audio))
    diff_no_crossfade = np.abs(np.diff(ref_no_crossfade))

    max_diff_crossfade = np.max(diff_crossfade)
    max_diff_no_crossfade = np.max(diff_no_crossfade)

    print(f"\nMax sample difference:")
    print(f"  With crossfade: {max_diff_crossfade:.6f}")
    print(f"  No crossfade: {max_diff_no_crossfade:.6f}")

    if max_diff_crossfade < max_diff_no_crossfade:
        print("  ✓ Crossfade improved continuity")
    else:
        print("  ✗ Crossfade did NOT improve continuity")

def test_real_tts_chunks():
    """Test with real TTS output, examining chunk boundaries in detail."""
    print("\n" + "="*70)
    print("TEST: Real TTS Chunk Boundaries")
    print("="*70)

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    tts = RealStreamingTTSEngine()
    tts.initialize()

    phrase = "hello how are you"

    print(f"\nGenerating: '{phrase}'")

    # Collect chunks and their raw versions
    chunks_raw = []
    chunks_processed = []
    chunk_info = []

    # Hook into the generation to capture intermediate states
    original_decode = tts._decode_frames

    frame_batches = []

    def capture_decode(frames, use_streaming_ctx=False):
        result = original_decode(frames, use_streaming_ctx)
        frame_batches.append(result.clone())
        return result

    tts._decode_frames = capture_decode

    # Generate
    for i, chunk in enumerate(tts.generate_stream(phrase, use_context=False)):
        chunk_np = chunk.cpu().numpy()
        chunks_processed.append(chunk_np)
        chunk_info.append({
            'index': i,
            'samples': len(chunk_np),
            'rms': np.sqrt(np.mean(chunk_np**2)),
            'peak': np.max(np.abs(chunk_np)),
            'start_5': chunk_np[:5].tolist(),
            'end_5': chunk_np[-5:].tolist()
        })

    print(f"\nGenerated {len(chunks_processed)} chunks")

    # Analyze boundaries
    print("\n--- Boundary Analysis ---")

    for i in range(1, len(chunks_processed)):
        prev_chunk = chunks_processed[i-1]
        curr_chunk = chunks_processed[i]

        # Boundary samples
        last_sample = prev_chunk[-1]
        first_sample = curr_chunk[0]
        diff = abs(first_sample - last_sample)

        print(f"\nBoundary {i-1}->{i}:")
        print(f"  Prev last 3: {prev_chunk[-3:]}")
        print(f"  Curr first 3: {curr_chunk[:3]}")
        print(f"  Discontinuity: {diff:.6f}")

        if diff > 0.1:
            print(f"  ⚠ Significant discontinuity!")
        elif diff > 0.3:
            print(f"  ✗ CLICK likely!")
        else:
            print(f"  ✓ Smooth transition")

    # Concatenate and save
    full_audio = np.concatenate(chunks_processed)
    save_wav(full_audio, str(OUTPUT_DIR / "real_tts_test.wav"))

    # Find all discontinuities > 0.1
    diff = np.abs(np.diff(full_audio))
    large_diffs = np.where(diff > 0.1)[0]
    print(f"\nSample positions with diff > 0.1: {len(large_diffs)}")
    for pos in large_diffs[:10]:
        print(f"  Position {pos}: diff={diff[pos]:.4f}, before={full_audio[pos]:.4f}, after={full_audio[pos+1]:.4f}")

def test_normalization_consistency():
    """Check if normalization is causing level mismatches."""
    print("\n" + "="*70)
    print("TEST: Normalization Consistency Across Chunks")
    print("="*70)

    from maya.engine.tts_streaming_real import _normalize_lufs

    # Create synthetic chunks with different energy levels
    sr = 24000
    duration = 0.5

    t = np.linspace(0, duration, int(sr * duration))

    # Low energy chunk
    chunk_low = torch.tensor(np.sin(2 * np.pi * 440 * t) * 0.1, dtype=torch.float32)

    # High energy chunk
    chunk_high = torch.tensor(np.sin(2 * np.pi * 440 * t) * 0.5, dtype=torch.float32)

    print(f"\nBefore normalization:")
    print(f"  Low chunk RMS: {torch.sqrt(torch.mean(chunk_low**2)):.4f}")
    print(f"  High chunk RMS: {torch.sqrt(torch.mean(chunk_high**2)):.4f}")

    # Normalize both
    chunk_low_norm = _normalize_lufs(chunk_low, target_lufs=-16.0)
    chunk_high_norm = _normalize_lufs(chunk_high, target_lufs=-16.0)

    print(f"\nAfter normalization (target -16 LUFS):")
    print(f"  Low chunk RMS: {torch.sqrt(torch.mean(chunk_low_norm**2)):.4f}")
    print(f"  High chunk RMS: {torch.sqrt(torch.mean(chunk_high_norm**2)):.4f}")

    # Check boundary if we concatenate
    last_low = chunk_low_norm[-1].item()
    first_high = chunk_high_norm[0].item()

    print(f"\nIf concatenated at boundary:")
    print(f"  Last sample of low: {last_low:.6f}")
    print(f"  First sample of high: {first_high:.6f}")
    print(f"  Discontinuity: {abs(first_high - last_low):.6f}")

    # This shows the issue: different chunks normalized independently
    # will have level jumps at boundaries

if __name__ == "__main__":
    try:
        test_crossfade_implementation()
        test_real_tts_chunks()
        test_normalization_consistency()

        print("\n" + "="*70)
        print("DIAGNOSTIC COMPLETE")
        print(f"Audio files in: {OUTPUT_DIR}")
        print("="*70)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
