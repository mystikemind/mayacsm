#!/usr/bin/env python3
"""
Test aggressive high-frequency enhancement to fix Mimi codec limitations.

The Mimi codec loses 90% of 8-12kHz content and 45% of 4-8kHz content.
We need aggressive spectral synthesis to restore the "air" and "presence"
that makes speech sound natural vs "old telephone".
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import wave
from pathlib import Path

SAMPLE_RATE = 24000
OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/audio_enhancement_test")
OUTPUT_DIR.mkdir(exist_ok=True)


def enhance_v1_current(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """Current enhancement (from tts_streaming_real.py) - for comparison."""
    nyq = sr / 2

    # Current harmonic excitation
    b_mid, a_mid = signal.butter(4, [500 / nyq, 2000 / nyq], btype='band')
    mid_content = signal.filtfilt(b_mid, a_mid, audio)
    harmonics = np.tanh(mid_content * 3) / 3
    b_hp, a_hp = signal.butter(2, 3000 / nyq, btype='high')
    high_harmonics = signal.filtfilt(b_hp, a_hp, harmonics)
    audio = audio + 0.3 * high_harmonics

    # Presence boost
    b_pres, a_pres = signal.butter(2, [5000 / nyq, 8000 / nyq], btype='band')
    presence = signal.filtfilt(b_pres, a_pres, audio)
    audio = audio + 0.2 * presence

    return audio


def enhance_v2_aggressive(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """
    Aggressive high-frequency restoration using multi-band excitation.

    Based on professional aural exciter technology (like Aphex Aural Exciter):
    1. Multiple harmonic generation bands
    2. Psychoacoustic enhancement at critical frequencies
    3. High-shelf boost with harmonic richness
    """
    nyq = sr / 2
    audio = audio.astype(np.float64)

    # === BAND 1: Low-Mid Harmonics (1-2kHz source -> 3-6kHz harmonics) ===
    b1, a1 = signal.butter(4, [1000 / nyq, 2000 / nyq], btype='band')
    band1 = signal.filtfilt(b1, a1, audio)

    # Generate harmonics via asymmetric clipping (creates even + odd harmonics)
    # This is more natural than symmetric soft-clip which only creates odd harmonics
    harmonics1 = np.tanh(band1 * 4) * 0.5 + np.tanh(band1 * 2) * 0.25

    # Extract just the high frequency harmonics (3-6kHz)
    b1_hp, a1_hp = signal.butter(3, 3000 / nyq, btype='high')
    harmonics1_hp = signal.filtfilt(b1_hp, a1_hp, harmonics1)

    # === BAND 2: Mid Harmonics (2-3kHz source -> 5-9kHz harmonics) ===
    b2, a2 = signal.butter(4, [2000 / nyq, 3000 / nyq], btype='band')
    band2 = signal.filtfilt(b2, a2, audio)

    # More aggressive harmonic generation for higher frequencies
    harmonics2 = np.tanh(band2 * 5) * 0.4

    # Extract 5-9kHz range
    b2_hp, a2_hp = signal.butter(3, 5000 / nyq, btype='high')
    harmonics2_hp = signal.filtfilt(b2_hp, a2_hp, harmonics2)

    # === BAND 3: Presence enhancement (3-4kHz source -> 7-11kHz) ===
    b3, a3 = signal.butter(4, [3000 / nyq, 4000 / nyq], btype='band')
    band3 = signal.filtfilt(b3, a3, audio)

    # Very aggressive for the "air" frequencies
    harmonics3 = np.tanh(band3 * 6) * 0.3

    b3_hp, a3_hp = signal.butter(3, 7000 / nyq, btype='high')
    harmonics3_hp = signal.filtfilt(b3_hp, a3_hp, harmonics3)

    # === HIGH-SHELF BOOST ===
    # Apply a gentle high-shelf starting at 4kHz to lift the entire high end
    # Using a 2nd order shelf approximation
    sos_shelf = signal.butter(2, 4000 / nyq, btype='high', output='sos')
    high_content = signal.sosfilt(sos_shelf, audio)

    # === MIX ===
    # Aggressive mixing levels to compensate for Mimi's 90% loss
    enhanced = (
        audio +
        0.5 * harmonics1_hp +   # 3-6kHz: 50% harmonic addition
        0.6 * harmonics2_hp +   # 5-9kHz: 60% harmonic addition
        0.5 * harmonics3_hp +   # 7-11kHz: 50% harmonic addition
        0.3 * high_content      # High shelf: 30% boost
    )

    # === DE-ESSER (prevent harsh sibilance) ===
    # Don't let 5-8kHz get TOO harsh
    b_de, a_de = signal.butter(2, [5000 / nyq, 8000 / nyq], btype='band')
    sibilance = signal.filtfilt(b_de, a_de, enhanced)
    sib_env = np.abs(signal.hilbert(sibilance))

    # If sibilance > 0.3, compress it
    sib_threshold = 0.3
    sib_ratio = 3.0
    mask = sib_env > sib_threshold
    if np.any(mask):
        over = sib_env[mask] - sib_threshold
        compressed = sib_threshold + over / sib_ratio
        gain = np.ones_like(enhanced)
        gain[mask] = compressed / (sib_env[mask] + 1e-10)
        enhanced = enhanced * gain

    # Normalize to prevent clipping
    peak = np.max(np.abs(enhanced))
    if peak > 0.95:
        enhanced = enhanced * (0.95 / peak)

    return enhanced.astype(np.float32)


def enhance_v3_spectral(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """
    Spectral exciter using FFT-based processing.

    This approach:
    1. Analyzes the spectral envelope in low-mid frequencies
    2. Extrapolates the envelope to high frequencies
    3. Synthesizes harmonically-rich content to fill the gap
    """
    nyq = sr / 2
    audio = audio.astype(np.float64)

    # Use STFT for time-varying processing
    nperseg = 512  # ~21ms at 24kHz
    hop = nperseg // 4

    # STFT
    f, t, Zxx = signal.stft(audio, sr, nperseg=nperseg, noverlap=nperseg-hop)

    # For each frame, enhance high frequencies
    enhanced_Zxx = Zxx.copy()

    for i in range(Zxx.shape[1]):
        frame = Zxx[:, i]
        mag = np.abs(frame)
        phase = np.angle(frame)

        # Find spectral slope in 1-3kHz region
        low_idx = int(1000 / (sr / nperseg))
        mid_idx = int(3000 / (sr / nperseg))
        high_start_idx = int(4000 / (sr / nperseg))

        if mid_idx > low_idx and np.mean(mag[low_idx:mid_idx]) > 1e-6:
            # Estimate spectral slope
            low_energy = np.mean(mag[low_idx:low_idx + 10] ** 2)
            mid_energy = np.mean(mag[mid_idx-10:mid_idx] ** 2)

            if low_energy > 1e-12:
                slope = mid_energy / low_energy

                # Extrapolate to high frequencies
                for j in range(high_start_idx, len(mag)):
                    freq = f[j]
                    # Calculate expected energy based on spectral slope
                    expected = mid_energy * (slope ** ((j - mid_idx) / (mid_idx - low_idx)))
                    current = mag[j] ** 2

                    # If current energy is too low, boost it
                    if current < expected * 0.5:
                        boost = np.sqrt(expected * 0.8) / (mag[j] + 1e-10)
                        boost = min(boost, 5.0)  # Limit boost to 5x
                        enhanced_Zxx[j, i] = frame[j] * boost

    # Inverse STFT
    _, enhanced = signal.istft(enhanced_Zxx, sr, nperseg=nperseg, noverlap=nperseg-hop)

    # Match length
    if len(enhanced) < len(audio):
        enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
    elif len(enhanced) > len(audio):
        enhanced = enhanced[:len(audio)]

    # Add warmth with subtle saturation
    enhanced = np.tanh(enhanced * 1.1) / 1.1

    # Normalize
    peak = np.max(np.abs(enhanced))
    if peak > 0.95:
        enhanced = enhanced * (0.95 / peak)

    return enhanced.astype(np.float32)


def enhance_v4_hybrid(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """
    Hybrid approach: spectral + harmonic exciter.

    Combines the best of both approaches:
    1. Time-domain harmonic generation for transient preservation
    2. Spectral enhancement for smooth frequency extension
    """
    # First apply harmonic exciter
    harmonic = enhance_v2_aggressive(audio, sr)

    # Then apply spectral enhancement (gentler)
    nyq = sr / 2

    # High-shelf EQ to lift brightness
    sos = signal.butter(2, 6000 / nyq, btype='high', output='sos')
    brightness = signal.sosfilt(sos, harmonic)

    # Mix with original enhanced
    enhanced = harmonic + 0.2 * brightness

    # Final de-essing
    b_de, a_de = signal.butter(2, [6000 / nyq, 9000 / nyq], btype='band')
    sibilance = signal.filtfilt(b_de, a_de, enhanced)
    sib_rms = np.sqrt(np.mean(sibilance ** 2))

    if sib_rms > 0.15:
        # Reduce sibilance band
        enhanced = enhanced - 0.3 * sibilance

    # Normalize
    peak = np.max(np.abs(enhanced))
    if peak > 0.95:
        enhanced = enhanced * (0.95 / peak)

    return enhanced.astype(np.float32)


def analyze_frequencies(audio: np.ndarray, sr: int, label: str):
    """Analyze frequency distribution."""
    spectrum = np.abs(fft.rfft(audio))
    freqs = fft.rfftfreq(len(audio), 1/sr)

    def band_pct(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.sum(spectrum[mask]**2) / np.sum(spectrum**2) * 100

    low = band_pct(0, 2000)
    mid = band_pct(2000, 4000)
    high = band_pct(4000, 8000)
    very_high = band_pct(8000, 12000)

    print(f"\n{label}:")
    print(f"  Low (0-2kHz): {low:.1f}%")
    print(f"  Mid (2-4kHz): {mid:.1f}%")
    print(f"  High (4-8kHz): {high:.1f}%")
    print(f"  Very High (8-12kHz): {very_high:.1f}%")

    return high, very_high


def save_wav(audio: np.ndarray, filepath: str, sample_rate: int = 24000):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def main():
    print("=" * 70)
    print("AGGRESSIVE HIGH-FREQUENCY ENHANCEMENT TEST")
    print("Goal: Restore frequencies lost by Mimi codec (90% loss at 8-12kHz)")
    print("=" * 70)

    # Generate test audio from TTS
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Initialize without enhancement to get raw codec output
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Generate test phrases
    test_phrases = [
        "hello how are you doing today",
        "thats a really interesting question",
        "let me think about that for a moment"
    ]

    for i, phrase in enumerate(test_phrases):
        print(f"\n{'='*70}")
        print(f"PHRASE {i+1}: '{phrase}'")
        print("=" * 70)

        # Generate raw audio (without enhancement)
        # We need to temporarily disable enhancement
        chunks = []
        for chunk in tts.generate_stream(phrase, use_context=False):
            chunks.append(chunk.cpu().numpy())

        if not chunks:
            print("  ERROR: No audio generated")
            continue

        raw_audio = np.concatenate(chunks)

        # Analyze raw (this has current v1 enhancement baked in)
        print("\nWith CURRENT enhancement (v1):")
        h1, vh1 = analyze_frequencies(raw_audio, SAMPLE_RATE, "Current (v1)")
        save_wav(raw_audio, str(OUTPUT_DIR / f"phrase{i+1}_v1_current.wav"))

        # Test V2: Aggressive harmonic
        audio_v2 = enhance_v2_aggressive(raw_audio, SAMPLE_RATE)
        h2, vh2 = analyze_frequencies(audio_v2, SAMPLE_RATE, "V2 Aggressive Harmonic")
        save_wav(audio_v2, str(OUTPUT_DIR / f"phrase{i+1}_v2_aggressive.wav"))

        # Test V3: Spectral
        audio_v3 = enhance_v3_spectral(raw_audio, SAMPLE_RATE)
        h3, vh3 = analyze_frequencies(audio_v3, SAMPLE_RATE, "V3 Spectral")
        save_wav(audio_v3, str(OUTPUT_DIR / f"phrase{i+1}_v3_spectral.wav"))

        # Test V4: Hybrid
        audio_v4 = enhance_v4_hybrid(raw_audio, SAMPLE_RATE)
        h4, vh4 = analyze_frequencies(audio_v4, SAMPLE_RATE, "V4 Hybrid")
        save_wav(audio_v4, str(OUTPUT_DIR / f"phrase{i+1}_v4_hybrid.wav"))

        print(f"\n  HIGH FREQ (4-8kHz) COMPARISON:")
        print(f"    Current: {h1:.1f}% | V2: {h2:.1f}% | V3: {h3:.1f}% | V4: {h4:.1f}%")
        print(f"  VERY HIGH (8-12kHz) COMPARISON:")
        print(f"    Current: {vh1:.1f}% | V2: {vh2:.1f}% | V3: {vh3:.1f}% | V4: {vh4:.1f}%")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"Audio files saved to: {OUTPUT_DIR}")
    print("Listen to compare quality - V2/V4 should sound brighter and clearer")
    print("=" * 70)


if __name__ == "__main__":
    main()
