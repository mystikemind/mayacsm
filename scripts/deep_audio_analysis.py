#!/usr/bin/env python3
"""
Deep audio quality analysis - Find root causes of:
1. Robotic sound
2. Background noise
3. Codec artifacts ("ship honking")
4. Old phone quality
"""

import numpy as np
import wave
from pathlib import Path
import scipy.signal as signal
import scipy.fft as fft

def analyze_noise_floor(audio: np.ndarray, sr: int) -> dict:
    """Analyze background noise in quiet sections."""
    # Find quiet sections (low energy)
    frame_size = int(sr * 0.02)  # 20ms frames
    energies = []
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i+frame_size]
        energies.append(np.sqrt(np.mean(frame**2)))

    energies = np.array(energies)
    quiet_threshold = np.percentile(energies, 10)  # Bottom 10% are "quiet"

    # Analyze noise in quiet sections
    quiet_frames = []
    for i, e in enumerate(energies):
        if e < quiet_threshold * 1.5:
            start = i * frame_size
            end = start + frame_size
            quiet_frames.append(audio[start:end])

    if quiet_frames:
        quiet_audio = np.concatenate(quiet_frames)
        noise_rms = np.sqrt(np.mean(quiet_audio**2))
        noise_db = 20 * np.log10(noise_rms) if noise_rms > 1e-10 else -100
    else:
        noise_db = -100

    return {
        'noise_floor_db': noise_db,
        'quiet_threshold': quiet_threshold,
        'num_quiet_frames': len(quiet_frames)
    }

def analyze_frequency_content(audio: np.ndarray, sr: int) -> dict:
    """Analyze frequency content for "old phone" quality issues."""
    # Compute spectrum
    spectrum = np.abs(fft.rfft(audio))
    freqs = fft.rfftfreq(len(audio), 1/sr)

    # Energy in different bands
    def band_energy(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.sum(spectrum[mask]**2)

    total = np.sum(spectrum**2)

    bass = band_energy(0, 300)      # Bass
    low_mid = band_energy(300, 1000)  # Low mids (fundamental)
    mid = band_energy(1000, 4000)    # Mids (clarity)
    high = band_energy(4000, 8000)   # Highs (presence)
    very_high = band_energy(8000, 12000)  # Very high (air)

    return {
        'bass_pct': bass / total * 100 if total > 0 else 0,
        'low_mid_pct': low_mid / total * 100 if total > 0 else 0,
        'mid_pct': mid / total * 100 if total > 0 else 0,
        'high_pct': high / total * 100 if total > 0 else 0,
        'very_high_pct': very_high / total * 100 if total > 0 else 0,
    }

def analyze_codec_artifacts(audio: np.ndarray, sr: int) -> dict:
    """Look for codec artifacts (tonal noise, birdie noise)."""
    # Compute spectrogram
    nperseg = int(sr * 0.02)  # 20ms windows
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=nperseg)

    # Look for constant tones (horizontal lines in spectrogram)
    # These appear as high variance across time at specific frequencies
    freq_consistency = np.std(Sxx, axis=1) / (np.mean(Sxx, axis=1) + 1e-10)

    # Find suspicious constant tones
    suspicious_freqs = f[freq_consistency < 0.5]  # Very consistent = suspicious

    # Look for sudden spectral changes (codec frame boundaries)
    spectral_diff = np.abs(np.diff(Sxx, axis=1))
    mean_spectral_change = np.mean(spectral_diff)
    max_spectral_change = np.max(spectral_diff)

    return {
        'num_suspicious_tones': len(suspicious_freqs),
        'suspicious_freqs_hz': suspicious_freqs[:10].tolist() if len(suspicious_freqs) > 0 else [],
        'mean_spectral_change': mean_spectral_change,
        'max_spectral_change': max_spectral_change,
    }

def analyze_roboticness(audio: np.ndarray, sr: int) -> dict:
    """Analyze factors that make speech sound robotic."""
    # Pitch variation (robotic = monotone)
    # Use autocorrelation to estimate pitch
    frame_size = int(sr * 0.03)  # 30ms
    hop = int(sr * 0.01)  # 10ms

    pitches = []
    for i in range(0, len(audio) - frame_size, hop):
        frame = audio[i:i+frame_size]
        if np.max(np.abs(frame)) < 0.01:  # Skip silence
            continue

        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        # Find first peak after initial decay
        min_lag = int(sr / 500)  # 500 Hz max
        max_lag = int(sr / 50)   # 50 Hz min

        if max_lag < len(corr):
            peak_region = corr[min_lag:max_lag]
            if len(peak_region) > 0:
                peak_idx = np.argmax(peak_region) + min_lag
                if corr[peak_idx] > 0.3 * corr[0]:  # Valid pitch
                    pitch = sr / peak_idx
                    if 50 < pitch < 500:
                        pitches.append(pitch)

    if pitches:
        pitch_std = np.std(pitches)
        pitch_mean = np.mean(pitches)
        pitch_variation = pitch_std / pitch_mean * 100
    else:
        pitch_variation = 0
        pitch_mean = 0

    # Energy variation (robotic = flat dynamics)
    frame_energies = []
    for i in range(0, len(audio) - frame_size, hop):
        frame = audio[i:i+frame_size]
        frame_energies.append(np.sqrt(np.mean(frame**2)))

    if frame_energies:
        energy_variation = np.std(frame_energies) / (np.mean(frame_energies) + 1e-10) * 100
    else:
        energy_variation = 0

    return {
        'pitch_variation_pct': pitch_variation,
        'mean_pitch_hz': pitch_mean,
        'energy_variation_pct': energy_variation,
        'num_pitched_frames': len(pitches)
    }

def analyze_file(filepath: str):
    """Complete analysis of an audio file."""
    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS: {Path(filepath).name}")
    print('='*70)

    with wave.open(filepath, 'r') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        sr = wav.getframerate()

    print(f"Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")

    # Basic stats
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    print(f"\nBasic: RMS={20*np.log10(rms):.1f}dB, Peak={20*np.log10(peak):.1f}dB")

    # Noise floor
    print("\n--- NOISE FLOOR ---")
    noise = analyze_noise_floor(audio, sr)
    print(f"Noise floor: {noise['noise_floor_db']:.1f}dB")
    if noise['noise_floor_db'] > -40:
        print("  ⚠ HIGH NOISE - Should be below -50dB for clean audio")
    elif noise['noise_floor_db'] > -50:
        print("  ⚠ Moderate noise - Could be cleaner")
    else:
        print("  ✓ Good noise floor")

    # Frequency content
    print("\n--- FREQUENCY BALANCE ---")
    freq = analyze_frequency_content(audio, sr)
    print(f"Bass (0-300Hz): {freq['bass_pct']:.1f}%")
    print(f"Low-mid (300-1kHz): {freq['low_mid_pct']:.1f}%")
    print(f"Mid (1-4kHz): {freq['mid_pct']:.1f}%")
    print(f"High (4-8kHz): {freq['high_pct']:.1f}%")
    print(f"Very high (8-12kHz): {freq['very_high_pct']:.1f}%")

    if freq['high_pct'] < 5:
        print("  ⚠ LOW HIGH FREQUENCIES - Will sound muffled/old phone")
    if freq['very_high_pct'] < 1:
        print("  ⚠ NO AIR/PRESENCE - Lacks clarity")

    # Codec artifacts
    print("\n--- CODEC ARTIFACTS ---")
    artifacts = analyze_codec_artifacts(audio, sr)
    print(f"Suspicious constant tones: {artifacts['num_suspicious_tones']}")
    if artifacts['suspicious_freqs_hz']:
        print(f"  Frequencies: {artifacts['suspicious_freqs_hz'][:5]}")
    if artifacts['num_suspicious_tones'] > 20:
        print("  ⚠ MANY TONAL ARTIFACTS - Codec issue")

    # Roboticness
    print("\n--- NATURALNESS ---")
    robot = analyze_roboticness(audio, sr)
    print(f"Pitch variation: {robot['pitch_variation_pct']:.1f}%")
    print(f"Mean pitch: {robot['mean_pitch_hz']:.0f}Hz")
    print(f"Energy variation: {robot['energy_variation_pct']:.1f}%")

    if robot['pitch_variation_pct'] < 10:
        print("  ⚠ LOW PITCH VARIATION - Sounds monotone/robotic")
    if robot['energy_variation_pct'] < 20:
        print("  ⚠ FLAT DYNAMICS - Sounds compressed/robotic")

def main():
    audio_dir = Path("/home/ec2-user/SageMaker/project_maya/audio_final_test")

    # Analyze a few samples
    files = [
        "01_quick_hi.wav",
        "05_natural_question.wav",
        "07_thinking.wav",
        "CONVERSATION_TEST.wav"
    ]

    for f in files:
        path = audio_dir / f
        if path.exists():
            analyze_file(str(path))

    # Also check the voice prompt
    voice_prompt = Path("/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.wav")
    if voice_prompt.exists():
        print("\n" + "="*70)
        print("VOICE PROMPT ANALYSIS (source of voice quality)")
        print("="*70)
        analyze_file(str(voice_prompt))

if __name__ == "__main__":
    main()
