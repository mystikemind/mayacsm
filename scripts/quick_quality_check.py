#!/usr/bin/env python3
"""Quick quality check of new audio files."""

import numpy as np
import wave
import scipy.signal as signal
import scipy.fft as fft
from pathlib import Path

def analyze(filepath):
    with wave.open(filepath, 'r') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        sr = wav.getframerate()

    print(f"\n{Path(filepath).name}:")
    print(f"  Duration: {len(audio)/sr:.2f}s")

    # Frequency content
    spectrum = np.abs(fft.rfft(audio))
    freqs = fft.rfftfreq(len(audio), 1/sr)

    def band_pct(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.sum(spectrum[mask]**2) / np.sum(spectrum**2) * 100

    high = band_pct(4000, 8000)
    very_high = band_pct(8000, 12000)

    print(f"  High freq (4-8kHz): {high:.1f}%")
    print(f"  Very high (8-12kHz): {very_high:.1f}%")

    if high < 1:
        print("  ⚠ Still muffled (low high frequencies)")
    else:
        print("  ✓ Good high frequency content")

    # Noise floor
    frame_size = int(sr * 0.02)
    energies = [np.sqrt(np.mean(audio[i:i+frame_size]**2))
                for i in range(0, len(audio)-frame_size, frame_size)]
    quiet = np.percentile(energies, 10) * 1.5
    quiet_frames = [audio[i:i+frame_size] for i, e in enumerate(energies)
                    if e < quiet]
    if quiet_frames:
        noise_rms = np.sqrt(np.mean(np.concatenate(quiet_frames)**2))
        noise_db = 20 * np.log10(noise_rms) if noise_rms > 1e-10 else -100
        print(f"  Noise floor: {noise_db:.1f}dB")
        if noise_db > -40:
            print("  ⚠ High background noise")
        else:
            print("  ✓ Good noise floor")

audio_dir = Path("/home/ec2-user/SageMaker/project_maya/audio_final_test")
for f in sorted(audio_dir.glob("test_*.wav")):
    analyze(str(f))
