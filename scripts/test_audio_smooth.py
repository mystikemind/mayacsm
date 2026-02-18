"""
Quick test to verify fade-in/fade-out works correctly.
"""

import torch
import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

from maya.engine.tts_compiled import CompiledTTSEngine
import torchaudio

print("=" * 60)
print("TESTING AUDIO SMOOTHING")
print("=" * 60)

# Initialize TTS
print("\nInitializing TTS engine...")
tts = CompiledTTSEngine()
tts.initialize()

# Generate test audio
print("\nGenerating test audio: 'Hello, nice to meet you!'")
audio = tts.generate("Hello, nice to meet you!")

# Check audio properties
duration = len(audio) / tts.SAMPLE_RATE
print(f"\nAudio duration: {duration:.2f}s")
print(f"Audio samples: {len(audio)}")

# Check fade-in (first 30ms should ramp up)
fade_in_samples = int(tts.SAMPLE_RATE * 0.03)
first_samples = audio[:fade_in_samples].abs()
print(f"\nFade-in check (first 30ms):")
print(f"  First sample amplitude: {first_samples[0].item():.6f}")
print(f"  Mid fade amplitude: {first_samples[fade_in_samples//2].item():.6f}")
print(f"  End fade amplitude: {first_samples[-1].item():.6f}")

# Check that audio isn't clipped or broken
max_amp = audio.abs().max().item()
mean_amp = audio.abs().mean().item()
print(f"\nAudio quality check:")
print(f"  Max amplitude: {max_amp:.4f} (should be < 1.0)")
print(f"  Mean amplitude: {mean_amp:.4f} (should be > 0.01)")

# Save test audio
output_path = "/tmp/test_smooth_audio.wav"
torchaudio.save(output_path, audio.cpu().unsqueeze(0), tts.SAMPLE_RATE)
print(f"\nTest audio saved to: {output_path}")

# Verdict
print("\n" + "=" * 60)
if max_amp < 1.0 and mean_amp > 0.005:
    print("✓ AUDIO QUALITY: GOOD")
else:
    print("✗ AUDIO QUALITY: POTENTIAL ISSUE")

if first_samples[0].item() < first_samples[-1].item():
    print("✓ FADE-IN: WORKING")
else:
    print("✗ FADE-IN: MAY NOT BE WORKING")

print("=" * 60)
