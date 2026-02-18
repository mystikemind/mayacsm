#!/usr/bin/env python3
"""
Test using the OFFICIAL CSM Generator class.
If this works, we know CSM itself is fine.
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import torchaudio

print("=" * 70)
print("TESTING OFFICIAL CSM GENERATOR")
print("=" * 70)

from generator import load_csm_1b, Segment

print("\nLoading CSM-1B via official generator...")
generator = load_csm_1b(device="cuda")

print("\nGenerating audio...")

# Test with empty context first
text = "hi how are you doing today"
print(f"Text: '{text}'")

audio = generator.generate(
    text=text,
    speaker=0,
    context=[],  # No context
    max_audio_length_ms=10000,
    temperature=0.9,
    topk=50
)

# Save
output_path = "/home/ec2-user/SageMaker/project_maya/tests/outputs/quality_test/official_generator.wav"
torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)

duration = len(audio) / generator.sample_rate
print(f"\nSaved: official_generator.wav ({duration:.1f}s)")
print("=" * 70)
