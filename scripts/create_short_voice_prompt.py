#!/usr/bin/env python3
"""Create a short (2-3s) voice prompt from the long one.

The fine-tuned model has Maya's voice identity baked into its weights,
so we only need a SHORT voice prompt for tone/style priming.
This avoids wasting the 2048 context window on a 158s prompt.
"""
import torch
import os

SR = 24000
LONG_PATH = "/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.pt"
SHORT_PATH = "/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt_short.pt"

# Load the long prompt
data = torch.load(LONG_PATH, weights_only=False)
audio = data["audio"]
if audio.dim() > 1:
    audio = audio.squeeze()

full_duration = audio.size(0) / SR
print(f"Original prompt: {full_duration:.1f}s, {audio.size(0)} samples")

# Extract first 2 seconds - this contains the first utterance
# "oh hey! yeah im doing pretty good"
target_seconds = 2.0
target_samples = int(target_seconds * SR)
short_audio = audio[:target_samples]

# Apply gentle fade-out to prevent click
fade_samples = int(0.02 * SR)  # 20ms fade
if short_audio.size(0) > fade_samples:
    fade = torch.linspace(1.0, 0.0, fade_samples)
    short_audio[-fade_samples:] = short_audio[-fade_samples:] * fade

# Matching text for the first ~2s of the prompt
short_text = "oh hey! yeah im doing pretty good"

short_data = {
    "audio": short_audio,
    "text": short_text,
    "sample_rate": SR,
    "duration_seconds": short_audio.size(0) / SR,
    "version": "short_for_finetuned_v1",
    "description": "Short voice prompt for fine-tuned CSM model",
}

torch.save(short_data, SHORT_PATH)

print(f"Short prompt saved: {short_audio.size(0)/SR:.1f}s, {short_audio.size(0)} samples")
print(f"Text: '{short_text}'")
print(f"Path: {SHORT_PATH}")
