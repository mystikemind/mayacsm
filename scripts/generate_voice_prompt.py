"""
Generate a high-quality voice prompt for Maya.

This creates a "golden" reference sample that establishes Maya's
speaking style, warmth, and cadence for all future generations.
"""

import torch
import sys
import os

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from generator import load_csm_1b

# Create output directory
os.makedirs('/home/ec2-user/SageMaker/project_maya/assets/voice_prompt', exist_ok=True)

print("Loading CSM-1B...")
generator = load_csm_1b(device="cuda")

# The voice prompt text - warm, natural, conversational
# This sets the tone for all of Maya's responses
VOICE_PROMPT_TEXT = "Hi there! I'm Maya. It's so nice to meet you. I love having conversations and getting to know people. What's on your mind today?"

print(f"\nGenerating voice prompt:")
print(f"Text: {VOICE_PROMPT_TEXT}")
print()

# Generate with optimal settings for quality
# Lower temperature for consistency, higher topk for naturalness
audio = generator.generate(
    text=VOICE_PROMPT_TEXT,
    speaker=0,
    context=[],  # No context for the reference
    max_audio_length_ms=15000,  # Allow up to 15 seconds
    temperature=0.8,  # Slightly lower for consistency
    topk=80,  # Slightly higher for naturalness
)

# Save the audio
output_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.pt'
torch.save({
    'audio': audio.cpu(),
    'text': VOICE_PROMPT_TEXT,
    'sample_rate': 24000,
    'settings': {
        'temperature': 0.8,
        'topk': 80,
        'speaker': 0,
    }
}, output_path)

print(f"Voice prompt saved to: {output_path}")
print(f"Audio duration: {len(audio) / 24000:.2f} seconds")
print(f"Audio samples: {len(audio)}")

# Also save as WAV for listening
import torchaudio
wav_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.wav'
torchaudio.save(wav_path, audio.cpu().unsqueeze(0), 24000)
print(f"WAV saved to: {wav_path}")

print("\nDone! Use this as permanent context for natural-sounding responses.")
