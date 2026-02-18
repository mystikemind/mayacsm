"""
Generate a BETTER voice prompt for Maya.

Key improvements:
1. Longer sample (8-10 seconds) - more voice characteristics
2. Expressive text - shows emotional range
3. Multiple takes - pick the best one
4. Natural punctuation - teaches CSM pacing

This gives CSM a better "voice anchor" for consistent, natural speech.
"""

import torch
import sys
import os
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from generator import load_csm_1b
import torchaudio

os.makedirs('/home/ec2-user/SageMaker/project_maya/assets/voice_prompt', exist_ok=True)

print("=" * 60)
print("GENERATING BETTER VOICE PROMPT FOR MAYA")
print("=" * 60)

print("\nLoading CSM-1B...")
generator = load_csm_1b(device="cuda")

# Expressive text that shows Maya's personality range:
# - Warm greeting
# - Natural pauses (commas)
# - Emotional variation
# - Conversational flow
MAYA_TEXT = """Hi there, it's really nice to meet you! I'm Maya. I love having conversations and learning about people. So, tell me, what's been on your mind lately? I'm all ears."""

print(f"\nVoice prompt text:")
print(f'"{MAYA_TEXT}"')
print(f"\nExpected duration: 8-10 seconds")
print()

# Generate multiple takes, pick the best
best_audio = None
best_score = -1

for i in range(5):
    print(f"Generating take {i+1}/5...")
    start = time.time()

    audio = generator.generate(
        text=MAYA_TEXT,
        speaker=0,
        context=[],
        max_audio_length_ms=15000,  # Allow up to 15 seconds
        temperature=0.9,  # Standard setting
        topk=50,  # Standard setting
    )

    elapsed = time.time() - start
    duration = len(audio) / 24000
    energy = audio.abs().mean().item()

    # Score based on:
    # 1. Duration in sweet spot (7-11 seconds for this text)
    # 2. Good energy (not too quiet, not too loud)
    # 3. Not too short (indicates problems)
    score = 0

    if 7.0 < duration < 11.0:
        score += 20  # Perfect duration
    elif 6.0 < duration < 12.0:
        score += 10  # Acceptable

    if 0.015 < energy < 0.06:
        score += 10  # Good energy
    elif 0.01 < energy < 0.08:
        score += 5   # Acceptable

    print(f"  Duration: {duration:.2f}s, Energy: {energy:.4f}, Score: {score}, Time: {elapsed:.1f}s")

    if score > best_score:
        best_score = score
        best_audio = audio.clone()
        print(f"  ^ New best!")

print(f"\n{'=' * 60}")
print(f"Best take: Score {best_score}, Duration {len(best_audio)/24000:.2f}s")

# Save
output_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.pt'
torch.save({
    'audio': best_audio.cpu(),
    'text': MAYA_TEXT,
    'sample_rate': 24000,
    'settings': {
        'temperature': 0.9,
        'topk': 50,
        'speaker': 0,
    }
}, output_path)
print(f"\nSaved to: {output_path}")

# Also save WAV
wav_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.wav'
torchaudio.save(wav_path, best_audio.cpu().unsqueeze(0), 24000)
print(f"WAV saved to: {wav_path}")

print(f"\n{'=' * 60}")
print("DONE - Better voice prompt ready!")
print("Restart server to use the new voice prompt.")
print("=" * 60)
