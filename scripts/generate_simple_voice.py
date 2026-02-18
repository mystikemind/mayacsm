"""
Generate a SIMPLE, CONSISTENT voice prompt for Maya.

Goal: Stability over "human-like" - short, clean, consistent.
"""

import torch
import sys
import os

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from generator import load_csm_1b
import torchaudio

os.makedirs('/home/ec2-user/SageMaker/project_maya/assets/voice_prompt', exist_ok=True)

print("=" * 60)
print("GENERATING SIMPLE, STABLE VOICE PROMPT")
print("=" * 60)

print("\nLoading CSM-1B...")
generator = load_csm_1b(device="cuda")

# Simple, clean text - not too fancy, just establishes voice
MAYA_TEXT = "Hi, I'm Maya. It's nice to meet you."

print(f"\nText: \"{MAYA_TEXT}\"")
print("Target: ~3-4 seconds, clean and consistent")

best_audio = None
best_score = -1

for i in range(3):
    print(f"\nGenerating take {i+1}/3...")

    audio = generator.generate(
        text=MAYA_TEXT,
        speaker=0,
        context=[],
        max_audio_length_ms=8000,
        temperature=0.9,
        topk=50,
    )

    duration = len(audio) / 24000
    energy = audio.abs().mean().item()

    # Score: prefer 2.5-4.5 seconds
    score = 0
    if 2.5 < duration < 4.5:
        score = 20
    elif 2.0 < duration < 5.0:
        score = 10

    if 0.01 < energy < 0.06:
        score += 10

    print(f"  Duration: {duration:.2f}s, Energy: {energy:.4f}, Score: {score}")

    if score > best_score:
        best_score = score
        best_audio = audio.clone()
        print("  ^ Best!")

print(f"\n{'=' * 60}")
print(f"Selected: {len(best_audio)/24000:.2f}s")

# Save
output_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.pt'
torch.save({
    'audio': best_audio.cpu(),
    'text': MAYA_TEXT,
    'sample_rate': 24000,
}, output_path)
print(f"Saved: {output_path}")

wav_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.wav'
torchaudio.save(wav_path, best_audio.cpu().unsqueeze(0), 24000)
print(f"WAV: {wav_path}")

print("\nDONE - Simple, stable voice prompt ready.")
