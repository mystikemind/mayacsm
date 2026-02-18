"""
Generate a HUMAN-LIKE voice prompt for Maya.

Goal: Sound like a real person talking, not reading.

Key elements of natural human speech:
1. Contractions (I'm, it's, you're)
2. Filler patterns ("you know", "I mean", "like")
3. Natural pauses (commas, ellipses)
4. Emotional warmth
5. Varied sentence structure
6. Questions with rising intonation
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
print("GENERATING HUMAN-LIKE VOICE PROMPT")
print("=" * 60)

print("\nLoading CSM-1B...")
generator = load_csm_1b(device="cuda")

# Natural, conversational text with:
# - Warmth and personality
# - Natural pauses (commas, ...)
# - Filler words that humans use
# - Emotional variation
# - Not "reading" - actually talking
MAYA_TEXT = """Oh hey, hi! It's really nice to meet you. I'm Maya. You know, I just love talking to people and, like, hearing what's going on in their lives. So... tell me, what's on your mind?"""

print(f"\nVoice prompt text (natural, conversational):")
print(f'"{MAYA_TEXT}"')
print()

# Generate multiple takes
best_audio = None
best_score = -1

for i in range(5):
    print(f"Generating take {i+1}/5...")
    start = time.time()

    audio = generator.generate(
        text=MAYA_TEXT,
        speaker=0,
        context=[],
        max_audio_length_ms=15000,
        temperature=0.9,
        topk=50,
    )

    elapsed = time.time() - start
    duration = len(audio) / 24000
    energy = audio.abs().mean().item()

    # Score based on natural duration and good energy
    score = 0

    # This text should take 6-10 seconds if spoken naturally
    if 6.0 < duration < 10.0:
        score += 20
    elif 5.0 < duration < 11.0:
        score += 10

    # Good energy range
    if 0.015 < energy < 0.05:
        score += 10
    elif 0.01 < energy < 0.07:
        score += 5

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

# Save WAV for listening
wav_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.wav'
torchaudio.save(wav_path, best_audio.cpu().unsqueeze(0), 24000)
print(f"WAV saved to: {wav_path}")

print(f"\n{'=' * 60}")
print("DONE - Human-like voice prompt ready!")
print("=" * 60)
