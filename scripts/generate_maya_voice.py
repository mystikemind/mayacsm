"""
Generate Maya's voice prompt FROM CSM itself.

This creates a "golden sample" that:
1. Stays in CSM's natural voice space (no external audio)
2. Establishes Maya's warm, friendly personality
3. Uses conversational text that sets the tone

The key insight: CSM generates consistent voices when given
the SAME context. By generating a high-quality sample and
using it as permanent context, we get voice consistency.
"""

import torch
import sys
import os
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from generator import load_csm_1b
import torchaudio

# Output directory
os.makedirs('/home/ec2-user/SageMaker/project_maya/assets/voice_prompt', exist_ok=True)

print("=" * 60)
print("GENERATING MAYA'S VOICE PROMPT")
print("=" * 60)

print("\nLoading CSM-1B...")
generator = load_csm_1b(device="cuda")

# Maya's personality-defining text
# This sets the warm, friendly, engaging tone
MAYA_TEXT = "Hey there! It's so nice to talk with you. I'm really curious to hear what's on your mind today."

print(f"\nGenerating voice with text:")
print(f'"{MAYA_TEXT}"')
print()

# Generate multiple samples and pick the best one
# (CSM can have variation, we want a good baseline)
best_audio = None
best_score = -1

for i in range(3):
    print(f"Generating sample {i+1}/3...")
    start = time.time()

    audio = generator.generate(
        text=MAYA_TEXT,
        speaker=0,
        context=[],  # No context = fresh voice
        max_audio_length_ms=10000,
        temperature=0.7,  # Lower for consistency
        topk=50,
    )

    elapsed = time.time() - start
    duration = len(audio) / 24000

    # Simple quality heuristic: prefer longer audio (more natural pacing)
    # and avoid extremely short or long generations
    score = 0
    if 2.5 < duration < 5.0:  # Good range for this text
        score = 10
    elif 2.0 < duration < 6.0:
        score = 5

    # Check for audio energy (avoid silent/bad generations)
    energy = audio.abs().mean().item()
    if energy > 0.01:
        score += 5

    print(f"  Duration: {duration:.2f}s, Energy: {energy:.4f}, Score: {score}, Time: {elapsed:.1f}s")

    if score > best_score:
        best_score = score
        best_audio = audio.clone()

print(f"\nBest sample score: {best_score}")

# Save the voice prompt
output_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.pt'
torch.save({
    'audio': best_audio.cpu(),
    'text': MAYA_TEXT,
    'sample_rate': 24000,
    'settings': {
        'temperature': 0.7,
        'topk': 50,
        'speaker': 0,
    }
}, output_path)

print(f"\nVoice prompt saved to: {output_path}")
print(f"Duration: {len(best_audio) / 24000:.2f}s")

# Also save as WAV for listening
wav_path = '/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.wav'
torchaudio.save(wav_path, best_audio.cpu().unsqueeze(0), 24000)
print(f"WAV saved to: {wav_path}")

print("\n" + "=" * 60)
print("DONE - Maya's voice is ready!")
print("=" * 60)
