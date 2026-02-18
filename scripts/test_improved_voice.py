#!/usr/bin/env python3
"""
Test the improved voice prompt (8 seconds vs 2 seconds).

Generates comparison samples to evaluate voice quality improvement.
"""

import sys
import time
import torch
import torchaudio

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from pathlib import Path

print("=" * 70)
print("TESTING IMPROVED VOICE PROMPT")
print("8-second expressive prompt for better voice cloning")
print("=" * 70)

# Check voice prompt
print("\n[1] Checking voice prompt...")
data = torch.load('/home/ec2-user/SageMaker/project_maya/assets/voice_prompt/maya_voice_prompt.pt')
print(f"    Duration: {len(data['audio']) / 24000:.2f}s")
print(f"    Text: {data['text'][:60]}...")

# Initialize TTS (will use new voice prompt)
print("\n[2] Initializing TTS with new voice prompt...")
from maya.engine.tts_compiled import CompiledTTSEngine

tts = CompiledTTSEngine()
tts.initialize()

# Create output directory
output_dir = Path("/home/ec2-user/SageMaker/project_maya/tests/outputs/improved_voice")
output_dir.mkdir(parents=True, exist_ok=True)

# Test phrases - same as before for comparison
test_phrases = [
    ("greeting", "Hi there!"),
    ("happy", "I'm so happy to hear that!"),
    ("curious", "Really? How does that work?"),
    ("thoughtful", "Hmm, let me think about that."),
    ("warm", "That's really sweet of you to say."),
    ("excited", "Oh wow, that sounds amazing!"),
    ("empathetic", "I understand how you feel."),
    ("conversational", "So what have you been up to lately?"),
]

print(f"\n[3] Generating {len(test_phrases)} samples with improved voice...")
print("-" * 70)

results = []

for name, text in test_phrases:
    print(f"\n  '{text}'")

    start = time.time()
    audio = tts.generate(text, use_context=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    duration = len(audio) / 24000
    rtf = elapsed / duration if duration > 0 else 0

    # Save audio
    output_path = output_dir / f"{name}.wav"
    audio_cpu = audio.cpu().unsqueeze(0)
    torchaudio.save(str(output_path), audio_cpu, 24000)

    results.append({
        'name': name,
        'text': text,
        'duration': duration,
        'rtf': rtf,
    })

    print(f"    -> {output_path.name} ({duration:.2f}s, RTF={rtf:.2f}x)")

# Summary
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

avg_rtf = sum(r['rtf'] for r in results) / len(results)

print(f"""
  Voice Prompt: 8.08s (improved from 1.92s)
  Samples Generated: {len(results)}
  Average RTF: {avg_rtf:.2f}x

  Output: {output_dir}

  COMPARE these samples with the previous ones in:
  /home/ec2-user/SageMaker/project_maya/tests/outputs/audio_samples/

  Listen for:
  - Voice consistency (does it sound like the same person?)
  - Naturalness (does it sound human?)
  - Emotional expression (can you hear the emotion?)
  - Clarity (is the speech clear and intelligible?)
""")

print("=" * 70)
