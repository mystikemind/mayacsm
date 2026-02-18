#!/usr/bin/env python3
"""
Generate audio samples for quality evaluation.

Generates diverse samples to evaluate:
1. Voice consistency
2. Natural prosody
3. Emotional expression
4. Clarity and intelligibility
"""

import sys
import time
import torch
import torchaudio

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from pathlib import Path

print("=" * 70)
print("AUDIO QUALITY EVALUATION")
print("Generating samples for human evaluation")
print("=" * 70)

# Initialize TTS
print("\n[1] Initializing TTS engine...")
from maya.engine.tts_compiled import CompiledTTSEngine

tts = CompiledTTSEngine()
tts.initialize()
print("    TTS ready")

# Create output directory
output_dir = Path("/home/ec2-user/SageMaker/project_maya/tests/outputs/audio_samples")
output_dir.mkdir(parents=True, exist_ok=True)

# Test phrases - covering different scenarios
test_phrases = [
    # Greetings
    ("greeting_1", "Hi there!"),
    ("greeting_2", "Hello, how are you?"),
    ("greeting_3", "Hey, nice to meet you!"),

    # Short responses (typical conversational)
    ("short_1", "I'm doing great!"),
    ("short_2", "That sounds fun!"),
    ("short_3", "Tell me more!"),
    ("short_4", "Interesting!"),

    # Medium responses
    ("medium_1", "I think that's a really good idea."),
    ("medium_2", "The weather is beautiful today."),
    ("medium_3", "I'd love to hear more about that."),

    # Emotional content
    ("happy", "I'm so happy to hear that!"),
    ("curious", "Really? How does that work?"),
    ("surprised", "Wow, that's amazing!"),
    ("thoughtful", "Hmm, let me think about that."),

    # Longer phrases (for quality testing)
    ("long_1", "It's really nice talking with you today."),
    ("long_2", "That's a wonderful story, thank you for sharing."),
]

print(f"\n[2] Generating {len(test_phrases)} audio samples...")
print("-" * 70)

results = []

for name, text in test_phrases:
    print(f"\n  Generating: '{text}'")

    start = time.time()
    audio = tts.generate(text, use_context=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    duration = len(audio) / 24000
    rtf = elapsed / duration if duration > 0 else 0

    # Save audio
    output_path = output_dir / f"{name}.wav"
    audio_cpu = audio.cpu().unsqueeze(0)  # Add channel dimension
    torchaudio.save(str(output_path), audio_cpu, 24000)

    results.append({
        'name': name,
        'text': text,
        'duration': duration,
        'gen_time': elapsed,
        'rtf': rtf,
        'path': output_path
    })

    print(f"    Saved: {output_path.name} ({duration:.2f}s, RTF={rtf:.2f}x)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

avg_rtf = sum(r['rtf'] for r in results) / len(results)
total_audio = sum(r['duration'] for r in results)
total_gen = sum(r['gen_time'] for r in results)

print(f"""
  Samples generated: {len(results)}
  Total audio:       {total_audio:.1f}s
  Total gen time:    {total_gen:.1f}s
  Average RTF:       {avg_rtf:.2f}x

  Output directory: {output_dir}
""")

print("\n[3] Audio samples saved for evaluation:")
for r in results:
    print(f"    {r['name']}: '{r['text'][:40]}...' ({r['duration']:.1f}s)")

print("\n" + "=" * 70)
print("Listen to samples in: " + str(output_dir))
print("=" * 70)
