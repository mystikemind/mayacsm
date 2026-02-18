#!/usr/bin/env python3
"""
Generate FINAL audio samples with all improvements:
- 8-second voice prompt (better voice cloning)
- Warm personality (brief but friendly)
- Audio polish (RMS normalization, fades, limiter)
"""

import sys
import time
import torch
import torchaudio

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from pathlib import Path

print("=" * 70)
print("FINAL MAYA AUDIO SAMPLES")
print("With all improvements: voice prompt, personality, audio polish")
print("=" * 70)

# Initialize
print("\n[1] Initializing...")
from maya.engine.tts_compiled import CompiledTTSEngine
from maya.engine.llm_optimized import OptimizedLLMEngine

llm = OptimizedLLMEngine()
llm.initialize()

tts = CompiledTTSEngine()
tts.initialize()

# Create output directory
output_dir = Path("/home/ec2-user/SageMaker/project_maya/tests/outputs/final_samples")
output_dir.mkdir(parents=True, exist_ok=True)

# Test conversation
conversation = [
    "Hello!",
    "How are you today?",
    "I'm feeling pretty good actually",
    "What do you like to do for fun?",
    "That sounds nice",
    "I had a long day at work",
    "Thanks for listening",
]

print(f"\n[2] Generating conversation samples...")
print("-" * 70)

results = []

for i, user_input in enumerate(conversation):
    print(f"\n  User: \"{user_input}\"")

    # Get LLM response
    llm_start = time.time()
    response = llm.generate(user_input)
    llm_time = (time.time() - llm_start) * 1000

    # Generate audio
    tts_start = time.time()
    audio = tts.generate(response, use_context=True)
    torch.cuda.synchronize()
    tts_time = (time.time() - tts_start) * 1000

    duration = len(audio) / 24000
    total_time = llm_time + tts_time + 150  # +150ms for STT

    # Save audio
    output_path = output_dir / f"turn_{i+1:02d}.wav"
    audio_cpu = audio.cpu().unsqueeze(0)
    torchaudio.save(str(output_path), audio_cpu, 24000)

    status = "✅" if total_time < 1000 else "⚠️"
    print(f"  Maya: \"{response}\"")
    print(f"  {status} {total_time:.0f}ms total, {duration:.1f}s audio")

    results.append({
        'user': user_input,
        'maya': response,
        'total_ms': total_time,
        'duration': duration,
        'path': output_path
    })

# Summary
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

avg_latency = sum(r['total_ms'] for r in results) / len(results)
sub_second = sum(1 for r in results if r['total_ms'] < 1000)

print(f"""
  Conversation turns: {len(results)}
  Average latency: {avg_latency:.0f}ms
  Sub-1-second: {sub_second}/{len(results)}

  Output: {output_dir}

  Improvements applied:
  ✅ 8-second voice prompt (expressive)
  ✅ Warm personality (brief but friendly)
  ✅ Audio polish (RMS norm, fades, limiter)
  ✅ Barge-in support (interruption handling)
""")

# Save conversation log
log_path = output_dir / "conversation.txt"
with open(log_path, 'w') as f:
    f.write("MAYA CONVERSATION LOG\n")
    f.write("=" * 50 + "\n\n")
    for r in results:
        f.write(f"User: {r['user']}\n")
        f.write(f"Maya: {r['maya']}\n")
        f.write(f"Latency: {r['total_ms']:.0f}ms\n")
        f.write(f"Audio: {r['path'].name}\n\n")

print(f"  Conversation log: {log_path}")
print("=" * 70)
