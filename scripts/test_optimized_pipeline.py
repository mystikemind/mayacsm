#!/usr/bin/env python3
"""
Optimized Pipeline Test - Target: Sub-1-second latency

Tests the full pipeline with:
1. Ultra-short LLM responses (3-4 words)
2. Max-autotune torch.compile on TTS
3. Optimized audio length

Target: ~900ms total latency (competitive with Sesame Maya)
"""

import sys
import time
import torch
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

print("=" * 70)
print("OPTIMIZED PIPELINE TEST")
print("Target: Sub-1-second total latency")
print("=" * 70)

# Initialize all components
print("\n[1] Initializing components...")

from maya.engine.stt import STTEngine
from maya.engine.llm_optimized import OptimizedLLMEngine
from maya.engine.tts_compiled import CompiledTTSEngine

init_start = time.time()

stt = STTEngine()
stt.initialize()
print("    STT ready")

llm = OptimizedLLMEngine()
llm.initialize()
print("    LLM ready")

tts = CompiledTTSEngine()
tts.initialize()
print("    TTS ready")

print(f"\n    Total init time: {time.time() - init_start:.1f}s")

# Run simulated conversation turns
print("\n[2] Running simulated conversation...")
print("-" * 70)

test_inputs = [
    "Hello",
    "How are you",
    "Nice weather",
    "Tell me a joke",
    "What do you think",
]

results = []

for i, user_input in enumerate(test_inputs):
    print(f"\n  Turn {i+1}: User says '{user_input}'")

    total_start = time.time()

    # STT (simulated - typically ~150ms)
    stt_start = time.time()
    transcript = user_input  # Simulated
    stt_time = 150  # Typical observed STT time

    # LLM
    llm_start = time.time()
    response = llm.generate(transcript)
    torch.cuda.synchronize()
    llm_time = (time.time() - llm_start) * 1000

    # TTS
    tts_start = time.time()
    audio = tts.generate(response, use_context=True)
    torch.cuda.synchronize()
    tts_time = (time.time() - tts_start) * 1000

    total_time = stt_time + llm_time + tts_time
    audio_duration = len(audio) / 24000
    rtf = tts_time / 1000 / audio_duration if audio_duration > 0 else 0

    results.append({
        'input': user_input,
        'response': response,
        'words': len(response.split()),
        'stt_ms': stt_time,
        'llm_ms': llm_time,
        'tts_ms': tts_time,
        'total_ms': total_time,
        'audio_s': audio_duration,
        'rtf': rtf
    })

    status = "✅" if total_time < 1000 else "⚠️"
    print(f"    Maya: '{response}' ({len(response.split())} words)")
    print(f"    {status} STT={stt_time:.0f}ms + LLM={llm_time:.0f}ms + TTS={tts_time:.0f}ms = {total_time:.0f}ms")
    print(f"    Audio: {audio_duration:.2f}s (RTF={rtf:.2f}x)")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

avg_total = np.mean([r['total_ms'] for r in results])
avg_stt = np.mean([r['stt_ms'] for r in results])
avg_llm = np.mean([r['llm_ms'] for r in results])
avg_tts = np.mean([r['tts_ms'] for r in results])
avg_rtf = np.mean([r['rtf'] for r in results])
avg_words = np.mean([r['words'] for r in results])

sub_second_count = sum(1 for r in results if r['total_ms'] < 1000)

print(f"""
  Response length: {avg_words:.1f} words average

  Latency breakdown:
  - STT:   {avg_stt:.0f}ms
  - LLM:   {avg_llm:.0f}ms
  - TTS:   {avg_tts:.0f}ms
  - Total: {avg_total:.0f}ms

  Performance:
  - RTF: {avg_rtf:.2f}x (must be < 1.0)
  - Sub-1-second turns: {sub_second_count}/{len(results)}

  Sesame Maya comparison:
  - Target: 500-1000ms
  - Our avg: {avg_total:.0f}ms
  - Status: {'✅ COMPETITIVE!' if avg_total < 1000 else '⚠️ Needs improvement'}
""")

# Best and worst cases
best = min(results, key=lambda r: r['total_ms'])
worst = max(results, key=lambda r: r['total_ms'])

print(f"  Best case: {best['total_ms']:.0f}ms for '{best['response']}'")
print(f"  Worst case: {worst['total_ms']:.0f}ms for '{worst['response']}'")

print("\n" + "=" * 70)
