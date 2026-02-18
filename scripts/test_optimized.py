#!/usr/bin/env python3
"""Test optimized pipeline with short LLM responses."""

import sys
import os

os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import time

print("=" * 70)
print("OPTIMIZED PIPELINE TEST")
print("(Short LLM responses for faster TTS)")
print("=" * 70)

# Test 1: LLM with new short prompt
print("\n[1] Testing optimized LLM...")
from maya.engine import LLMEngine

llm = LLMEngine()
llm.initialize()

test_inputs = [
    "Hello, how are you?",
    "What's the weather like?",
    "I'm feeling sad today.",
    "Tell me a joke.",
]

for inp in test_inputs:
    start = time.time()
    response = llm.generate(inp)
    elapsed = (time.time() - start) * 1000
    words = len(response.split())
    print(f"  '{inp}' -> {elapsed:.0f}ms, {words} words")
    print(f"    Response: '{response}'")

# Test 2: TTS with short responses
print("\n[2] Testing TTS with short responses...")
from maya.engine import TTSEngine

tts = TTSEngine()
tts.initialize()

# Warmup
_ = tts.generate("Hi", use_context=False)

short_responses = [
    "Oh, I'm doing great! How about you?",
    "Hmm, I'm not sure, but it looks nice out!",
    "Aw, I'm sorry to hear that.",
    "Ha! Here's one: Why did the coffee file a police report? It got mugged!"
]

for text in short_responses:
    start = time.time()
    audio = tts.generate(text, use_context=False)
    elapsed = (time.time() - start) * 1000
    duration = len(audio) / 24000
    rtf = elapsed / 1000 / duration
    print(f"  '{text[:30]}...' -> {elapsed:.0f}ms ({duration:.1f}s audio, RTF={rtf:.2f}x)")

# Test 3: Full pipeline simulation
print("\n[3] Full pipeline simulation with fillers...")
from maya.engine import STTEngine
from maya.conversation.natural_fillers import NaturalFillerSystem

stt = STTEngine()
stt.initialize()

fillers = NaturalFillerSystem()
fillers.initialize()

# Simulate user input (3 seconds)
test_audio = torch.randn(24000 * 3)

print("\nSimulating full turn:")
total_start = time.time()

# Filler (instant)
filler_start = time.time()
filler, name = fillers.get_thinking_filler()
filler_time = (time.time() - filler_start) * 1000
filler_duration = len(filler) / 24000 * 1000
print(f"  [Filler] Ready in {filler_time:.2f}ms, plays for {filler_duration:.0f}ms ('{name}')")

# STT
stt_start = time.time()
transcript = stt.transcribe(test_audio)
stt_time = (time.time() - stt_start) * 1000
print(f"  [STT] {stt_time:.0f}ms -> '{transcript}'")

# LLM
llm_start = time.time()
response = llm.generate("Hello there!")  # Use fixed input for consistency
llm_time = (time.time() - llm_start) * 1000
words = len(response.split())
print(f"  [LLM] {llm_time:.0f}ms -> '{response}' ({words} words)")

# TTS
tts_start = time.time()
response_audio = tts.generate(response, use_context=False)
tts_time = (time.time() - tts_start) * 1000
audio_duration = len(response_audio) / 24000 * 1000
print(f"  [TTS] {tts_time:.0f}ms -> {audio_duration:.0f}ms audio")

total_time = (time.time() - total_start) * 1000

print(f"\n" + "=" * 70)
print("LATENCY ANALYSIS")
print("=" * 70)
print(f"""
Breakdown:
  - Filler ready:  {filler_time:.2f}ms
  - STT:           {stt_time:.0f}ms
  - LLM:           {llm_time:.0f}ms
  - TTS:           {tts_time:.0f}ms
  - TOTAL:         {total_time:.0f}ms

User Experience:
  - t=0ms:    User stops speaking
  - t=~1ms:   Filler starts ("Hmm...", "Yeah...")
  - t={filler_duration:.0f}ms: Filler ends
  - t={total_time:.0f}ms: Response audio ready

Gap Analysis:
  - Filler plays for: {filler_duration:.0f}ms
  - Processing takes: {total_time:.0f}ms
  - Gap after filler: {total_time - filler_duration:.0f}ms
""")

if total_time - filler_duration > 0:
    num_extra_fillers = int((total_time - filler_duration) / 600)  # ~600ms per filler
    print(f"  Will need ~{num_extra_fillers + 1} fillers to cover the wait")
else:
    print(f"  Response ready before filler ends!")

print(f"\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
