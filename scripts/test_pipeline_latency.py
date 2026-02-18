#!/usr/bin/env python3
"""
END-TO-END PIPELINE LATENCY TEST
Tests actual latency with OPTIMIZED TTS (torch.compile).

Target: <1 second from user stop to first audio
"""

import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import torchaudio
import time
import numpy as np
from pathlib import Path

print("=" * 70)
print("BRUTAL PIPELINE LATENCY TEST")
print("=" * 70)

# Test audio - 3 seconds of speech-like signal
SAMPLE_RATE = 24000
test_audio_path = Path("/home/ec2-user/SageMaker/project_maya/test_audio.wav")

# Load or create test audio
if test_audio_path.exists():
    test_audio, sr = torchaudio.load(str(test_audio_path))
    if sr != SAMPLE_RATE:
        test_audio = torchaudio.functional.resample(test_audio, sr, SAMPLE_RATE)
    test_audio = test_audio.squeeze()
    print(f"Loaded test audio: {len(test_audio)/SAMPLE_RATE:.2f}s")
else:
    # Create synthetic speech-like audio
    duration = 3.0
    t = torch.linspace(0, duration, int(SAMPLE_RATE * duration))
    # Mix of frequencies like speech
    test_audio = (
        0.3 * torch.sin(2 * np.pi * 200 * t) +  # fundamental
        0.2 * torch.sin(2 * np.pi * 400 * t) +  # harmonic
        0.1 * torch.sin(2 * np.pi * 800 * t) +  # harmonic
        0.1 * torch.randn(len(t))               # noise
    )
    test_audio = test_audio / test_audio.abs().max() * 0.8
    print(f"Created synthetic test audio: {duration}s")

results = {}

# ============================================================
# TEST 1: STT (Whisper) Latency
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: STT (Whisper Turbo)")
print("=" * 70)

from maya.engine import STTEngine

stt = STTEngine()
print("Loading STT...")
start = time.time()
stt.initialize()
stt_load_time = time.time() - start
print(f"STT loaded in {stt_load_time:.2f}s")

# Warmup
print("Warming up STT...")
_ = stt.transcribe(test_audio[:SAMPLE_RATE])

# Test transcription
print("Testing transcription speed...")
times = []
for i in range(3):
    start = time.time()
    transcript = stt.transcribe(test_audio)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.0f}ms -> '{transcript}'")

stt_avg = np.mean(times)
results["STT"] = stt_avg
print(f"\nSTT Average: {stt_avg:.0f}ms for {len(test_audio)/SAMPLE_RATE:.1f}s audio")
print(f"STT RTF: {stt_avg/1000 / (len(test_audio)/SAMPLE_RATE):.2f}x")

# ============================================================
# TEST 2: LLM (Llama 3.2 3B) Latency
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: LLM (Llama 3.2 3B)")
print("=" * 70)

from maya.engine import LLMEngine

llm = LLMEngine()
print("Loading LLM...")
start = time.time()
llm.initialize()
llm_load_time = time.time() - start
print(f"LLM loaded in {llm_load_time:.2f}s")

# Warmup
print("Warming up LLM...")
_ = llm.generate("Hello")

# Test generation
test_prompts = [
    "Hello, how are you?",
    "What's the weather like today?",
    "Tell me a short joke.",
]

print("Testing LLM speed...")
times = []
for prompt in test_prompts:
    start = time.time()
    response = llm.generate(prompt)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    print(f"  '{prompt}' -> {elapsed:.0f}ms")
    print(f"    Response: '{response[:60]}...'")

llm_avg = np.mean(times)
results["LLM"] = llm_avg
print(f"\nLLM Average: {llm_avg:.0f}ms")

# ============================================================
# TEST 3: TTS (OPTIMIZED CSM with torch.compile)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: TTS (OPTIMIZED CSM with torch.compile)")
print("=" * 70)

from maya.engine.tts_optimized import OptimizedTTSEngine

tts = OptimizedTTSEngine()
print("Loading TTS (this takes ~2 minutes for compilation)...")
start = time.time()
tts.initialize()
tts_load_time = time.time() - start
print(f"TTS loaded in {tts_load_time:.2f}s")

# Warmup
print("Warming up TTS...")
_ = tts.generate("Hi", use_context=False)

# Test generation
test_texts = [
    "Hello!",
    "How can I help you today?",
    "That's a great question, let me think about it.",
]

print("Testing TTS speed...")
times = []
for text in test_texts:
    start = time.time()
    audio = tts.generate(text, use_context=False)
    elapsed = (time.time() - start) * 1000
    audio_duration = len(audio) / SAMPLE_RATE
    rtf = elapsed / 1000 / audio_duration
    times.append(elapsed)
    print(f"  '{text}' -> {elapsed:.0f}ms ({audio_duration:.2f}s audio, RTF={rtf:.2f}x)")

tts_avg = np.mean(times)
results["TTS"] = tts_avg
print(f"\nTTS Average: {tts_avg:.0f}ms")

# ============================================================
# TEST 4: Natural Fillers
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Natural Fillers (Pre-generated)")
print("=" * 70)

from maya.conversation.natural_fillers import NaturalFillerSystem

fillers = NaturalFillerSystem()
fillers.initialize()

print("Testing filler retrieval speed...")
times = []
for i in range(10):
    start = time.time()
    audio, name = fillers.get_thinking_filler()
    elapsed = (time.time() - start) * 1000000  # microseconds
    times.append(elapsed)

filler_avg = np.mean(times)
results["Filler_retrieval_us"] = filler_avg
print(f"Filler retrieval: {filler_avg:.0f} microseconds (essentially instant)")

# ============================================================
# TEST 5: Full Pipeline Simulation
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: FULL PIPELINE SIMULATION")
print("=" * 70)

print("\nSimulating user speaks for 3s, then stops...")

# Phase 1: User speaks (buffering audio)
print("\n[Phase 1] User speaking... (audio buffered)")

# Phase 2: User stops - measure time to first audio (filler)
start_time = time.time()

# Get filler (instant)
filler_start = time.time()
filler_audio, filler_name = fillers.get_thinking_filler()
filler_time = (time.time() - filler_start) * 1000
print(f"\n[Phase 2] User stopped -> Filler ready in {filler_time:.2f}ms ('{filler_name}')")

# Phase 3: STT (parallel with filler playback)
stt_start = time.time()
transcript = stt.transcribe(test_audio)
stt_time = (time.time() - stt_start) * 1000
print(f"[Phase 3] STT complete: {stt_time:.0f}ms -> '{transcript}'")

# Phase 4: LLM
llm_start = time.time()
response = llm.generate(transcript if transcript else "Hello")
llm_time = (time.time() - llm_start) * 1000
print(f"[Phase 4] LLM complete: {llm_time:.0f}ms -> '{response[:50]}...'")

# Phase 5: TTS
tts_start = time.time()
response_audio = tts.generate(response, use_context=False)
tts_time = (time.time() - tts_start) * 1000
audio_duration = len(response_audio) / SAMPLE_RATE
print(f"[Phase 5] TTS complete: {tts_time:.0f}ms -> {audio_duration:.2f}s audio")

total_time = (time.time() - start_time) * 1000

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 70)
print("LATENCY REPORT")
print("=" * 70)

print(f"""
Component Breakdown:
  - Filler retrieval:  {filler_time:.2f}ms  (INSTANT - pre-generated)
  - STT (Whisper):     {stt_time:.0f}ms    (while filler plays)
  - LLM (Llama 3.2):   {llm_time:.0f}ms
  - TTS (CSM-1B):      {tts_time:.0f}ms

Timeline:
  t=0ms    : User stops speaking
  t={filler_time:.0f}ms   : Filler starts playing (user hears something!)
  t={stt_time:.0f}ms : STT complete
  t={stt_time + llm_time:.0f}ms : LLM complete
  t={total_time:.0f}ms: Full response audio ready

KEY METRICS:
  - Time to first audio (filler): {filler_time:.2f}ms  {'✓ INSTANT' if filler_time < 10 else '✗ TOO SLOW'}
  - Time to response start:       {stt_time + llm_time + tts_time:.0f}ms  {'✓ OK' if stt_time + llm_time + tts_time < 8000 else '✗ SLOW'}

BOTTLENECK: {'TTS (CSM)' if tts_time > max(stt_time, llm_time) else 'LLM' if llm_time > stt_time else 'STT'}
""")

# User perceives:
# - t=0: They stop speaking
# - t=<10ms: Hear filler ("Hmm...", "Yeah...")  <- INSTANT FEEDBACK
# - t=filler_duration: Filler ends
# - t=total_time: Response audio starts

filler_duration = len(filler_audio) / SAMPLE_RATE * 1000
print(f"""
USER PERCEPTION:
  - At t=0ms: User stops speaking
  - At t={filler_time:.0f}ms: User hears filler start
  - Filler plays for {filler_duration:.0f}ms
  - At t={total_time:.0f}ms: Response audio ready

  Gap between filler end and response: {total_time - filler_duration:.0f}ms
""")

if total_time - filler_duration > 0:
    print(f"  WARNING: {total_time - filler_duration:.0f}ms gap - user might notice silence!")
    print(f"  SOLUTION: Use longer fillers or optimize TTS")
else:
    print(f"  GREAT: Response ready before filler ends - seamless!")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
