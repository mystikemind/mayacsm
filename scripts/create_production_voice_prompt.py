#!/usr/bin/env python3
"""
Create Production Voice Prompt for CSM Fine-tuned Model

Based on research findings:
- Recommended voice prompt length: 2-3 minutes
- Voice prompt provides speaker context to CSM
- Longer prompts = more consistent voice output

This script creates a long voice prompt from the best Expresso samples.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.io.wavfile as wav

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
EXPRESSO_DIR = PROJECT_ROOT / "training" / "data" / "expresso_talia"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "voice_prompt"

print("=" * 70)
print("CREATING PRODUCTION VOICE PROMPT")
print("Target: 2-3 minutes of audio context")
print("=" * 70)

# Load metadata
metadata_path = EXPRESSO_DIR / "metadata.json"
if not metadata_path.exists():
    print(f"ERROR: Metadata not found at {metadata_path}")
    sys.exit(1)

with open(metadata_path) as f:
    metadata = json.load(f)

print(f"\nFound {len(metadata)} samples in Expresso dataset")

# Sort by duration (we want variety, not just long samples)
# Mix different emotions and lengths for natural variety
samples_with_duration = []
for sample in metadata:
    audio_path = EXPRESSO_DIR / sample["audio"]
    if audio_path.exists():
        samples_with_duration.append({
            **sample,
            "audio_path": audio_path,
        })

print(f"Valid samples: {len(samples_with_duration)}")

# Strategy: Select diverse samples to reach ~2-3 minutes
# Prefer samples with:
# 1. Clear speech (no noise)
# 2. Varied emotions (happy, neutral, sad, etc.)
# 3. Different lengths

TARGET_DURATION_SEC = 150  # 2.5 minutes

# Group by emotion/style if available
emotion_samples = {}
for sample in samples_with_duration:
    text = sample.get("text", "").lower()
    # Simple emotion detection from text/filename
    style = "neutral"
    if "happy" in str(sample.get("audio", "")).lower() or any(w in text for w in ["love", "great", "amazing", "wonderful"]):
        style = "happy"
    elif "sad" in str(sample.get("audio", "")).lower() or any(w in text for w in ["sorry", "sad", "miss", "hard"]):
        style = "sad"
    elif "confused" in str(sample.get("audio", "")).lower() or "?" in text:
        style = "confused"

    if style not in emotion_samples:
        emotion_samples[style] = []
    emotion_samples[style].append(sample)

print(f"\nSamples by style:")
for style, samples in emotion_samples.items():
    total_dur = sum(s.get("duration", 3) for s in samples)
    print(f"  {style}: {len(samples)} samples ({total_dur:.1f}s)")

# Select samples to reach target duration
selected_samples = []
current_duration = 0

# First, take some from each style
for style in ["neutral", "happy", "sad", "confused"]:
    if style in emotion_samples:
        # Sort by duration (prefer medium length)
        style_samples = sorted(emotion_samples[style], key=lambda x: abs(x.get("duration", 3) - 4))[:10]
        for sample in style_samples:
            if current_duration < TARGET_DURATION_SEC:
                selected_samples.append(sample)
                current_duration += sample.get("duration", 3)

# If still need more, add more neutral samples
if current_duration < TARGET_DURATION_SEC:
    for sample in sorted(emotion_samples.get("neutral", []), key=lambda x: x.get("duration", 0), reverse=True):
        if sample not in selected_samples and current_duration < TARGET_DURATION_SEC:
            selected_samples.append(sample)
            current_duration += sample.get("duration", 3)

print(f"\nSelected {len(selected_samples)} samples ({current_duration:.1f}s)")

# Load and concatenate audio
print("\nLoading audio files...")
audio_segments = []
text_segments = []

for i, sample in enumerate(selected_samples):
    audio_path = sample["audio_path"]
    try:
        sr, audio = wav.read(audio_path)
        if sr != 24000:
            # Resample if needed
            import librosa
            audio = audio.astype(np.float32) / 32768.0
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        else:
            audio = audio.astype(np.float32) / 32768.0

        audio_segments.append(audio)
        text_segments.append(sample.get("text", ""))

        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{len(selected_samples)}...")
    except Exception as e:
        print(f"  Warning: Failed to load {audio_path}: {e}")

print(f"\nLoaded {len(audio_segments)} audio segments")

# Concatenate with small silence gaps
silence_samples = int(0.3 * 24000)  # 300ms silence between segments
silence = np.zeros(silence_samples, dtype=np.float32)

combined_audio = []
for i, audio in enumerate(audio_segments):
    combined_audio.append(audio)
    if i < len(audio_segments) - 1:
        combined_audio.append(silence)

combined_audio = np.concatenate(combined_audio)
combined_text = " ".join(text_segments)

print(f"\nCombined audio: {len(combined_audio) / 24000:.1f}s")
print(f"Combined text: {len(combined_text)} chars, {len(combined_text.split())} words")

# Normalize audio
peak = np.abs(combined_audio).max()
if peak > 0:
    combined_audio = combined_audio / peak * 0.8  # -2dB headroom

# Save as WAV
output_wav = OUTPUT_DIR / "maya_voice_prompt_production.wav"
wav.write(str(output_wav), 24000, (combined_audio * 32767).astype(np.int16))
print(f"\nSaved WAV: {output_wav}")

# Create voice prompt tensor for CSM
print("\nTokenizing for CSM...")

# Load Mimi codec
from moshi.models import loaders
mimi = loaders.get_mimi("/home/ec2-user/SageMaker/csm/tokenizer-e351c8d8-checkpoint125.safetensors", device="cuda")
mimi.eval()

# Tokenize audio
audio_tensor = torch.tensor(combined_audio).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, samples)
with torch.no_grad():
    codes = mimi.encode(audio_tensor)  # (1, 32, frames)
codes = codes.squeeze(0)  # (32, frames)

print(f"Audio tokens: {codes.shape}")

# Create Segment for voice prompt
from generator import Segment

voice_prompt = Segment(
    text=combined_text[:1000],  # Truncate text if too long (CSM has limits)
    speaker=0,
    audio=torch.tensor(combined_audio),
    audio_tokens=codes.cpu(),
)

# Save voice prompt
output_pt = OUTPUT_DIR / "maya_voice_prompt_production.pt"
torch.save(voice_prompt, output_pt)
print(f"Saved voice prompt: {output_pt}")

# Also save a shorter version (30 seconds) for faster inference
if len(combined_audio) > 30 * 24000:
    short_audio = combined_audio[:30 * 24000]
    short_codes = codes[:, :int(30 * 12.5)]  # 12.5 Hz

    short_prompt = Segment(
        text=combined_text[:300],
        speaker=0,
        audio=torch.tensor(short_audio),
        audio_tokens=short_codes.cpu(),
    )

    short_output = OUTPUT_DIR / "maya_voice_prompt_production_short.pt"
    torch.save(short_prompt, short_output)
    print(f"Saved short version: {short_output}")

print("\n" + "=" * 70)
print("VOICE PROMPT CREATED")
print("=" * 70)
print(f"\nFiles:")
print(f"  WAV: {output_wav} ({len(combined_audio)/24000:.1f}s)")
print(f"  PT:  {output_pt}")
print(f"\nUse in TTS:")
print(f"  voice_prompt = torch.load('{output_pt}')")
print("=" * 70)
