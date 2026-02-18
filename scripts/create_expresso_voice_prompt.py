#!/usr/bin/env python3
"""
Create a proper voice prompt from Expresso ex04 training data.

The model was trained on Expresso ex04 speaker with style tags.
The voice prompt MUST be from the same speaker for consistent output.
"""
import sys
import os
import json

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import torch
import torchaudio
import numpy as np

print('='*70)
print('  CREATING VOICE PROMPT FROM EXPRESSO EX04')
print('='*70)

# Load training data manifest
with open('training/data/csm_ready_ex04/train.json', 'r') as f:
    train_data = json.load(f)

print(f'\nTotal training samples: {len(train_data)}')

# Analyze styles
styles = {}
for item in train_data:
    style = item.get('style', 'unknown')
    if style not in styles:
        styles[style] = []
    styles[style].append(item)

print('\nStyles in training data:')
for style, items in sorted(styles.items(), key=lambda x: -len(x[1])):
    total_dur = sum(i['duration'] for i in items)
    print(f'  {style}: {len(items)} samples, {total_dur:.1f}s total')

# For voice prompt, use "default" style (neutral, conversational)
# This is the most natural for general conversation
target_styles = ['default', 'happy', 'confused']  # Natural conversational styles

print('\n' + '-'*70)
print('  Selecting samples for voice prompt')
print('-'*70)

# Select samples that are:
# 1. From target styles (default, happy, confused)
# 2. Between 2-6 seconds (good length for context)
# 3. Have natural conversational text

selected = []
total_duration = 0
MAX_DURATION = 30  # 30 seconds of voice prompt

for item in train_data:
    if item['style'] not in target_styles:
        continue
    if item['duration'] < 2 or item['duration'] > 6:
        continue
    if total_duration + item['duration'] > MAX_DURATION:
        continue

    selected.append(item)
    total_duration += item['duration']

print(f'\nSelected {len(selected)} samples, total duration: {total_duration:.1f}s')

# Load and concatenate audio
print('\nLoading audio...')
audio_chunks = []
texts = []

for item in selected:
    audio_path = f"training/data/csm_ready_ex04/{item['path']}"
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 24kHz if needed
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_chunks.append(waveform.squeeze())

    # Clean text (remove style tags for voice prompt text)
    text = item['text']
    # Remove [style] tags
    import re
    text = re.sub(r'\[[\w]+\]\s*', '', text)
    texts.append(text)

# Concatenate with small gaps
GAP_SAMPLES = int(24000 * 0.3)  # 300ms gap between utterances
gap = torch.zeros(GAP_SAMPLES)

combined_audio = []
for i, chunk in enumerate(audio_chunks):
    combined_audio.append(chunk)
    if i < len(audio_chunks) - 1:
        combined_audio.append(gap)

combined_audio = torch.cat(combined_audio)
combined_text = ' ... '.join(texts)

print(f'\nCombined audio: {len(combined_audio)/24000:.1f}s')
print(f'Combined text: {combined_text[:200]}...')

# Save voice prompt
voice_prompt = {
    'audio': combined_audio,
    'text': combined_text,
    'speaker': 'ex04',
    'source': 'expresso',
    'styles': target_styles,
    'num_samples': len(selected)
}

output_path = 'assets/voice_prompt/maya_voice_prompt_expresso.pt'
torch.save(voice_prompt, output_path)
print(f'\nSaved: {output_path}')

# Also save as the main voice prompt
main_path = 'assets/voice_prompt/maya_voice_prompt.pt'
torch.save(voice_prompt, main_path)
print(f'Saved: {main_path} (main voice prompt)')

# Save audio as WAV for listening
import scipy.io.wavfile as wav
wav_path = 'assets/voice_prompt/maya_voice_prompt_expresso.wav'
wav.write(wav_path, 24000, (combined_audio.numpy() * 32767).astype(np.int16))
print(f'Saved: {wav_path} (for listening)')

print('\n' + '='*70)
print('  VOICE PROMPT CREATED FROM EXPRESSO EX04')
print('='*70)
print(f'\n  Duration: {len(combined_audio)/24000:.1f}s')
print(f'  Samples: {len(selected)}')
print(f'  Styles: {target_styles}')
print('\n  This voice prompt matches the training data speaker!')
print('='*70)
