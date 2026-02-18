#!/usr/bin/env python3
"""
Deep diagnosis of click sources in TTS pipeline.
Tests each processing stage to isolate the cause.
"""
import sys
import os

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.io.wavfile as wav

print('='*70)
print('  CLICK DIAGNOSIS - Finding the Source')
print('='*70)

# Import TTS module directly to inspect/modify
from maya.engine.tts_streaming_real import (
    RealStreamingTTSEngine,
    _normalize_lufs,
    _enhance_audio_quality
)

# Initialize
print('\nInitializing TTS engine...')
tts = RealStreamingTTSEngine()
tts.initialize()

test_text = 'hello how are you doing today'
print(f'\nTest phrase: "{test_text}"')

def analyze_clicks(audio_tensor, name):
    """Analyze audio for clicks."""
    audio = audio_tensor.cpu().numpy() if torch.is_tensor(audio_tensor) else audio_tensor
    diff = np.abs(np.diff(audio))
    severe = int(np.sum(diff > 0.3))
    moderate = int(np.sum(diff > 0.2))
    max_disc = float(np.max(diff))
    return {'name': name, 'severe': severe, 'moderate': moderate, 'max_disc': max_disc, 'len': len(audio)}

def save_audio(audio, name):
    """Save audio for manual listening."""
    audio_np = audio.cpu().numpy() if torch.is_tensor(audio) else audio
    path = f'/home/ec2-user/SageMaker/project_maya/diag_{name}.wav'
    wav.write(path, 24000, (audio_np * 32767).astype(np.int16))
    return path

print('\n' + '-'*70)
print('  TEST 1: Raw chunks from streaming context (no processing)')
print('-'*70)

# Collect raw chunks before any processing
raw_chunks = []
from moshi.utils.compile import no_cuda_graph

tts._model.reset_caches()
gen_tokens, gen_mask = tts._tokenize_text_segment(test_text, speaker=0)
tokens = [gen_tokens]
tokens_mask = [gen_mask]

# Skip voice prompt for cleaner diagnosis
# Voice prompt encoding requires special handling

prompt_tokens = torch.cat(tokens, dim=0).long().to("cuda")
prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to("cuda")

curr_tokens = prompt_tokens.unsqueeze(0)
curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to("cuda")

FIRST_CHUNK = 2
CHUNK_SIZE = 8
max_frames = 100

with no_cuda_graph(), tts._audio_tokenizer.streaming(1):
    frame_buffer = []
    is_first = True

    for _ in range(max_frames):
        sample = tts._model.generate_frame(
            curr_tokens, curr_tokens_mask, curr_pos,
            temperature=0.9, topk=50
        )

        if torch.all(sample == 0):
            break

        frame_buffer.append(sample.clone())

        chunk_size = FIRST_CHUNK if is_first else CHUNK_SIZE

        if len(frame_buffer) >= chunk_size:
            # Decode - THIS is the raw output
            cloned_frames = [f.clone() for f in frame_buffer]
            stacked = torch.stack(cloned_frames).permute(1, 2, 0)
            audio_chunk = tts._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)

            raw_chunks.append(audio_chunk.clone())
            frame_buffer = []
            is_first = False

        curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to("cuda")], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to("cuda")], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1

    # Last chunk
    if frame_buffer:
        cloned_frames = [f.clone() for f in frame_buffer]
        stacked = torch.stack(cloned_frames).permute(1, 2, 0)
        audio_chunk = tts._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)
        raw_chunks.append(audio_chunk.clone())

print(f'\nGenerated {len(raw_chunks)} raw chunks')

# Test 1A: Analyze individual raw chunks
print('\n  1A. Individual raw chunks:')
for i, chunk in enumerate(raw_chunks):
    stats = analyze_clicks(chunk, f'chunk_{i}')
    status = 'CLEAN' if stats['severe'] == 0 else f'{stats["severe"]} clicks'
    print(f'    Chunk {i}: {stats["len"]/24000*1000:.0f}ms, max_disc={stats["max_disc"]:.4f}, {status}')

# Test 1B: Concatenate raw chunks directly
print('\n  1B. Concatenated raw chunks (no processing):')
raw_concat = torch.cat(raw_chunks)
stats = analyze_clicks(raw_concat, 'raw_concat')
print(f'    Total: {stats["len"]/24000*1000:.0f}ms, max_disc={stats["max_disc"]:.4f}')
print(f'    Severe clicks: {stats["severe"]}, Moderate: {stats["moderate"]}')
save_audio(raw_concat, 'raw_concat')

print('\n' + '-'*70)
print('  TEST 2: DC offset removal only')
print('-'*70)

dc_removed_chunks = []
for chunk in raw_chunks:
    dc_removed = chunk - chunk.mean()
    dc_removed_chunks.append(dc_removed)

dc_concat = torch.cat(dc_removed_chunks)
stats = analyze_clicks(dc_concat, 'dc_removed')
print(f'  Total: {stats["len"]/24000*1000:.0f}ms, max_disc={stats["max_disc"]:.4f}')
print(f'  Severe clicks: {stats["severe"]}, Moderate: {stats["moderate"]}')
save_audio(dc_concat, 'dc_removed')

print('\n' + '-'*70)
print('  TEST 3: LUFS normalization only')
print('-'*70)

lufs_chunks = []
for chunk in raw_chunks:
    dc_removed = chunk - chunk.mean()
    lufs_norm = _normalize_lufs(dc_removed, target_lufs=-16.0, true_peak_limit=0.89)
    lufs_chunks.append(lufs_norm)

lufs_concat = torch.cat(lufs_chunks)
stats = analyze_clicks(lufs_concat, 'lufs_norm')
print(f'  Total: {stats["len"]/24000*1000:.0f}ms, max_disc={stats["max_disc"]:.4f}')
print(f'  Severe clicks: {stats["severe"]}, Moderate: {stats["moderate"]}')
save_audio(lufs_concat, 'lufs_norm')

print('\n' + '-'*70)
print('  TEST 4: Crossfade between raw chunks')
print('-'*70)

# Apply crossfade to raw chunks
CROSSFADE = 240  # 10ms at 24kHz
t = torch.linspace(0, 1, CROSSFADE, device="cuda")
fade_out = torch.cos(t * 3.14159 / 2)
fade_in = torch.sin(t * 3.14159 / 2)

crossfaded_output = []
prev_tail = None

for i, chunk in enumerate(raw_chunks):
    # DC remove first
    chunk = chunk - chunk.mean()

    if len(chunk) < CROSSFADE * 3:
        crossfaded_output.append(chunk)
        continue

    new_tail = chunk[-CROSSFADE:].clone()

    if prev_tail is not None:
        crossfaded = prev_tail * fade_out + chunk[:CROSSFADE] * fade_in
        middle = chunk[CROSSFADE:-CROSSFADE]
        crossfaded_output.append(torch.cat([crossfaded, middle]))
    else:
        crossfaded_output.append(chunk[:-CROSSFADE])

    prev_tail = new_tail

# Output remaining tail
if prev_tail is not None:
    crossfaded_output.append(prev_tail)

cf_concat = torch.cat(crossfaded_output)
stats = analyze_clicks(cf_concat, 'crossfaded')
print(f'  Total: {stats["len"]/24000*1000:.0f}ms, max_disc={stats["max_disc"]:.4f}')
print(f'  Severe clicks: {stats["severe"]}, Moderate: {stats["moderate"]}')
save_audio(cf_concat, 'crossfaded')

print('\n' + '-'*70)
print('  TEST 5: Check chunk boundaries')
print('-'*70)

# Find where clicks occur at chunk boundaries
print('\n  Analyzing discontinuities at chunk boundaries:')
position = 0
for i in range(len(raw_chunks) - 1):
    chunk1 = raw_chunks[i].cpu().numpy()
    chunk2 = raw_chunks[i + 1].cpu().numpy()

    # Discontinuity at boundary
    boundary_disc = abs(chunk1[-1] - chunk2[0])
    print(f'    Boundary {i}->{i+1}: disc={boundary_disc:.4f}' +
          (' *** SEVERE' if boundary_disc > 0.3 else ''))

    position += len(chunk1)

print('\n' + '='*70)
print('  DIAGNOSIS SUMMARY')
print('='*70)

print('\n  Audio files saved for manual listening:')
print('    - diag_raw_concat.wav (unprocessed)')
print('    - diag_dc_removed.wav (DC offset removed)')
print('    - diag_lufs_norm.wav (LUFS normalized)')
print('    - diag_crossfaded.wav (with crossfade)')
print('\n  Compare these to find the click source!')
print('='*70)
