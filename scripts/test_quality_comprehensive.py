#!/usr/bin/env python3
"""
Comprehensive quality test - find optimal settings for click-free, high-quality audio.
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
import time
import scipy.io.wavfile as wav

print('='*70)
print('  COMPREHENSIVE QUALITY TEST')
print('='*70)

# Direct import to modify parameters
from maya.engine.tts_streaming_real import RealStreamingTTSEngine

# Initialize engine
print('\nInitializing TTS engine...')
tts = RealStreamingTTSEngine()
tts.initialize()

# Check which model was loaded
print('\nModel check:')
if hasattr(tts, '_model'):
    print('  Model loaded: YES')
    # Check if it has LoRA weights
    total_params = sum(p.numel() for p in tts._model.parameters())
    print(f'  Total parameters: {total_params:,}')

test_phrases = [
    'hi there',
    'hello how are you',
    'thats really interesting',
]

def analyze_audio(audio_tensor):
    """Comprehensive audio analysis."""
    audio = audio_tensor.cpu().numpy() if torch.is_tensor(audio_tensor) else audio_tensor
    diff = np.abs(np.diff(audio))

    return {
        'severe_clicks': int(np.sum(diff > 0.3)),
        'moderate_clicks': int(np.sum(diff > 0.2)),
        'max_disc': float(np.max(diff)),
        'rms': float(np.sqrt(np.mean(audio**2))),
        'peak': float(np.max(np.abs(audio))),
        'duration_ms': len(audio) / 24000 * 1000,
    }

print('\n' + '='*70)
print('  TEST 1: Current Settings (temp=1.0)')
print('='*70)

results_t10 = []
for phrase in test_phrases:
    for _ in range(5):
        chunks = list(tts.generate_stream(phrase, use_context=False))
        if chunks:
            audio = torch.cat(chunks)
            stats = analyze_audio(audio)
            results_t10.append(stats)
        torch.cuda.empty_cache()

clean_t10 = sum(1 for r in results_t10 if r['severe_clicks'] == 0)
print(f'\n  Clean runs: {clean_t10}/{len(results_t10)} ({100*clean_t10/len(results_t10):.0f}%)')
print(f'  Max disc range: {min(r["max_disc"] for r in results_t10):.3f} - {max(r["max_disc"] for r in results_t10):.3f}')

print('\n' + '='*70)
print('  TEST 2: Different Temperatures')
print('='*70)

# We need to modify the temperature for generation
# This is stored in TTS_CONFIG but used in generate_stream

from maya.config import TTS as TTS_CONFIG

# Backup original temperature
orig_temp = TTS_CONFIG.temperature

# Test different temperatures by patching
import maya.engine.tts_streaming_real as tts_module

for temp in [0.7, 0.8, 0.9]:
    print(f'\n--- Temperature {temp} ---')

    # Create a patched version of generate_stream that uses different temp
    results = []
    for phrase in test_phrases:
        for _ in range(5):
            # Generate with modified temperature
            # We need to call the model directly with different temp
            tts._model.reset_caches()

            from maya.engine.audio_processor import reset_processor
            reset_processor()

            text = tts._preprocess_for_speech(phrase)

            # Build prompt
            tokens, tokens_mask = [], []
            if tts._voice_prompt:
                seg_tokens, seg_mask = tts._tokenize_segment(tts._voice_prompt)
                tokens.append(seg_tokens)
                tokens_mask.append(seg_mask)

            gen_tokens, gen_mask = tts._tokenize_text_segment(text, speaker=0)
            tokens.append(gen_tokens)
            tokens_mask.append(gen_mask)

            prompt_tokens = torch.cat(tokens, dim=0).long().to("cuda")
            prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to("cuda")

            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to("cuda")

            from moshi.utils.compile import no_cuda_graph

            with no_cuda_graph(), tts._audio_tokenizer.streaming(1):
                frames = []
                for _ in range(100):
                    sample = tts._model.generate_frame(
                        curr_tokens, curr_tokens_mask, curr_pos,
                        temperature=temp,  # MODIFIED TEMPERATURE
                        topk=50
                    )
                    if torch.all(sample == 0):
                        break
                    frames.append(sample.clone())

                    curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to("cuda")], dim=1).unsqueeze(1)
                    curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to("cuda")], dim=1).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1

                if frames:
                    audio = tts._decode_frames_in_context(frames)
                    # Simple normalization
                    audio = audio - audio.mean()
                    peak = audio.abs().max()
                    if peak > 1e-6:
                        audio = audio * (0.7 / peak)

                    stats = analyze_audio(audio)
                    results.append(stats)

            torch.cuda.empty_cache()

    if results:
        clean = sum(1 for r in results if r['severe_clicks'] == 0)
        print(f'  Clean runs: {clean}/{len(results)} ({100*clean/len(results):.0f}%)')
        print(f'  Max disc range: {min(r["max_disc"] for r in results):.3f} - {max(r["max_disc"] for r in results):.3f}')

        # Save a sample
        chunks = list(tts.generate_stream('hello how are you', use_context=False))
        if chunks:
            sample_audio = torch.cat(chunks).cpu().numpy()
            wav.write(f'/home/ec2-user/SageMaker/project_maya/sample_temp{temp}.wav',
                     24000, (sample_audio * 32767).astype(np.int16))
            print(f'  Sample saved: sample_temp{temp}.wav')

print('\n' + '='*70)
print('  TEST 3: Without Enhancement (baseline)')
print('='*70)

# Patch to disable enhancement
orig_enhance = tts_module._enhance_audio_quality
tts_module._enhance_audio_quality = lambda a, sr=24000: a

results_no_enhance = []
for phrase in test_phrases:
    for _ in range(5):
        chunks = list(tts.generate_stream(phrase, use_context=False))
        if chunks:
            audio = torch.cat(chunks)
            stats = analyze_audio(audio)
            results_no_enhance.append(stats)
        torch.cuda.empty_cache()

clean_no = sum(1 for r in results_no_enhance if r['severe_clicks'] == 0)
print(f'\n  Clean runs: {clean_no}/{len(results_no_enhance)} ({100*clean_no/len(results_no_enhance):.0f}%)')
print(f'  Max disc range: {min(r["max_disc"] for r in results_no_enhance):.3f} - {max(r["max_disc"] for r in results_no_enhance):.3f}')

# Restore enhancement
tts_module._enhance_audio_quality = orig_enhance

print('\n' + '='*70)
print('  SUMMARY')
print('='*70)
print('\n  Listen to sample_temp*.wav files to judge quality')
print('  Higher clean % = fewer clicks')
print('  Need BOTH high quality AND no clicks')
print('='*70)
