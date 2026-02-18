#!/usr/bin/env python3
"""
Test the complete expressive pipeline:
1. Expresso voice prompt (matches training speaker)
2. Style tags preserved in TTS preprocessing
3. LLM generates emotion tags
4. TTS temperature 1.0 for expressiveness
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
print('  EXPRESSIVE PIPELINE TEST')
print('='*70)

# Test LLM emotion tagging
print('\n1. Testing LLM emotion tagging...')

from maya.engine.llm_vllm import VLLMEngine

llm = VLLMEngine()
llm.initialize()

test_inputs = [
    "hi how are you",
    "my dog just died",
    "i got promoted today!",
    "what do you mean by that?",
]

print('\nLLM responses:')
for user_input in test_inputs:
    response = llm.generate(user_input)
    has_tag = response.startswith('[')
    tag_status = 'HAS TAG' if has_tag else 'NO TAG'
    print(f'  User: "{user_input}"')
    print(f'  Maya: "{response}" [{tag_status}]')
    print()

# Test TTS with emotion tags
print('\n2. Testing TTS with emotion tags...')

from maya.engine.tts_streaming_real import RealStreamingTTSEngine

tts = RealStreamingTTSEngine()
tts.initialize()

# Check preprocessing preserves tags
test_text = "[happy] oh wow thats amazing!"
preprocessed = tts._preprocess_for_speech(test_text)
print(f'\n  Original: "{test_text}"')
print(f'  Preprocessed: "{preprocessed}"')
print(f'  Tags preserved: {"YES" if "[happy]" in preprocessed else "NO - BUG!"}')

# Generate expressive samples
output_dir = '/home/ec2-user/SageMaker/project_maya/audio_expressive_test'
os.makedirs(output_dir, exist_ok=True)

test_cases = [
    ('happy_excited', '[happy] oh wow thats so exciting i love it!'),
    ('sad_sympathy', '[sad] oh no im so sorry to hear that'),
    ('confused_question', '[confused] wait what do you mean by that?'),
    ('whisper_secret', '[whisper] hey can i tell you something'),
    ('neutral_response', 'yeah that makes sense to me'),
]

print(f'\n3. Generating expressive audio samples...')
print(f'   Output: {output_dir}/')

for name, text in test_cases:
    chunks = list(tts.generate_stream(text, use_context=False))
    if chunks:
        audio = torch.cat(chunks)
        audio_np = audio.cpu().numpy()

        # Analyze
        diff = np.abs(np.diff(audio_np))
        clicks = int(np.sum(diff > 0.3))

        # Save
        path = f'{output_dir}/{name}.wav'
        wav.write(path, 24000, (audio_np * 32767).astype(np.int16))
        print(f'   {name}.wav: {len(audio_np)/24000:.1f}s, {clicks} clicks')

    torch.cuda.empty_cache()

# Test full pipeline (LLM -> TTS)
print('\n4. Full pipeline test (LLM -> TTS)...')

full_tests = [
    "hi how are you doing today",
    "my cat is sick and im worried",
    "i just won the lottery!",
]

for user_input in full_tests:
    # Get LLM response
    response = llm.generate(user_input)
    print(f'\n   User: "{user_input}"')
    print(f'   Maya: "{response}"')

    # Generate audio
    chunks = list(tts.generate_stream(response, use_context=False))
    if chunks:
        audio = torch.cat(chunks)
        audio_np = audio.cpu().numpy()
        clicks = int(np.sum(np.abs(np.diff(audio_np)) > 0.3))

        # Save
        safe_name = user_input.replace(' ', '_')[:20]
        path = f'{output_dir}/full_{safe_name}.wav'
        wav.write(path, 24000, (audio_np * 32767).astype(np.int16))
        print(f'   Audio: {len(audio_np)/24000:.1f}s, {clicks} clicks')

    torch.cuda.empty_cache()

print('\n' + '='*70)
print('  TEST COMPLETE - Listen to audio files in:')
print(f'  {output_dir}/')
print('='*70)
