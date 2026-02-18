#!/usr/bin/env python3
"""
Test ORIGINAL CSM - no modifications, 32 codebooks.
This should produce good quality audio as baseline.
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import torchaudio
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda"

print("=" * 70)
print("BASELINE CSM TEST - Original 32 codebooks, NO modifications")
print("=" * 70)

from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

# Load model - ORIGINAL, no compile, no modifications
print("\nLoading CSM-1B (original, unmodified)...")
model = Model.from_pretrained("sesame/csm-1b")
model.to(device=device, dtype=torch.bfloat16)
model.setup_caches(1)

# Load Mimi - ORIGINAL 32 codebooks
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)
# DO NOT set_num_codebooks - use default 32

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
bos = tokenizer.bos_token
eos = tokenizer.eos_token
tokenizer._tokenizer.post_processor = TemplateProcessing(
    single=f"{bos}:0 $A:0 {eos}:0",
    pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
    special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
)

def tokenize_text(text, speaker):
    text_tokens = tokenizer.encode(f"[{speaker}]{text}")
    text_frame = torch.zeros(len(text_tokens), 33).long()
    text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
    text_frame[:, -1] = torch.tensor(text_tokens)
    text_frame_mask[:, -1] = True
    return text_frame.to(device), text_frame_mask.to(device)

# Test sentences
test_sentences = [
    "hi how are you",
    "im doing great thanks for asking",
]

print("\n" + "=" * 70)
print("GENERATING WITH ORIGINAL CSM (32 codebooks)")
print("=" * 70)

for idx, text in enumerate(test_sentences):
    print(f"\nGenerating: '{text}'")
    
    model.reset_caches()
    gen_tokens, gen_mask = tokenize_text(text, 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)
    
    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)
    
    frames = []
    max_frames = 60
    
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(max_frames):
        # Use ORIGINAL generate_frame - no modifications
        sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50)
        
        if torch.all(sample == 0):
            break
        
        frames.append(sample)
        curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1
    
    torch.cuda.synchronize()
    gen_time = time.time() - start
    
    if frames:
        # Decode with ALL 32 codebooks
        stacked = torch.stack(frames).permute(1, 2, 0)
        audio = mimi.decode(stacked).squeeze()
        
        # Normalize
        audio = audio - audio.mean()
        peak = audio.abs().max()
        if peak > 0:
            audio = audio * (0.5 / peak)
        
        duration = len(audio) / 24000
        rtf = gen_time / duration
        
        filename = f"baseline_32cb_{idx}.wav"
        output_path = f"/home/ec2-user/SageMaker/project_maya/tests/outputs/quality_test/{filename}"
        torchaudio.save(output_path, audio.unsqueeze(0).detach().cpu(), 24000)
        
        print(f"  → {filename}: {duration:.1f}s audio, RTF={rtf:.2f}x")
        print(f"     Frames: {len(frames)}, Gen time: {gen_time:.1f}s")

print("\n" + "=" * 70)
print("BASELINE TEST COMPLETE")
print("=" * 70)
print("\nIf these sound good, the problem is in the fast decoder.")
print("If these also sound bad, there's a deeper issue.")
print("\nFiles saved to: tests/outputs/quality_test/baseline_*.wav")
