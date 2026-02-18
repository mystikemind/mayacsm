#!/usr/bin/env python3
"""
Simple compile test - use mode="default" to avoid CUDA graph OOM.
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Apply patches
from maya.patches import apply_all_patches
apply_all_patches()

import torch
import time
import os

# Set memory config to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda"

print("=" * 70)
print("SIMPLE COMPILE TEST")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

from huggingface_hub import hf_hub_download
from models import Model
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

# Load model
print("\nLoading CSM-1B...")
model = Model.from_pretrained("sesame/csm-1b")
model.to(device=device, dtype=torch.bfloat16)
model.setup_caches(1)

# Test 1: No compile baseline
print("\n" + "=" * 50)
print("TEST 1: No torch.compile (baseline)")
print("=" * 50)

# Load tokenizer
tokenizer_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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

def run_benchmark(model, num_frames=25, warmup=3):
    """Run benchmark and return RTF."""
    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)

    # Warmup
    for _ in range(warmup):
        model.reset_caches()
        gen_tokens, gen_mask = tokenize_text("hi", 0)
        curr_tokens = gen_tokens.unsqueeze(0)
        curr_tokens_mask = gen_mask.unsqueeze(0)
        curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)
        for i in range(5):
            sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50)
            if torch.all(sample == 0):
                break
            curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

    # Benchmark
    model.reset_caches()
    gen_tokens, gen_mask = tokenize_text("hello how are you", 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    torch.cuda.synchronize()
    start = time.time()
    frames = []

    for i in range(num_frames):
        sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50)
        if torch.all(sample == 0):
            break
        frames.append(sample)
        curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1

    torch.cuda.synchronize()
    elapsed = time.time() - start

    audio_duration = len(frames) * 0.08
    rtf = elapsed / audio_duration if audio_duration > 0 else 0
    frame_time = elapsed / len(frames) * 1000 if frames else 0

    return rtf, frame_time, len(frames)


rtf, frame_time, num = run_benchmark(model)
print(f"Frames: {num}, RTF: {rtf:.3f}x, Frame time: {frame_time:.0f}ms")
baseline_rtf = rtf

# Test 2: torch.compile with default mode
print("\n" + "=" * 50)
print("TEST 2: torch.compile(mode='default')")
print("=" * 50)

# Reload model fresh
del model
torch.cuda.empty_cache()

model2 = Model.from_pretrained("sesame/csm-1b")
model2.to(device=device, dtype=torch.bfloat16)
model2.setup_caches(1)

try:
    model2 = torch.compile(model2, mode="default")
    print("Compiled with mode='default'")
    rtf, frame_time, num = run_benchmark(model2)
    print(f"Frames: {num}, RTF: {rtf:.3f}x, Frame time: {frame_time:.0f}ms")
    default_rtf = rtf
except Exception as e:
    print(f"Failed: {e}")
    default_rtf = baseline_rtf

# Test 3: torch.compile generate_frame only with default
print("\n" + "=" * 50)
print("TEST 3: Compile only generate_frame (default mode)")
print("=" * 50)

del model2
torch.cuda.empty_cache()

model3 = Model.from_pretrained("sesame/csm-1b")
model3.to(device=device, dtype=torch.bfloat16)
model3.setup_caches(1)

try:
    # Compile only the generate_frame method
    original_gf = model3.generate_frame
    model3.generate_frame = torch.compile(original_gf, mode="default", fullgraph=False)
    print("Compiled generate_frame with mode='default'")
    rtf, frame_time, num = run_benchmark(model3)
    print(f"Frames: {num}, RTF: {rtf:.3f}x, Frame time: {frame_time:.0f}ms")
    gf_rtf = rtf
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    gf_rtf = baseline_rtf

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Baseline (no compile):     RTF {baseline_rtf:.3f}x")
print(f"Full model compile:        RTF {default_rtf:.3f}x ({(baseline_rtf-default_rtf)/baseline_rtf*100:+.1f}%)")
print(f"generate_frame compile:    RTF {gf_rtf:.3f}x ({(baseline_rtf-gf_rtf)/baseline_rtf*100:+.1f}%)")
print(f"\nTarget: < 1.0x (i-LAVA achieved 0.383x)")
print(f"Current best: {min(baseline_rtf, default_rtf, gf_rtf):.3f}x")
print("=" * 70)
