#!/usr/bin/env python3
"""
Test RTF with patches applied BEFORE model loading.

CRITICAL: Import patches FIRST before any model imports!
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# CRITICAL: Apply patches BEFORE importing models!
print("=" * 70)
print("TESTING PATCHED RTF - Applying fixes before model loading")
print("=" * 70)

print("\nStep 1: Applying patches...")
from maya.patches import apply_all_patches
apply_all_patches()

# Also enable dynamo config
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True
print("torch._dynamo scalar capture enabled")

import torch
import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda"

print(f"\nGPU: {torch.cuda.get_device_name(0)}")

# NOW import and load the model
print("\nStep 2: Loading CSM-1B...")
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

model = Model.from_pretrained("sesame/csm-1b")
model.to(device=device, dtype=torch.bfloat16)
model.setup_caches(1)

print("\nStep 3: Applying torch.compile...")
try:
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    print("torch.compile with fullgraph=True applied!")
except Exception as e:
    print(f"fullgraph failed: {e}")
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile with reduce-overhead applied (fallback)")
    except Exception as e2:
        print(f"All compile failed: {e2}")

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


def benchmark(num_frames=30, num_runs=3):
    """Run benchmark and return average RTF."""
    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)

    rtfs = []

    for run in range(num_runs):
        model.reset_caches()

        gen_tokens, gen_mask = tokenize_text("hello how are you doing today", 0)
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
        rtfs.append(rtf)

        print(f"  Run {run+1}: {len(frames)} frames, {elapsed*1000:.0f}ms, RTF={rtf:.3f}x")

    return sum(rtfs) / len(rtfs)


# Warmup
print("\nStep 4: Warmup (3 generations)...")
for i in range(3):
    model.reset_caches()
    gen_tokens, gen_mask = tokenize_text("warmup", 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)
    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)

    for _ in range(10):
        sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50)
        if torch.all(sample == 0):
            break
        curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1
    print(f"  Warmup {i+1} done")


print("\nStep 5: Benchmark with patches...")
avg_rtf = benchmark(num_frames=30, num_runs=3)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Average RTF: {avg_rtf:.3f}x")
print(f"Target: < 1.0 (i-LAVA paper achieved 0.785x)")
print(f"Frame time: {avg_rtf * 80:.1f}ms (need 80ms for RTF 1.0)")
print("=" * 70)

if avg_rtf < 1.0:
    print("SUCCESS! Real-time streaming is possible!")
elif avg_rtf < 1.5:
    print("CLOSE! May work with buffering.")
else:
    print("STILL TOO SLOW. Need more optimization.")
