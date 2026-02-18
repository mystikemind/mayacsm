#!/usr/bin/env python3
"""
FIX CPU OVERHEAD - The real bottleneck

DISCOVERY:
- CUDA time: 43.6ms (RTF 0.55x - better than the paper!)
- CPU time: 490.4ms (11x longer!)

The GPU is fast. Python/PyTorch dispatch overhead is the problem.

This script attempts various fixes:
1. torch.compile with different modes
2. Compile backbone and decoder separately
3. Try torch.jit.script/trace
4. Disable graph breaks
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import time
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

print("=" * 70)
print("FIXING CPU OVERHEAD - Achieving Real-Time RTF")
print("=" * 70)

device = "cuda"

from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

# Load model WITHOUT compile first
print("\nLoading CSM-1B without compile...")
model = Model.from_pretrained("sesame/csm-1b")
model.to(device=device, dtype=torch.bfloat16)
model.setup_caches(1)

# Load tokenizers
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

def benchmark_model(model, name, num_frames=30):
    """Benchmark a model configuration."""
    model.reset_caches()

    gen_tokens, gen_mask = tokenize_text("hello how are you", 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)

    # Warmup
    for _ in range(3):
        model.reset_caches()
        curr_tokens_tmp = gen_tokens.unsqueeze(0)
        curr_tokens_mask_tmp = gen_mask.unsqueeze(0)
        curr_pos_tmp = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)
        for i in range(5):
            sample = model.generate_frame(curr_tokens_tmp, curr_tokens_mask_tmp, curr_pos_tmp, 0.8, 50)
            if torch.all(sample == 0):
                break
            curr_tokens_tmp = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
            curr_tokens_mask_tmp = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
            curr_pos_tmp = curr_pos_tmp[:, -1:] + 1

    # Actual benchmark
    model.reset_caches()
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

    print(f"\n{name}:")
    print(f"  Frames: {len(frames)}, Time: {elapsed*1000:.0f}ms")
    print(f"  Frame time: {frame_time:.1f}ms (need 80ms for RTF 1.0)")
    print(f"  RTF: {rtf:.3f}x")

    return rtf, frame_time


# Test 1: No compile (baseline)
print("\n" + "=" * 60)
print("TEST 1: No torch.compile (baseline)")
print("=" * 60)
rtf_baseline, _ = benchmark_model(model, "No compile")


# Test 2: torch.compile with default mode
print("\n" + "=" * 60)
print("TEST 2: torch.compile(model) - default mode")
print("=" * 60)

model2 = Model.from_pretrained("sesame/csm-1b")
model2.to(device=device, dtype=torch.bfloat16)
model2.setup_caches(1)

try:
    model2 = torch.compile(model2)
    rtf_default, _ = benchmark_model(model2, "compile() default")
except Exception as e:
    print(f"Failed: {e}")
    rtf_default = rtf_baseline


# Test 3: torch.compile with reduce-overhead mode
print("\n" + "=" * 60)
print("TEST 3: torch.compile(model, mode='reduce-overhead')")
print("=" * 60)

model3 = Model.from_pretrained("sesame/csm-1b")
model3.to(device=device, dtype=torch.bfloat16)
model3.setup_caches(1)

try:
    model3 = torch.compile(model3, mode="reduce-overhead")
    rtf_reduce, _ = benchmark_model(model3, "compile() reduce-overhead")
except Exception as e:
    print(f"Failed: {e}")
    rtf_reduce = rtf_baseline


# Test 4: torch.compile with max-autotune mode
print("\n" + "=" * 60)
print("TEST 4: torch.compile(model, mode='max-autotune')")
print("=" * 60)

model4 = Model.from_pretrained("sesame/csm-1b")
model4.to(device=device, dtype=torch.bfloat16)
model4.setup_caches(1)

try:
    model4 = torch.compile(model4, mode="max-autotune")
    rtf_autotune, _ = benchmark_model(model4, "compile() max-autotune")
except Exception as e:
    print(f"Failed: {e}")
    rtf_autotune = rtf_baseline


# Test 5: Compile only the generate_frame function
print("\n" + "=" * 60)
print("TEST 5: Compile only generate_frame method")
print("=" * 60)

model5 = Model.from_pretrained("sesame/csm-1b")
model5.to(device=device, dtype=torch.bfloat16)
model5.setup_caches(1)

try:
    # Compile just the generate_frame method
    model5.generate_frame = torch.compile(model5.generate_frame, mode="reduce-overhead", fullgraph=False)
    rtf_method, _ = benchmark_model(model5, "compile(generate_frame)")
except Exception as e:
    print(f"Failed: {e}")
    rtf_method = rtf_baseline


# Test 6: Try with CUDA graphs manually
print("\n" + "=" * 60)
print("TEST 6: Try CUDA graphs for static input")
print("=" * 60)

model6 = Model.from_pretrained("sesame/csm-1b")
model6.to(device=device, dtype=torch.bfloat16)
model6.setup_caches(1)

try:
    # Create static tensors for CUDA graph
    gen_tokens, gen_mask = tokenize_text("hello how are you", 0)

    # Pre-allocate static buffers
    static_tokens = gen_tokens.unsqueeze(0).clone()
    static_mask = gen_mask.unsqueeze(0).clone()
    static_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    # Warmup before graph capture
    model6.reset_caches()
    for _ in range(3):
        model6.reset_caches()
        _ = model6.generate_frame(static_tokens, static_mask, static_pos, 0.8, 50)

    # Try to capture graph
    print("Attempting CUDA graph capture...")
    g = torch.cuda.CUDAGraph()

    # Capture
    model6.reset_caches()
    with torch.cuda.graph(g):
        output = model6.generate_frame(static_tokens, static_mask, static_pos, 0.8, 50)

    print("CUDA graph captured successfully!")

    # Replay benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(30):
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"30 frame replays: {elapsed*1000:.0f}ms")
    print(f"Frame time: {elapsed/30*1000:.1f}ms")
    print(f"RTF: {elapsed/30/0.08:.3f}x")

except Exception as e:
    print(f"CUDA graph failed: {e}")
    import traceback
    traceback.print_exc()


# Test 7: Try torch._dynamo.explain to understand graph breaks
print("\n" + "=" * 60)
print("TEST 7: Analyze graph breaks with torch._dynamo.explain")
print("=" * 60)

model7 = Model.from_pretrained("sesame/csm-1b")
model7.to(device=device, dtype=torch.bfloat16)
model7.setup_caches(1)

try:
    gen_tokens, gen_mask = tokenize_text("hello", 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    explanation = torch._dynamo.explain(model7.generate_frame)(
        curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50
    )

    print(f"\nGraph breaks: {explanation.graph_break_count}")
    if explanation.break_reasons:
        print("Break reasons (first 5):")
        for i, reason in enumerate(explanation.break_reasons[:5]):
            print(f"  {i+1}. {reason}")

except Exception as e:
    print(f"Explain failed: {e}")


# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results = [
    ("No compile (baseline)", rtf_baseline),
    ("compile() default", rtf_default),
    ("compile() reduce-overhead", rtf_reduce),
    ("compile() max-autotune", rtf_autotune),
    ("compile(generate_frame)", rtf_method),
]

print(f"\n{'Configuration':<30} {'RTF':<10} {'vs Baseline':<15}")
print("-" * 55)
for name, rtf in results:
    improvement = (rtf_baseline - rtf) / rtf_baseline * 100
    print(f"{name:<30} {rtf:.3f}x     {improvement:+.1f}%")

print(f"\nTarget RTF: < 1.0 (currently {rtf_baseline:.2f}x)")
print(f"i-LAVA paper achieved: 0.785x on L4 GPU")
print("=" * 70)
