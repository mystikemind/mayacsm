#!/usr/bin/env python3
"""
Test compiling backbone and decoder separately (i-LAVA paper approach).

The paper says:
"Code to Kernel: CSM-1B uses a Llama 3.2 1B as backbone and Llama 3.2 100M as decoder.
The open source implementation of the code was optimized by generating Kernels via
JIT Compilation using torch.compile for the backbone, decoder and other functions"

This suggests compiling backbone and decoder separately, not the whole model.
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Apply patches first
from maya.patches import apply_all_patches
apply_all_patches()

import torch
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True

import time

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda"

print("=" * 70)
print("TESTING SEPARATE COMPILATION (i-LAVA approach)")
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

# Compile backbone and decoder SEPARATELY
print("\nCompiling backbone and decoder separately...")

try:
    # Compile backbone with reduce-overhead for minimum latency
    model.backbone = torch.compile(model.backbone, mode="reduce-overhead", fullgraph=False)
    print("  Backbone compiled ✓")
except Exception as e:
    print(f"  Backbone compile failed: {e}")

try:
    # Compile decoder with reduce-overhead
    model.decoder = torch.compile(model.decoder, mode="reduce-overhead", fullgraph=False)
    print("  Decoder compiled ✓")
except Exception as e:
    print(f"  Decoder compile failed: {e}")

# Also compile the embedding and head operations
try:
    model.text_embeddings = torch.compile(model.text_embeddings, mode="reduce-overhead")
    model.audio_embeddings = torch.compile(model.audio_embeddings, mode="reduce-overhead")
    print("  Embeddings compiled ✓")
except Exception as e:
    print(f"  Embeddings compile failed: {e}")

try:
    model.projection = torch.compile(model.projection, mode="reduce-overhead")
    model.codebook0_head = torch.compile(model.codebook0_head, mode="reduce-overhead")
    print("  Projection and heads compiled ✓")
except Exception as e:
    print(f"  Projection/heads compile failed: {e}")

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


def benchmark(model, num_frames=30, num_warmup=3, num_runs=3):
    """Benchmark frame generation."""
    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)

    # Warmup
    print("\nWarming up...")
    for w in range(num_warmup):
        model.reset_caches()
        gen_tokens, gen_mask = tokenize_text("hello", 0)
        curr_tokens = gen_tokens.unsqueeze(0)
        curr_tokens_mask = gen_mask.unsqueeze(0)
        curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

        for i in range(10):
            sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50)
            if torch.all(sample == 0):
                break
            curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        print(f"  Warmup {w+1} done")

    # Benchmark runs
    print("\nBenchmarking...")
    rtfs = []
    frame_times_all = []

    for run in range(num_runs):
        model.reset_caches()

        gen_tokens, gen_mask = tokenize_text("hello how are you doing today", 0)
        curr_tokens = gen_tokens.unsqueeze(0)
        curr_tokens_mask = gen_mask.unsqueeze(0)
        curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

        frame_times = []
        frames = []

        for i in range(num_frames):
            torch.cuda.synchronize()
            frame_start = time.time()

            sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50)

            torch.cuda.synchronize()
            frame_time = (time.time() - frame_start) * 1000
            frame_times.append(frame_time)

            if torch.all(sample == 0):
                break

            frames.append(sample)
            curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        total_time = sum(frame_times) / 1000
        audio_duration = len(frames) * 0.08
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        rtfs.append(rtf)
        frame_times_all.extend(frame_times[1:])  # Skip first frame (compilation overhead)

        print(f"  Run {run+1}: {len(frames)} frames, RTF={rtf:.3f}x, avg frame={sum(frame_times)/len(frame_times):.0f}ms")

    return sum(rtfs) / len(rtfs), frame_times_all


# Run benchmark
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

avg_rtf, frame_times = benchmark(model, num_frames=30, num_warmup=5, num_runs=3)

print(f"\nAverage RTF: {avg_rtf:.3f}x")
print(f"Target: < 1.0 (i-LAVA achieved 0.383x one-shot, 0.785x streaming)")

if frame_times:
    avg_frame = sum(frame_times) / len(frame_times)
    min_frame = min(frame_times)
    max_frame = max(frame_times)
    print(f"\nFrame timing (after warmup):")
    print(f"  Average: {avg_frame:.1f}ms")
    print(f"  Min: {min_frame:.1f}ms")
    print(f"  Max: {max_frame:.1f}ms")
    print(f"  Target for RTF 1.0: 80ms")

print("\n" + "=" * 70)

if avg_rtf < 1.0:
    print("SUCCESS! Real-time streaming possible!")
elif avg_rtf < 1.5:
    print("CLOSE! May work with buffering.")
elif avg_rtf < 2.0:
    print("Getting better, but still needs optimization.")
else:
    print("Still too slow. The CPU overhead remains the bottleneck.")
    print("\nNext steps:")
    print("1. Try TorchScript/ONNX export")
    print("2. Use TensorRT for inference")
    print("3. Custom CUDA kernels for decoder loop")
