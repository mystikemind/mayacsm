#!/usr/bin/env python3
"""
DEEP INVESTIGATION: Why is our RTF 2.8x when the i-LAVA paper achieves 0.785x?

Both use:
- CSM-1B model
- 32 codebooks
- Similar GPU (L4 vs A10G)

The paper achieved RTF 0.785x on L4 GPU with 32 codebooks.
We're getting RTF 2.8x - that's 3.5x slower!

Investigation areas:
1. Is torch.compile actually working?
2. Are there hidden CPU-GPU sync points?
3. What's the actual breakdown of time per operation?
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import time
import os

print("=" * 70)
print("RTF INVESTIGATION - Matching i-LAVA Paper Performance")
print("=" * 70)

# Check CUDA device
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

# Check if torch.compile is available
print(f"\ntorch.compile available: {hasattr(torch, 'compile')}")

# Load the model
print("\n" + "=" * 60)
print("STEP 1: Loading CSM-1B and checking torch.compile status")
print("=" * 60)

from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

device = "cuda"

# Load model
print("Loading CSM-1B...")
model = Model.from_pretrained("sesame/csm-1b")
model.to(device=device, dtype=torch.bfloat16)
model.setup_caches(1)

print(f"Model type before compile: {type(model)}")

# Apply torch.compile like the paper suggests
print("\nApplying torch.compile (the i-LAVA paper's key optimization)...")

try:
    # The paper says: "generating Kernels via JIT Compilation using torch.compile
    # for the backbone, decoder and other functions"
    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    print(f"Model type after compile: {type(compiled_model)}")
    print("torch.compile applied successfully!")
    model = compiled_model
except Exception as e:
    print(f"torch.compile failed: {e}")
    print("Using unoptimized model")

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

# Load Mimi
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)
mimi.set_num_codebooks(32)

print("\n" + "=" * 60)
print("STEP 2: Warmup (i-LAVA paper says 2+ generations needed)")
print("=" * 60)

def tokenize_text(text, speaker):
    text_tokens = tokenizer.encode(f"[{speaker}]{text}")
    text_frame = torch.zeros(len(text_tokens), 33).long()
    text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
    text_frame[:, -1] = torch.tensor(text_tokens)
    text_frame_mask[:, -1] = True
    return text_frame.to(device), text_frame_mask.to(device)

def generate_audio(text, max_frames=50):
    """Generate audio frames and return timing breakdown."""
    model.reset_caches()

    gen_tokens, gen_mask = tokenize_text(text, 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    frames = []
    frame_times = []

    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)

    for i in range(max_frames):
        torch.cuda.synchronize()
        frame_start = time.time()

        sample = model.generate_frame(
            curr_tokens,
            curr_tokens_mask,
            curr_pos,
            temperature=0.8,
            topk=50
        )

        torch.cuda.synchronize()
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)

        if torch.all(sample == 0):
            break

        frames.append(sample)

        curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1

    return frames, frame_times

# Warmup
print("Warming up (3 generations)...")
for i in range(3):
    frames, _ = generate_audio("hello there", max_frames=20)
    print(f"  Warmup {i+1}: {len(frames)} frames")

print("\n" + "=" * 60)
print("STEP 3: Detailed timing analysis")
print("=" * 60)

# Test with the same text as the paper
test_text = "i am an ai i am designed to assist and provide helpful responses"

print(f"\nTest text: '{test_text}'")
print("\nGenerating and measuring frame-by-frame timing...")

torch.cuda.synchronize()
total_start = time.time()

frames, frame_times = generate_audio(test_text, max_frames=100)

torch.cuda.synchronize()
total_time = time.time() - total_start

print(f"\nGenerated {len(frames)} frames")
print(f"Total generation time: {total_time*1000:.0f}ms")

# Calculate RTF
audio_duration = len(frames) * 0.08  # 80ms per frame
rtf = total_time / audio_duration

print(f"Audio duration: {audio_duration:.2f}s")
print(f"RTF: {rtf:.3f}x")

# Analyze frame timing
avg_frame_time = sum(frame_times) / len(frame_times) * 1000
first_frame_time = frame_times[0] * 1000
later_frames_avg = sum(frame_times[1:]) / len(frame_times[1:]) * 1000 if len(frame_times) > 1 else 0

print(f"\nFrame timing analysis:")
print(f"  First frame: {first_frame_time:.1f}ms")
print(f"  Subsequent frames avg: {later_frames_avg:.1f}ms")
print(f"  All frames avg: {avg_frame_time:.1f}ms")
print(f"  Target for RTF 1.0: 80ms per frame")

# Check for variance
min_time = min(frame_times[1:]) * 1000 if len(frame_times) > 1 else 0
max_time = max(frame_times[1:]) * 1000 if len(frame_times) > 1 else 0
print(f"  Min frame time: {min_time:.1f}ms")
print(f"  Max frame time: {max_time:.1f}ms")

print("\n" + "=" * 60)
print("STEP 4: Decode and measure Mimi timing")
print("=" * 60)

if frames:
    stacked = torch.stack(frames).permute(1, 2, 0)

    torch.cuda.synchronize()
    decode_start = time.time()
    audio = mimi.decode(stacked)
    torch.cuda.synchronize()
    decode_time = (time.time() - decode_start) * 1000

    audio = audio.squeeze()
    print(f"Decoded {len(audio)} samples ({len(audio)/24000:.2f}s audio)")
    print(f"Mimi decode time: {decode_time:.0f}ms")
    print(f"Mimi is {'NOT' if decode_time > 100 else ''} the bottleneck")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

print(f"""
i-LAVA paper achieved on L4 GPU:
- RTF: 0.785x with 32 codebooks
- First chunk latency: 1381.9ms

Our results on A10G:
- RTF: {rtf:.3f}x with 32 codebooks
- Frame time: {avg_frame_time:.1f}ms (need 80ms for RTF 1.0)

Gap analysis:
- We're {rtf/0.785:.1f}x slower than the paper
- Our frame time is {avg_frame_time/80:.1f}x slower than needed

Possible causes:
1. torch.compile not working properly
2. Hidden CPU-GPU synchronization points
3. Different torch/CUDA versions
4. KVCache operations causing sync

The paper says: "sequential Python code and PyTorch operations
resulting in several handoffs between GPU and CPU computation"
""")

print("\n" + "=" * 60)
print("STEP 5: Profile the generate_frame function in detail")
print("=" * 60)

# Let's look at what generate_frame does internally
print("\nProfiling with torch.profiler...")

try:
    from torch.profiler import profile, record_function, ProfilerActivity

    model.reset_caches()
    gen_tokens, gen_mask = tokenize_text("hello", 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with record_function("generate_frame"):
            sample = model.generate_frame(
                curr_tokens,
                curr_tokens_mask,
                curr_pos,
                temperature=0.8,
                topk=50
            )

    # Print the top operations
    print("\nTop 10 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

except Exception as e:
    print(f"Profiling failed: {e}")

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
