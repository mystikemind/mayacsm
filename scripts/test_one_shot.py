#!/usr/bin/env python3
"""
Test ONE-SHOT generation like the i-LAVA paper.

The paper says:
"One Shot Generation: The model is able to achieve a Real Time Factor (RTF) < 1 (0.383)
on GPU processing without reducing RVQ Iterations"

Let's test one-shot (batch all frames at once) vs streaming (frame by frame).
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Apply patches
from maya.patches import apply_all_patches
apply_all_patches()

import torch
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

import time
import torchaudio

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda"

print("=" * 70)
print("ONE-SHOT vs STREAMING TEST")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from generator import Generator, Segment

# Load model
print("\nLoading CSM-1B...")
model = Model.from_pretrained("sesame/csm-1b")
model.to(device=device, dtype=torch.bfloat16)
model.setup_caches(1)

# Try compile
print("Applying torch.compile...")
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("torch.compile applied")
except Exception as e:
    print(f"compile failed: {e}")

# Load Mimi
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)
mimi.set_num_codebooks(32)

# Load generator (uses the standard generate method)
generator = Generator(model)

# Test sentence from the paper
test_text = "i am an ai i am designed to assist and provide helpful responses to your queries i am a machine learning model trained on a vast amount of text data"

print(f"\nTest text ({len(test_text)} chars): '{test_text[:50]}...'")

# Warmup
print("\nWarming up...")
for _ in range(3):
    _ = generator.generate(
        text=test_text[:20],
        speaker=0,
        context=[],
        max_audio_length_ms=1000,
    )
print("Warmup done")

# Test ONE-SHOT generation
print("\n" + "=" * 60)
print("TEST: ONE-SHOT GENERATION")
print("=" * 60)

torch.cuda.synchronize()
start = time.time()

audio = generator.generate(
    text=test_text,
    speaker=0,
    context=[],
    max_audio_length_ms=30000,  # 30 seconds max
)

torch.cuda.synchronize()
elapsed = time.time() - start

audio_duration = len(audio) / 24000
rtf = elapsed / audio_duration

print(f"\nGenerated audio: {len(audio)} samples ({audio_duration:.2f}s)")
print(f"Generation time: {elapsed*1000:.0f}ms")
print(f"RTF: {rtf:.3f}x")

if rtf < 1.0:
    print("SUCCESS! Real-time capable!")
else:
    print(f"NOT real-time. Need {rtf:.1f}x improvement.")

# Save audio
output_path = "/home/ec2-user/SageMaker/project_maya/tests/outputs/one_shot_test.wav"
audio_normalized = audio - audio.mean()
peak = audio_normalized.abs().max()
if peak > 0:
    audio_normalized = audio_normalized * (0.5 / peak)
torchaudio.save(output_path, audio_normalized.unsqueeze(0).cpu(), 24000)
print(f"Saved: {output_path}")

# Analyze frame-by-frame timing
print("\n" + "=" * 60)
print("FRAME-BY-FRAME ANALYSIS")
print("=" * 60)

from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

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

# Profile individual frame generation
model.reset_caches()
gen_tokens, gen_mask = tokenize_text(test_text, 0)
curr_tokens = gen_tokens.unsqueeze(0)
curr_tokens_mask = gen_mask.unsqueeze(0)
curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)

frame_times = []
print("\nGenerating and timing each frame...")

for i in range(100):
    torch.cuda.synchronize()
    frame_start = time.time()

    sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50)

    torch.cuda.synchronize()
    frame_time = (time.time() - frame_start) * 1000
    frame_times.append(frame_time)

    if torch.all(sample == 0):
        break

    curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
    curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
    curr_pos = curr_pos[:, -1:] + 1

print(f"\nGenerated {len(frame_times)} frames")
print(f"First frame: {frame_times[0]:.1f}ms")
print(f"Average frame (after first): {sum(frame_times[1:])/len(frame_times[1:]):.1f}ms")
print(f"Min frame: {min(frame_times[1:]):.1f}ms")
print(f"Max frame: {max(frame_times[1:]):.1f}ms")
print(f"Target for RTF 1.0: 80ms")

# Show distribution
slow_frames = sum(1 for t in frame_times[1:] if t > 100)
very_slow_frames = sum(1 for t in frame_times[1:] if t > 200)
print(f"\nFrames > 100ms: {slow_frames} ({slow_frames/len(frame_times)*100:.0f}%)")
print(f"Frames > 200ms: {very_slow_frames} ({very_slow_frames/len(frame_times)*100:.0f}%)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"""
One-shot RTF: {rtf:.3f}x
Target: < 1.0 (i-LAVA achieved 0.383x on L4)

Gap analysis:
- We're {rtf/0.383:.1f}x slower than the paper
- Our frame time: {sum(frame_times)/len(frame_times):.0f}ms
- Target frame time: 80ms

The CPU overhead is still the bottleneck.
Need custom CUDA kernels or TensorRT to achieve real-time.
""")
