#!/usr/bin/env python3
"""
Find optimal codebook count for speed/quality tradeoff.

Tests different codebook configurations and measures:
1. RTF (must be < 1.0 for real-time, < 1.3 for bufferable)
2. First chunk latency
3. Audio quality (save files for listening comparison)
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import torchaudio
import time
import os

os.makedirs("/home/ec2-user/SageMaker/project_maya/tests/outputs/codebook_test", exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda"

print("=" * 70)
print("FINDING OPTIMAL CODEBOOK COUNT")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

# Load model once
print("\nLoading CSM-1B...")
model = Model.from_pretrained("sesame/csm-1b")
model.to(device=device, dtype=torch.bfloat16)
model.setup_caches(1)

# Load Mimi
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)

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


def test_codebook_config(num_codebooks_gen, num_codebooks_decode, test_text, max_frames=40):
    """
    Test a specific codebook configuration.

    Args:
        num_codebooks_gen: Number of codebooks to generate with CSM (affects speed)
        num_codebooks_decode: Number of codebooks to use in Mimi decoder (affects quality)
    """
    print(f"\n--- Testing: Generate {num_codebooks_gen} codebooks, Decode with {num_codebooks_decode} ---")

    mimi.set_num_codebooks(num_codebooks_decode)

    # Warmup
    model.reset_caches()
    gen_tokens, gen_mask = tokenize_text("hi", 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    ones_mask = torch.ones(1, 32, dtype=torch.bool, device=device)

    for _ in range(3):
        model.reset_caches()
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

    # Actual test
    model.reset_caches()
    gen_tokens, gen_mask = tokenize_text(test_text, 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    frames = []
    frame_times = []

    torch.cuda.synchronize()
    total_start = time.time()

    for i in range(max_frames):
        torch.cuda.synchronize()
        frame_start = time.time()

        sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50)

        torch.cuda.synchronize()
        frame_time = (time.time() - frame_start) * 1000
        frame_times.append(frame_time)

        if torch.all(sample == 0):
            break

        # Truncate to num_codebooks_gen if needed
        if num_codebooks_gen < 32:
            sample_truncated = sample.clone()
            sample_truncated[:, num_codebooks_gen:] = 0
            frames.append(sample_truncated)
        else:
            frames.append(sample)

        curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1

    torch.cuda.synchronize()
    total_time = time.time() - total_start

    if not frames:
        print("  No frames generated!")
        return None

    # Decode audio
    stacked = torch.stack(frames).permute(1, 2, 0)  # (batch, codebooks, frames)
    stacked = stacked[:, :num_codebooks_decode, :]  # Use only first N codebooks
    audio = mimi.decode(stacked).squeeze()

    # Calculate metrics
    audio_duration = len(audio) / 24000
    rtf = total_time / audio_duration
    first_chunk_time = frame_times[0] if frame_times else 0
    avg_frame_time = sum(frame_times) / len(frame_times)

    # Normalize and save audio
    audio = audio - audio.mean()
    peak = audio.abs().max()
    if peak > 0:
        audio = audio * (0.5 / peak)

    filename = f"gen{num_codebooks_gen}_dec{num_codebooks_decode}.wav"
    output_path = f"/home/ec2-user/SageMaker/project_maya/tests/outputs/codebook_test/{filename}"
    torchaudio.save(output_path, audio.unsqueeze(0).detach().cpu(), 24000)

    status = "✓ REAL-TIME" if rtf < 1.0 else ("~ BUFFERABLE" if rtf < 1.3 else "✗ TOO SLOW")
    print(f"  Frames: {len(frames)}, Audio: {audio_duration:.2f}s")
    print(f"  First frame: {first_chunk_time:.0f}ms, Avg frame: {avg_frame_time:.0f}ms")
    print(f"  RTF: {rtf:.2f}x {status}")
    print(f"  Saved: {filename}")

    return {
        'gen_codebooks': num_codebooks_gen,
        'dec_codebooks': num_codebooks_decode,
        'rtf': rtf,
        'first_frame_ms': first_chunk_time,
        'avg_frame_ms': avg_frame_time,
        'audio_duration': audio_duration,
        'status': status
    }


# Test configurations
test_text = "hello how are you doing today im happy to help you"

configs = [
    # (gen_codebooks, dec_codebooks)
    (32, 32),  # Full quality (baseline)
    (32, 16),  # Full gen, reduced decode (tests if decode affects quality)
    (24, 24),  # Slightly reduced
    (20, 20),  # More reduced
    (16, 16),  # Paper's recommended
    (12, 12),  # More aggressive
    (8, 8),    # Very aggressive
]

print("\n" + "=" * 70)
print("TESTING CONFIGURATIONS")
print("=" * 70)

results = []
for gen_cb, dec_cb in configs:
    result = test_codebook_config(gen_cb, dec_cb, test_text)
    if result:
        results.append(result)
    torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Config':<15} {'RTF':<10} {'First(ms)':<12} {'Avg(ms)':<12} {'Status':<15}")
print("-" * 70)

for r in results:
    config = f"{r['gen_codebooks']}/{r['dec_codebooks']}"
    print(f"{config:<15} {r['rtf']:.2f}x{'':<6} {r['first_frame_ms']:.0f}{'':<8} {r['avg_frame_ms']:.0f}{'':<8} {r['status']}")

# Recommendation
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

realtime_configs = [r for r in results if r['rtf'] < 1.0]
bufferable_configs = [r for r in results if r['rtf'] < 1.3]

if realtime_configs:
    best_quality_realtime = max(realtime_configs, key=lambda x: x['gen_codebooks'])
    print(f"Best for REAL-TIME (RTF < 1.0):")
    print(f"  → {best_quality_realtime['gen_codebooks']} codebooks")
    print(f"     RTF: {best_quality_realtime['rtf']:.2f}x, Frame: {best_quality_realtime['avg_frame_ms']:.0f}ms")
else:
    print("No configuration achieved real-time RTF < 1.0")

if bufferable_configs:
    best_quality_bufferable = max(bufferable_configs, key=lambda x: x['gen_codebooks'])
    print(f"\nBest for BUFFERED (RTF < 1.3):")
    print(f"  → {best_quality_bufferable['gen_codebooks']} codebooks")
    print(f"     RTF: {best_quality_bufferable['rtf']:.2f}x, Frame: {best_quality_bufferable['avg_frame_ms']:.0f}ms")
    print(f"     Needs ~300ms client buffer to smooth gaps")

print(f"\nAudio files saved to: tests/outputs/codebook_test/")
print("Listen to compare quality at different codebook levels.")
print("=" * 70)
