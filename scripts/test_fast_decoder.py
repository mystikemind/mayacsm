#!/usr/bin/env python3
"""
Test ACTUAL fast decoder - stop the depth decoder loop early.

The key insight: we need to STOP the decoder loop early, not just truncate output.
This script modifies generate_frame to stop after N codebooks.
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import torchaudio
import time
import os

os.makedirs("/home/ec2-user/SageMaker/project_maya/tests/outputs/fast_decoder", exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda"

print("=" * 70)
print("FAST DECODER TEST - Early stopping of depth decoder loop")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

from huggingface_hub import hf_hub_download
from models import Model, sample_topk, _index_causal_mask
from moshi.models import loaders
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

# Load model
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


def generate_frame_fast(model, tokens, tokens_mask, input_pos, temperature, topk, num_codebooks=16):
    """
    Fast generate_frame that STOPS the depth decoder loop early.

    This is the KEY optimization - instead of running 31 decoder iterations,
    we only run (num_codebooks - 1) iterations.

    The remaining codebooks are padded with zeros.
    """
    dtype = next(model.parameters()).dtype
    b, s, _ = tokens.size()

    # Get the underlying model if compiled
    m = model._orig_mod if hasattr(model, '_orig_mod') else model

    # BACKBONE (same as original) - runs once per frame
    assert m.backbone.caches_are_enabled(), "backbone caches are not enabled"
    curr_backbone_mask = _index_causal_mask(m.backbone_causal_mask, input_pos)
    embeds = m._embed_tokens(tokens)
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)
    h = m.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

    # Generate codebook 0 (semantic) - backbone output
    last_h = h[:, -1, :]
    c0_logits = m.codebook0_head(last_h)
    c0_sample = sample_topk(c0_logits, topk, temperature)
    c0_embed = m._embed_audio(0, c0_sample)

    curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
    curr_sample = c0_sample.clone()
    curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

    # DEPTH DECODER - STOP EARLY after num_codebooks iterations
    m.decoder.reset_caches()
    for i in range(1, num_codebooks):  # Only run num_codebooks-1 iterations instead of 31
        curr_decoder_mask = _index_causal_mask(m.decoder_causal_mask, curr_pos)
        decoder_h = m.decoder(m.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(dtype=dtype)
        ci_logits = torch.mm(decoder_h[:, -1, :], m.audio_head[i - 1])
        ci_sample = sample_topk(ci_logits, topk, temperature)
        ci_embed = m._embed_audio(i, ci_sample)

        curr_h = ci_embed
        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
        curr_pos = curr_pos[:, -1:] + 1

    # Pad remaining codebooks with zeros (for Mimi compatibility)
    if num_codebooks < 32:
        padding = torch.zeros(b, 32 - num_codebooks, dtype=curr_sample.dtype, device=curr_sample.device)
        curr_sample = torch.cat([curr_sample, padding], dim=1)

    return curr_sample


def test_codebook_config(num_codebooks, test_text, max_frames=40):
    """Test a specific codebook configuration with actual early stopping."""
    print(f"\n--- Testing {num_codebooks} codebooks (actual early stopping) ---")

    mimi.set_num_codebooks(num_codebooks)

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
            sample = generate_frame_fast(model, curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50, num_codebooks)
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

    for i in range(max_frames):
        torch.cuda.synchronize()
        frame_start = time.time()

        sample = generate_frame_fast(model, curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50, num_codebooks)

        torch.cuda.synchronize()
        frame_time = (time.time() - frame_start) * 1000
        frame_times.append(frame_time)

        if torch.all(sample == 0):
            break

        frames.append(sample)
        curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1

    if not frames:
        print("  No frames generated!")
        return None

    # Decode audio
    stacked = torch.stack(frames).permute(1, 2, 0)
    stacked = stacked[:, :num_codebooks, :]
    audio = mimi.decode(stacked).squeeze()

    # Calculate metrics
    total_time = sum(frame_times) / 1000
    audio_duration = len(audio) / 24000
    rtf = total_time / audio_duration
    first_frame_time = frame_times[0]
    avg_frame_time = sum(frame_times) / len(frame_times)

    # Normalize and save
    audio = audio - audio.mean()
    peak = audio.abs().max()
    if peak > 0:
        audio = audio * (0.5 / peak)

    filename = f"fast_{num_codebooks}cb.wav"
    output_path = f"/home/ec2-user/SageMaker/project_maya/tests/outputs/fast_decoder/{filename}"
    torchaudio.save(output_path, audio.unsqueeze(0).detach().cpu(), 24000)

    status = "✓ REAL-TIME" if rtf < 1.0 else ("~ BUFFERABLE" if rtf < 1.3 else "✗ TOO SLOW")
    print(f"  Frames: {len(frames)}, Audio: {audio_duration:.2f}s")
    print(f"  First frame: {first_frame_time:.0f}ms, Avg frame: {avg_frame_time:.0f}ms")
    print(f"  RTF: {rtf:.2f}x {status}")
    print(f"  Saved: {filename}")

    return {
        'codebooks': num_codebooks,
        'rtf': rtf,
        'first_frame_ms': first_frame_time,
        'avg_frame_ms': avg_frame_time,
        'audio_duration': audio_duration,
        'status': status
    }


# Test configurations
test_text = "hello how are you doing today im happy to help you"

codebook_counts = [32, 24, 20, 16, 12, 8, 4]

print("\n" + "=" * 70)
print("TESTING WITH ACTUAL EARLY STOPPING")
print("=" * 70)

results = []
for num_cb in codebook_counts:
    result = test_codebook_config(num_cb, test_text)
    if result:
        results.append(result)
    torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 70)
print("SUMMARY - ACTUAL EARLY STOPPING")
print("=" * 70)
print(f"{'Codebooks':<12} {'RTF':<12} {'First(ms)':<12} {'Avg(ms)':<12} {'Status':<15}")
print("-" * 70)

for r in results:
    print(f"{r['codebooks']:<12} {r['rtf']:.2f}x{'':<7} {r['first_frame_ms']:.0f}{'':<8} {r['avg_frame_ms']:.0f}{'':<8} {r['status']}")

# Find optimal
print("\n" + "=" * 70)
print("OPTIMAL CONFIGURATION")
print("=" * 70)

realtime_configs = [r for r in results if r['rtf'] < 1.0]
bufferable_configs = [r for r in results if r['rtf'] < 1.3]

if realtime_configs:
    best = max(realtime_configs, key=lambda x: x['codebooks'])
    print(f"Best REAL-TIME (RTF < 1.0): {best['codebooks']} codebooks")
    print(f"  RTF: {best['rtf']:.2f}x, Frame time: {best['avg_frame_ms']:.0f}ms")
elif bufferable_configs:
    best = max(bufferable_configs, key=lambda x: x['codebooks'])
    print(f"Best BUFFERABLE (RTF < 1.3): {best['codebooks']} codebooks")
    print(f"  RTF: {best['rtf']:.2f}x, Frame time: {best['avg_frame_ms']:.0f}ms")
else:
    print("No configuration achieved acceptable RTF")
    # Find closest to real-time
    best = min(results, key=lambda x: x['rtf'])
    print(f"Closest to real-time: {best['codebooks']} codebooks")
    print(f"  RTF: {best['rtf']:.2f}x, Frame time: {best['avg_frame_ms']:.0f}ms")

print(f"\nAudio files saved to: tests/outputs/fast_decoder/")
print("=" * 70)
