#!/usr/bin/env python3
"""
Audio Quality Test - Generate samples at different codebook levels.

Listen to these BEFORE going live to verify:
1. Words are clear and understandable
2. No clicking/popping between chunks
3. No random noises
4. Audio doesn't cut off mid-sentence
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import torchaudio
import time
import os

os.makedirs("/home/ec2-user/SageMaker/project_maya/tests/outputs/quality_test", exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda"

print("=" * 70)
print("AUDIO QUALITY TEST - Listen before going live!")
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
    """Fast generate_frame that stops depth decoder early."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    dtype = next(m.parameters()).dtype
    b, s, _ = tokens.size()

    # Backbone
    assert m.backbone.caches_are_enabled(), "backbone caches are not enabled"
    curr_backbone_mask = _index_causal_mask(m.backbone_causal_mask, input_pos)
    embeds = m._embed_tokens(tokens)
    masked_embeds = embeds * tokens_mask.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)
    h = m.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

    # Codebook 0
    last_h = h[:, -1, :]
    c0_logits = m.codebook0_head(last_h)
    c0_sample = sample_topk(c0_logits, topk, temperature)
    c0_embed = m._embed_audio(0, c0_sample)

    curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
    curr_sample = c0_sample.clone()
    curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

    # Depth decoder - stop early
    m.decoder.reset_caches()
    for i in range(1, num_codebooks):
        curr_decoder_mask = _index_causal_mask(m.decoder_causal_mask, curr_pos)
        decoder_h = m.decoder(m.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(dtype=dtype)
        ci_logits = torch.mm(decoder_h[:, -1, :], m.audio_head[i - 1])
        ci_sample = sample_topk(ci_logits, topk, temperature)
        ci_embed = m._embed_audio(i, ci_sample)

        curr_h = ci_embed
        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
        curr_pos = curr_pos[:, -1:] + 1

    # Pad remaining codebooks
    if num_codebooks < 32:
        padding = torch.zeros(b, 32 - num_codebooks, dtype=curr_sample.dtype, device=curr_sample.device)
        curr_sample = torch.cat([curr_sample, padding], dim=1)

    return curr_sample


def generate_audio(text, num_codebooks, max_frames=60):
    """Generate audio with specific codebook count."""
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

    for _ in range(2):
        model.reset_caches()
        curr_tokens = gen_tokens.unsqueeze(0)
        curr_tokens_mask = gen_mask.unsqueeze(0)
        curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)
        for i in range(3):
            sample = generate_frame_fast(model, curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50, num_codebooks)
            if torch.all(sample == 0):
                break
            curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

    # Generate
    model.reset_caches()
    gen_tokens, gen_mask = tokenize_text(text, 0)
    curr_tokens = gen_tokens.unsqueeze(0)
    curr_tokens_mask = gen_mask.unsqueeze(0)
    curr_pos = torch.arange(0, gen_tokens.size(0)).unsqueeze(0).long().to(device)

    frames = []
    frame_times = []

    torch.cuda.synchronize()
    start = time.time()

    for i in range(max_frames):
        torch.cuda.synchronize()
        frame_start = time.time()

        sample = generate_frame_fast(model, curr_tokens, curr_tokens_mask, curr_pos, 0.8, 50, num_codebooks)

        torch.cuda.synchronize()
        frame_times.append((time.time() - frame_start) * 1000)

        if torch.all(sample == 0):
            break

        frames.append(sample)
        curr_tokens = torch.cat([sample, zero_token], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([ones_mask, zero_mask], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1

    torch.cuda.synchronize()
    total_time = time.time() - start

    if not frames:
        return None, 0, 0

    # Decode
    stacked = torch.stack(frames).permute(1, 2, 0)
    stacked = stacked[:, :num_codebooks, :]
    audio = mimi.decode(stacked).squeeze()

    # Normalize
    audio = audio - audio.mean()
    peak = audio.abs().max()
    if peak > 0:
        audio = audio * (0.5 / peak)

    audio_duration = len(audio) / 24000
    rtf = total_time / audio_duration
    first_frame_ms = frame_times[0]

    return audio, rtf, first_frame_ms


# Test sentences - various types
test_sentences = [
    ("short", "hi how are you"),
    ("medium", "im doing great thanks for asking how can i help you today"),
    ("question", "what would you like to know about"),
    ("statement", "thats a really interesting question let me think about that"),
]

# Test configurations
configs = [
    (8, "8cb_fast"),   # Our current config (RTF ~1.09x)
    (4, "4cb_ultra"),  # Ultra fast (RTF ~0.68x) - might be lower quality
    (12, "12cb_balanced"),  # Middle ground
]

print("\n" + "=" * 70)
print("GENERATING TEST AUDIO FILES")
print("=" * 70)

results = []

for num_cb, config_name in configs:
    print(f"\n--- {config_name.upper()} ({num_cb} codebooks) ---")

    for sentence_type, text in test_sentences:
        print(f"  Generating: '{text[:40]}...'")

        audio, rtf, first_ms = generate_audio(text, num_cb)

        if audio is not None:
            filename = f"{config_name}_{sentence_type}.wav"
            output_path = f"/home/ec2-user/SageMaker/project_maya/tests/outputs/quality_test/{filename}"
            torchaudio.save(output_path, audio.unsqueeze(0).detach().cpu(), 24000)

            duration = len(audio) / 24000
            status = "✓ RT" if rtf < 1.0 else ("~ BUF" if rtf < 1.2 else "✗ SLOW")
            print(f"    → {filename}: {duration:.1f}s, RTF={rtf:.2f}x {status}, first={first_ms:.0f}ms")

            results.append({
                'config': config_name,
                'type': sentence_type,
                'rtf': rtf,
                'first_ms': first_ms,
                'duration': duration
            })

        torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for config_name, _ in configs:
    config_results = [r for r in results if r['config'] == config_name]
    if config_results:
        avg_rtf = sum(r['rtf'] for r in config_results) / len(config_results)
        avg_first = sum(r['first_ms'] for r in config_results) / len(config_results)
        print(f"{config_name}: Avg RTF={avg_rtf:.2f}x, Avg First Frame={avg_first:.0f}ms")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
1. Listen to the audio files in: tests/outputs/quality_test/
2. Check for:
   - Clear, understandable words
   - No clicking or popping
   - No random noise
   - Complete sentences (no cutoff)
3. Compare 4cb vs 8cb vs 12cb quality
4. If 4cb is acceptable, use it (true real-time RTF 0.68x)
5. If 8cb sounds better, use it (bufferable RTF 1.09x)
""")
print("=" * 70)
