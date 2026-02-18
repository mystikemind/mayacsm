#!/usr/bin/env python3
"""
Test Orpheus 3B TTS Quality

Uses the transformers backend (no vLLM) to test Orpheus quality
on the same sentences as the CSM comparison.

Orpheus uses SNAC codec (24kHz) - same as our CSM pipeline.
"""

import sys
import os
import time
import json
import torch
import numpy as np
import soundfile as sf

OUTPUT_DIR = "/home/ec2-user/SageMaker/project_maya/audio_comparison_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same test sentences as CSM comparison
TEST_SENTENCES = [
    "hmm, that's actually really interesting, tell me more",
    "oh yeah, I totally get what you mean",
    "aww, that sounds tough, I'm here for you",
    "haha, no way, that's hilarious",
    "well, I think you should just go for it, you know?",
    "yeah, I'm doing pretty good, thanks for asking",
    "oh really? I didn't know that, that's cool",
    "hmm, let me think about that for a second",
]

# Sentences with paralinguistic tags (Orpheus exclusive feature)
TAG_SENTENCES = [
    "I can't believe you did that <laugh> that's absolutely hilarious",
    "oh no <sigh> I guess we'll have to start over",
    "<gasp> you scared me! don't do that again",
    "yeah I was thinking about it and uhm I'm not really sure what to do",
    "that's so sweet of you <chuckle> you really didn't have to",
    "I'm so tired <yawn> maybe we should call it a night",
]


def load_orpheus(device="cuda:2"):
    """Load Orpheus 3B using transformers (no vLLM needed)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from snac import SNAC

    print("=" * 60)
    print("LOADING ORPHEUS 3B")
    print("=" * 60)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft")

    print("Loading model (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    print("Loading SNAC codec...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

    print("Models loaded!")
    return model, tokenizer, snac_model, device


def decode_snac(snac_model, code_list, device):
    """Decode SNAC token codes into audio waveform."""
    layer_1, layer_2, layer_3 = [], [], []

    num_complete = len(code_list) // 7
    for i in range(num_complete):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))

    if not layer_1:
        return np.zeros(24000, dtype=np.float32)

    codes = [
        torch.tensor(layer_1).unsqueeze(0).to(device),
        torch.tensor(layer_2).unsqueeze(0).to(device),
        torch.tensor(layer_3).unsqueeze(0).to(device),
    ]

    with torch.no_grad():
        audio = snac_model.decode(codes).squeeze().cpu().numpy()

    return audio


def generate_speech(model, tokenizer, snac_model, device, text, voice="tara"):
    """Generate speech using Orpheus 3B."""
    # Build prompt
    prompt = f"{voice}: {text}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Add special tokens
    start_token = torch.tensor([[128259]], dtype=torch.long, device=device)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.long, device=device)
    modified_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    # Pad to expected length
    pad_len = max(0, 2048 - modified_ids.shape[1])
    if pad_len > 0:
        pad = torch.full((1, pad_len), 128263, dtype=torch.long, device=device)
        modified_ids = torch.cat([pad, modified_ids], dim=1)
        attention_mask = torch.cat([
            torch.zeros((1, pad_len), dtype=torch.long, device=device),
            torch.ones((1, modified_ids.shape[1] - pad_len), dtype=torch.long, device=device),
        ], dim=1)
    else:
        attention_mask = torch.ones_like(modified_ids)

    start = time.time()

    with torch.no_grad():
        generated = model.generate(
            input_ids=modified_ids,
            attention_mask=attention_mask,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=128258,
            use_cache=True,
        )

    gen_time = time.time() - start

    # Extract audio tokens (after token 128257)
    token_to_find = 128257
    indices = (generated == token_to_find).nonzero(as_tuple=True)
    if len(indices[1]) > 0:
        last_idx = indices[1][-1].item()
        cropped = generated[:, last_idx + 1:]
    else:
        cropped = generated

    # Remove stop tokens and clean
    cleaned = cropped[cropped != 128258]
    cleaned = cleaned[cleaned != 128263]

    # Trim to multiple of 7
    trimmed = cleaned[:len(cleaned) // 7 * 7]
    if len(trimmed) == 0:
        return np.zeros(24000, dtype=np.float32), gen_time

    # Convert to SNAC codes
    codes = [int(t) - 128266 for t in trimmed]

    # Decode audio
    audio = decode_snac(snac_model, codes, device)

    # Normalize
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio * (0.8 / peak)

    return audio, gen_time


def analyze_audio(audio, sr=24000, label=""):
    """Analyze audio quality metrics."""
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.abs(audio).max()
    duration = len(audio) / sr
    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio)) if len(audio) > 1 else 0
    crest_factor = peak / rms if rms > 0 else 0

    # Silence ratio
    frame_size = int(0.025 * sr)
    silence_count = total_frames = 0
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i+frame_size]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        total_frames += 1
        if frame_rms < 0.01:
            silence_count += 1
    silence_ratio = silence_count / max(total_frames, 1)

    print(f"  {label}")
    print(f"    Duration: {duration:.2f}s")
    print(f"    RMS: {rms:.4f}, Peak: {peak:.4f}")
    print(f"    ZCR: {zcr:.4f}")
    print(f"    Crest factor: {crest_factor:.2f}")
    print(f"    Silence ratio: {silence_ratio:.1%}")

    return {
        "duration": duration, "rms": float(rms), "peak": float(peak),
        "zcr": float(zcr), "crest_factor": float(crest_factor),
        "silence_ratio": float(silence_ratio),
    }


def main():
    print("=" * 70)
    print("ORPHEUS 3B TTS QUALITY TEST")
    print("Testing human-likeness with paralinguistic tags")
    print("=" * 70)

    model, tokenizer, snac_model, device = load_orpheus(device="cuda:2")

    # Test all available voices
    voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        generate_speech(model, tokenizer, snac_model, device, "hello there", voice="tara")

    results = {"standard": [], "with_tags": []}

    # Test 1: Standard sentences (same as CSM test)
    print("\n" + "=" * 70)
    print("TEST 1: STANDARD SENTENCES (same as CSM comparison)")
    print("=" * 70)

    for i, text in enumerate(TEST_SENTENCES):
        print(f"\n--- Sentence {i+1}: '{text}' ---")
        audio, gen_time = generate_speech(model, tokenizer, snac_model, device, text, voice="tara")
        path = os.path.join(OUTPUT_DIR, f"orpheus_{i+1:02d}.wav")
        sf.write(path, audio, 24000)
        metrics = analyze_audio(audio, label=f"ORPHEUS tara (gen: {gen_time*1000:.0f}ms)")
        results["standard"].append({"text": text, "time_ms": gen_time * 1000, **metrics})

    # Test 2: Sentences with paralinguistic tags
    print("\n" + "=" * 70)
    print("TEST 2: PARALINGUISTIC TAGS (Orpheus exclusive)")
    print("=" * 70)

    for i, text in enumerate(TAG_SENTENCES):
        print(f"\n--- Tag sentence {i+1}: '{text}' ---")
        audio, gen_time = generate_speech(model, tokenizer, snac_model, device, text, voice="tara")
        path = os.path.join(OUTPUT_DIR, f"orpheus_tags_{i+1:02d}.wav")
        sf.write(path, audio, 24000)
        metrics = analyze_audio(audio, label=f"ORPHEUS tags (gen: {gen_time*1000:.0f}ms)")
        results["with_tags"].append({"text": text, "time_ms": gen_time * 1000, **metrics})

    # Test 3: Multiple voices (find best for Maya)
    print("\n" + "=" * 70)
    print("TEST 3: VOICE COMPARISON")
    print("=" * 70)

    test_text = "yeah I'm doing pretty good, thanks for asking. how about you?"
    for voice in voices:
        print(f"\n--- Voice: {voice} ---")
        audio, gen_time = generate_speech(model, tokenizer, snac_model, device, test_text, voice=voice)
        path = os.path.join(OUTPUT_DIR, f"orpheus_voice_{voice}.wav")
        sf.write(path, audio, 24000)
        analyze_audio(audio, label=f"ORPHEUS {voice} (gen: {gen_time*1000:.0f}ms)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    std_times = [r["time_ms"] for r in results["standard"]]
    tag_times = [r["time_ms"] for r in results["with_tags"]]
    std_durations = [r["duration"] for r in results["standard"]]
    std_zcr = [r["zcr"] for r in results["standard"]]

    print(f"\nStandard sentences:")
    print(f"  Avg generation time: {np.mean(std_times):.0f}ms")
    print(f"  Avg audio duration: {np.mean(std_durations):.2f}s")
    print(f"  Avg ZCR: {np.mean(std_zcr):.4f}")
    print(f"\nTag sentences:")
    print(f"  Avg generation time: {np.mean(tag_times):.0f}ms")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "orpheus_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudio saved to: {OUTPUT_DIR}/")
    print(f"Results saved to: {results_path}")

    # VRAM usage
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    print(f"\nVRAM Usage: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


if __name__ == "__main__":
    main()
