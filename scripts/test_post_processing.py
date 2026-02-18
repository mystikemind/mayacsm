#!/usr/bin/env python3
"""Quick test of the studio post-processing chain + UTMOS comparison."""

import sys
import os
import time
import re
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maya.engine.audio_post_processor import post_process, studio_process, normalize_lufs

# Test prompts
PROMPTS = [
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "hmm yeah that makes sense, ive been thinking about that too",
    "oh really",
    "yeah for sure",
]

def generate_audio(text, voice="jess"):
    """Generate audio via llama-server."""
    import requests
    AUDIO_TOKEN_BASE = 128266
    AUDIO_TOKEN_MIN = AUDIO_TOKEN_BASE
    AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * 4096) - 1
    CUSTOM_TOKEN_OFFSET = 128256

    prompt = f"<|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"
    payload = {
        "prompt": prompt, "max_tokens": 500,
        "temperature": 0.6, "top_p": 0.9, "top_k": 50, "min_p": 0.05,
        "repeat_penalty": 1.1, "repeat_last_n": 64,
        "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 3,
        "dry_penalty_last_n": 128,
        "logit_bias": [[128258, -3.0]],
        "stop": ["<custom_token_2>"], "stream": False,
    }
    resp = requests.post("http://127.0.0.1:5006/v1/completions", json=payload, timeout=60)
    resp.raise_for_status()
    text_output = resp.json()["choices"][0]["text"]

    token_ids = []
    for match in re.finditer(r'<custom_token_(\d+)>', text_output):
        custom_num = int(match.group(1))
        token_id = CUSTOM_TOKEN_OFFSET + custom_num
        if AUDIO_TOKEN_MIN <= token_id <= AUDIO_TOKEN_MAX:
            token_ids.append(token_id)

    return token_ids


def decode_snac(token_ids, snac, device):
    """Decode SNAC tokens to audio."""
    AUDIO_TOKEN_BASE = 128266
    n = (len(token_ids) // 7) * 7
    if n < 7:
        return None
    token_ids = token_ids[:n]
    codes = [t - AUDIO_TOKEN_BASE for t in token_ids]
    l0, l1, l2 = [], [], []
    for i in range(n // 7):
        b = 7 * i
        l0.append(max(0, min(4095, codes[b])))
        l1.append(max(0, min(4095, codes[b + 1] - 4096)))
        l2.append(max(0, min(4095, codes[b + 2] - 2 * 4096)))
        l2.append(max(0, min(4095, codes[b + 3] - 3 * 4096)))
        l1.append(max(0, min(4095, codes[b + 4] - 4 * 4096)))
        l2.append(max(0, min(4095, codes[b + 5] - 5 * 4096)))
        l2.append(max(0, min(4095, codes[b + 6] - 6 * 4096)))

    snac_codes = [
        torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(device),
    ]
    with torch.inference_mode():
        audio = snac.decode(snac_codes)
    return audio.squeeze().cpu().numpy()


def score_utmos(audio_np, utmos_model, device):
    """Score audio with UTMOS."""
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device))
    return score.item()


def main():
    device = "cuda:2"
    print("Loading SNAC + UTMOS...")
    from snac import SNAC
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    print("\n" + "=" * 70)
    print("  POST-PROCESSING A/B TEST")
    print("=" * 70)
    print("  Voice: jess | Device:", device)

    raw_scores = []
    processed_scores = []

    for i, text in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}: '{text[:50]}...'")

        tokens = generate_audio(text, "jess")
        if len(tokens) < 7:
            print("  SKIP: no tokens")
            continue

        audio_np = decode_snac(tokens, snac, device)
        if audio_np is None:
            print("  SKIP: decode failed")
            continue

        # Score raw audio
        raw_score = score_utmos(audio_np, utmos, device)

        # Apply post-processing
        t0 = time.time()
        processed = post_process(audio_np, sample_rate=24000)
        proc_time_ms = (time.time() - t0) * 1000

        # Score processed audio
        proc_score = score_utmos(processed, utmos, device)

        diff = proc_score - raw_score
        print(f"  Raw:       UTMOS {raw_score:.3f}")
        print(f"  Processed: UTMOS {proc_score:.3f} ({'+' if diff > 0 else ''}{diff:.3f})")
        print(f"  Proc time: {proc_time_ms:.1f}ms")
        print(f"  Audio: {len(audio_np)/24000:.2f}s raw, {len(processed)/24000:.2f}s processed")

        raw_scores.append(raw_score)
        processed_scores.append(proc_score)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    if raw_scores:
        print(f"  Raw mean:       UTMOS {np.mean(raw_scores):.3f} ± {np.std(raw_scores):.3f}")
        print(f"  Processed mean: UTMOS {np.mean(processed_scores):.3f} ± {np.std(processed_scores):.3f}")
        diff = np.mean(processed_scores) - np.mean(raw_scores)
        print(f"  Improvement:    {'+' if diff > 0 else ''}{diff:.3f}")


if __name__ == "__main__":
    main()
