#!/usr/bin/env python3
"""
Benchmark All 8 Orpheus Voices for Conversational Quality
==========================================================

Tests each voice on 15 conversational prompts, measuring UTMOS.
Also tests new sampling parameters vs old parameters.

Usage:
    python scripts/benchmark_voices.py

Requires: llama-server running on port 5006
"""

import sys
import os
import time
import json
import re
import logging
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

# All 8 Orpheus voices (ranked by Canopy Labs conversational realism)
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

# Conversational test prompts (mix of short, medium, emotional)
TEST_PROMPTS = [
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "aww that sounds really tough, im sorry youre dealing with that",
    "oh my gosh thats hilarious, i cant believe that happened",
    "hmm yeah that makes sense, ive been thinking about that too",
    "honestly i think thats a great idea, you should definitely go for it",
    "yeah i know what you mean, sometimes things just dont work out",
    "oh really",
    "yeah for sure",
    "hmm thats interesting",
    "well you know what i think, lets just go for it",
    "that reminds me of something that happened last week",
    "mhm",
    "wow thats amazing",
    "okay cool, thanks for letting me know",
]

# Old sampling params (what we had before)
OLD_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}

# New sampling params (research-backed)
NEW_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 50,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 3,
    "dry_penalty_last_n": 128,
    "logit_bias": [[128258, -3.0]],
}


def setup_snac_and_utmos(device="cuda:2"):
    """Load SNAC codec and UTMOS scorer."""
    from snac import SNAC
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

    utmos = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
        trust_repo=True
    ).to(device).eval()

    return snac, utmos


def generate_audio(text, voice, params, server_url="http://127.0.0.1:5006", snac=None, device="cuda:2"):
    """Generate audio for text using specified voice and params."""
    import requests

    AUDIO_TOKEN_BASE = 128266
    AUDIO_TOKEN_MIN = AUDIO_TOKEN_BASE
    AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * 4096) - 1
    CUSTOM_TOKEN_OFFSET = 128256

    prompt = f"<|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"

    payload = {
        "prompt": prompt,
        "max_tokens": 500,
        "stop": ["<custom_token_2>"],
        "stream": False,
        **params,
    }

    try:
        resp = requests.post(f"{server_url}/v1/completions", json=payload, timeout=60)
        resp.raise_for_status()
        text_output = resp.json()["choices"][0]["text"]
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return None

    # Extract audio tokens
    token_ids = []
    for match in re.finditer(r'<custom_token_(\d+)>', text_output):
        custom_num = int(match.group(1))
        token_id = CUSTOM_TOKEN_OFFSET + custom_num
        if AUDIO_TOKEN_MIN <= token_id <= AUDIO_TOKEN_MAX:
            token_ids.append(token_id)

    if len(token_ids) < 7:
        return None

    # Decode via SNAC
    n = (len(token_ids) // 7) * 7
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

    return audio.squeeze().cpu()


def score_utmos(audio, utmos_model, device="cuda:2"):
    """Score audio with UTMOS."""
    if audio is None or audio.numel() < 2400:  # < 100ms
        return 0.0

    wav = audio.unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device))
    return score.item()


def benchmark_voice(voice, prompts, params, snac, utmos, device="cuda:2"):
    """Benchmark a voice across all prompts."""
    scores = []
    for i, text in enumerate(prompts):
        audio = generate_audio(text, voice, params, snac=snac, device=device)
        score = score_utmos(audio, utmos, device)
        scores.append(score)
        logger.info(f"  [{i+1:2d}/{len(prompts)}] {text[:50]:50s} → UTMOS={score:.3f}")
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--voices", nargs="*", default=None,
                       help="Specific voices to test (default: all 8)")
    parser.add_argument("--compare-params", action="store_true",
                       help="Also compare old vs new sampling params")
    parser.add_argument("--output", default="scripts/voice_benchmark_results.json")
    args = parser.parse_args()

    voices = args.voices or VOICES
    device = args.device

    print("=" * 70)
    print("  ORPHEUS VOICE BENCHMARK")
    print("=" * 70)
    print(f"  Voices: {', '.join(voices)}")
    print(f"  Prompts: {len(TEST_PROMPTS)}")
    print(f"  Device: {device}")
    print()

    # Load models
    print("Loading SNAC + UTMOS...")
    snac, utmos = setup_snac_and_utmos(device)
    print("Ready.\n")

    results = {}

    # Benchmark each voice with new params
    for voice in voices:
        print(f"\n{'='*50}")
        print(f"  Voice: {voice} (new params)")
        print(f"{'='*50}")
        scores = benchmark_voice(voice, TEST_PROMPTS, NEW_PARAMS, snac, utmos, device)
        mean = np.mean(scores)
        std = np.std(scores)
        results[voice] = {
            "mean": mean,
            "std": std,
            "min": min(scores),
            "max": max(scores),
            "scores": scores,
            "params": "new",
        }
        print(f"\n  {voice}: UTMOS {mean:.3f} ± {std:.3f} [{min(scores):.3f}-{max(scores):.3f}]")

    # Optionally compare old vs new params on best voice
    if args.compare_params:
        best_voice = max(results.keys(), key=lambda v: results[v]["mean"])
        print(f"\n{'='*50}")
        print(f"  Comparing params on best voice: {best_voice}")
        print(f"{'='*50}")
        print(f"\n  OLD params:")
        old_scores = benchmark_voice(best_voice, TEST_PROMPTS, OLD_PARAMS, snac, utmos, device)
        old_mean = np.mean(old_scores)
        old_std = np.std(old_scores)
        results[f"{best_voice}_old_params"] = {
            "mean": old_mean,
            "std": old_std,
            "min": min(old_scores),
            "max": max(old_scores),
            "scores": old_scores,
            "params": "old",
        }
        print(f"\n  {best_voice} OLD: UTMOS {old_mean:.3f} ± {old_std:.3f}")
        print(f"  {best_voice} NEW: UTMOS {results[best_voice]['mean']:.3f} ± {results[best_voice]['std']:.3f}")
        diff = results[best_voice]["mean"] - old_mean
        print(f"  Difference: {'+' if diff > 0 else ''}{diff:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  VOICE RANKING (by UTMOS)")
    print(f"{'='*70}")
    voice_only = {k: v for k, v in results.items() if "_old" not in k}
    ranked = sorted(voice_only.items(), key=lambda x: x[1]["mean"], reverse=True)
    for rank, (voice, data) in enumerate(ranked, 1):
        indicator = " ← CURRENT" if voice == "zoe" else (" ← BEST" if rank == 1 else "")
        print(f"  #{rank}: {voice:6s} UTMOS {data['mean']:.3f} ± {data['std']:.3f} "
              f"[{data['min']:.3f}-{data['max']:.3f}]{indicator}")

    # Save results
    output_path = args.output
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompts": TEST_PROMPTS,
        "new_params": NEW_PARAMS,
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
