#!/usr/bin/env python3
"""
Quick Voice Benchmark - Test all Orpheus voices with best config.
Based on NO_VOICE results: "creative" (T=0.8) scored highest UTMOS=4.188.
"""
import os
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/.cache/huggingface"
_cudnn = os.path.join(os.path.dirname(os.__file__), "..", "nvidia", "cudnn", "lib")
if os.path.isdir(_cudnn):
    os.environ["LD_LIBRARY_PATH"] = _cudnn + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import torchaudio
import numpy as np
import time
import json
import re
import requests
from pathlib import Path
from snac import SNAC

SERVER_URL = "http://127.0.0.1:5006"
OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/audio_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)

VOICES = [None, "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

# Diverse conversational utterances
TEST_UTTERANCES = [
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
]

# Best config from NO_VOICE results + also test expressive
CONFIGS = {
    "creative": {"temperature": 0.8, "top_p": 0.95, "repeat_penalty": 1.15},
    "expressive": {"temperature": 0.75, "top_p": 0.98, "repeat_penalty": 1.05},
}

# ============================================================================
CUSTOM_TOKEN_OFFSET = 128256
AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * 4096) - 1

print("Loading SNAC codec...", flush=True)
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda:2" if torch.cuda.is_available() else "cpu"
snac_model = snac_model.to(snac_device)

print("Loading UTMOS...", flush=True)
try:
    utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos_model = utmos_model.to(snac_device).eval()
    HAS_UTMOS = True
except Exception as e:
    print(f"UTMOS not available: {e}")
    HAS_UTMOS = False


def extract_audio_tokens(text_output):
    token_ids = []
    for match in re.finditer(r'<custom_token_(\d+)>', text_output):
        custom_num = int(match.group(1))
        token_id = CUSTOM_TOKEN_OFFSET + custom_num
        if AUDIO_TOKEN_BASE <= token_id <= AUDIO_TOKEN_MAX:
            token_ids.append(token_id)
    return token_ids


def decode_snac_frames(token_ids):
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
        torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(snac_device),
        torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(snac_device),
        torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(snac_device),
    ]
    with torch.inference_mode():
        audio = snac_model.decode(snac_codes)
    return audio.squeeze().cpu()


def generate_speech(text, voice, params):
    if voice:
        prompt = f"<|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"
    else:
        prompt = f"<|begin_of_text|><custom_token_3>{text}<custom_token_4><custom_token_5>"
    payload = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "repeat_penalty": params["repeat_penalty"],
        "stop": ["<custom_token_2>"],
        "stream": False,
    }
    t0 = time.time()
    resp = requests.post(f"{SERVER_URL}/v1/completions", json=payload, timeout=120)
    gen_time = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    text_output = data["choices"][0]["text"]
    token_ids = extract_audio_tokens(text_output)
    if not token_ids:
        return None, gen_time, 0
    audio = decode_snac_frames(token_ids)
    if audio is None:
        return None, gen_time, 0
    duration = audio.shape[-1] / 24000
    rtf = gen_time / duration if duration > 0 else float('inf')
    return audio, gen_time, rtf


def compute_utmos(audio):
    if not HAS_UTMOS or audio is None:
        return None
    try:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        resampled = torchaudio.functional.resample(audio, 24000, 16000)
        resampled = resampled.to(snac_device)
        with torch.inference_mode():
            score = utmos_model(resampled, 16000)
        return float(score.item())
    except:
        return None


# ============================================================================
print(f"\n{'='*80}", flush=True)
print(f"  ORPHEUS VOICE BENCHMARK", flush=True)
print(f"  {len(VOICES)} voices x {len(CONFIGS)} configs x {len(TEST_UTTERANCES)} utterances = {len(VOICES)*len(CONFIGS)*len(TEST_UTTERANCES)} samples", flush=True)
print(f"{'='*80}", flush=True)

try:
    r = requests.get(f"{SERVER_URL}/health", timeout=5)
    print(f"llama-server: OK", flush=True)
except:
    print("ERROR: llama-server not reachable")
    sys.exit(1)

# Warmup
generate_speech("hello", "tara", CONFIGS["creative"])
generate_speech("how are you", "tara", CONFIGS["creative"])

all_scores = []

for voice in VOICES:
    voice_label = voice or "NO_VOICE"
    for config_name, params in CONFIGS.items():
        scores = []
        rtfs = []

        print(f"\n--- {voice_label} + {config_name} ---", flush=True)

        for i, text in enumerate(TEST_UTTERANCES):
            audio, gen_time, rtf = generate_speech(text, voice, params)
            if audio is not None and audio.numel() > 0:
                utmos = compute_utmos(audio)
                # Save audio
                voice_dir = OUTPUT_DIR / voice_label / config_name
                voice_dir.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(voice_dir / f"sample_{i:02d}.wav"), audio.unsqueeze(0), 24000)
                if utmos:
                    scores.append(utmos)
                rtfs.append(rtf)
                dur = audio.shape[-1] / 24000
                print(f"  [{i+1}/{len(TEST_UTTERANCES)}] UTMOS={utmos:.3f} RTF={rtf:.2f} Dur={dur:.1f}s '{text[:35]}...'", flush=True)
            else:
                print(f"  [{i+1}/{len(TEST_UTTERANCES)}] FAILED", flush=True)

        if scores:
            entry = {
                "voice": voice_label,
                "config": config_name,
                "avg_utmos": float(np.mean(scores)),
                "std_utmos": float(np.std(scores)),
                "min_utmos": float(min(scores)),
                "max_utmos": float(max(scores)),
                "avg_rtf": float(np.mean(rtfs)),
                "n": len(scores),
            }
            all_scores.append(entry)
            print(f"  => UTMOS={entry['avg_utmos']:.3f}±{entry['std_utmos']:.3f} RTF={entry['avg_rtf']:.2f}", flush=True)

# ============================================================================
print(f"\n{'='*80}", flush=True)
print(f"  FINAL RANKINGS (sorted by avg UTMOS)", flush=True)
print(f"{'='*80}", flush=True)

all_scores.sort(key=lambda x: x["avg_utmos"], reverse=True)

print(f"\n{'Rank':<5} {'Voice':<12} {'Config':<12} {'UTMOS':<18} {'RTF':<8}", flush=True)
print("-" * 60, flush=True)

for rank, entry in enumerate(all_scores, 1):
    print(f"{rank:<5} {entry['voice']:<12} {entry['config']:<12} "
          f"{entry['avg_utmos']:.3f}±{entry['std_utmos']:.3f}     "
          f"{entry['avg_rtf']:.2f}", flush=True)

# Best overall
if all_scores:
    winner = all_scores[0]
    print(f"\n{'='*80}", flush=True)
    print(f"  WINNER: {winner['voice']} + {winner['config']}", flush=True)
    print(f"  UTMOS: {winner['avg_utmos']:.3f} (range: {winner['min_utmos']:.3f}-{winner['max_utmos']:.3f})", flush=True)
    print(f"  RTF: {winner['avg_rtf']:.2f}", flush=True)
    print(f"{'='*80}", flush=True)

    # Best female voices
    print(f"\n  Best female voices:", flush=True)
    for s in all_scores:
        if s["voice"] in ["tara", "leah", "jess", "mia", "zoe"]:
            print(f"    {s['voice']:<8} {s['config']:<12} UTMOS={s['avg_utmos']:.3f}", flush=True)

# Save results
with open(OUTPUT_DIR / "voice_benchmark_results.json", "w") as f:
    json.dump(all_scores, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/voice_benchmark_results.json", flush=True)
