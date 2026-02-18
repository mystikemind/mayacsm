#!/usr/bin/env python3
"""
Comprehensive Orpheus TTS Benchmark - Find Optimal Voice & Parameters

Tests all 8 voices, multiple parameter configs, measures UTMOS quality.
Goal: Find the absolute best configuration BEFORE fine-tuning.

Produces audio samples + detailed quality metrics for each configuration.
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
import wave
import requests
from pathlib import Path
from snac import SNAC

# ============================================================================
# Configuration
# ============================================================================
SERVER_URL = "http://127.0.0.1:5006"
OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/audio_benchmark")
OUTPUT_DIR.mkdir(exist_ok=True)

# All 8 Orpheus voices
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

# Test utterances - conversational Maya-style responses
TEST_UTTERANCES = [
    # Warm/friendly
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "aww that sounds really tough, im sorry youre dealing with that",
    # With emotion tags
    "oh my gosh <laugh> thats hilarious, i cant believe that happened",
    "hmm <sigh> yeah that makes sense, ive been thinking about that too",
    # Longer conversational
    "honestly i think thats a great idea, you should definitely go for it",
    "yeah i know what you mean, sometimes things just dont work out the way you expect",
    # Short/reactive
    "oh really",
    "yeah for sure",
    "hmm thats interesting",
]

# Parameter configurations to test
PARAM_CONFIGS = {
    "default": {"temperature": 0.6, "top_p": 0.95, "repeat_penalty": 1.1},
    "warmer": {"temperature": 0.7, "top_p": 0.95, "repeat_penalty": 1.1},
    "creative": {"temperature": 0.8, "top_p": 0.95, "repeat_penalty": 1.15},
    "precise": {"temperature": 0.5, "top_p": 0.90, "repeat_penalty": 1.1},
    "expressive": {"temperature": 0.75, "top_p": 0.98, "repeat_penalty": 1.05},
}

# ============================================================================
# SNAC + Audio Token Extraction
# ============================================================================
CUSTOM_TOKEN_OFFSET = 128256
AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * 4096) - 1

print("Loading SNAC codec...", flush=True)
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda:2" if torch.cuda.is_available() else "cpu"
snac_model = snac_model.to(snac_device)
print(f"SNAC loaded on {snac_device}", flush=True)


def extract_audio_tokens(text_output):
    """Extract audio token IDs from llama.cpp text output."""
    token_ids = []
    for match in re.finditer(r'<custom_token_(\d+)>', text_output):
        custom_num = int(match.group(1))
        token_id = CUSTOM_TOKEN_OFFSET + custom_num
        if AUDIO_TOKEN_BASE <= token_id <= AUDIO_TOKEN_MAX:
            token_ids.append(token_id)
    return token_ids


def decode_snac_frames(token_ids):
    """Decode audio token IDs to waveform via SNAC."""
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
    """Generate speech via llama-server API."""
    # Build Orpheus prompt WITH voice name
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


def compute_audio_metrics(audio):
    """Compute basic audio quality metrics."""
    if audio is None or audio.numel() == 0:
        return {}

    audio_np = audio.numpy()
    rms = np.sqrt(np.mean(audio_np ** 2))
    peak = np.max(np.abs(audio_np))
    duration = len(audio_np) / 24000

    # Check for clipping
    clipped = np.sum(np.abs(audio_np) > 0.99) / len(audio_np)

    # Check for silence ratio
    silence_threshold = 0.01
    silence_ratio = np.sum(np.abs(audio_np) < silence_threshold) / len(audio_np)

    return {
        "rms": float(rms),
        "peak": float(peak),
        "duration": float(duration),
        "clipped_ratio": float(clipped),
        "silence_ratio": float(silence_ratio),
    }


def save_audio(audio, filepath):
    """Save audio to WAV file."""
    if audio is None:
        return
    torchaudio.save(str(filepath), audio.unsqueeze(0), 24000)


# ============================================================================
# Load UTMOS model for quality scoring
# ============================================================================
print("Loading UTMOS model for quality scoring...", flush=True)
try:
    utmos_model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
        trust_repo=True
    )
    utmos_model = utmos_model.to(snac_device).eval()
    HAS_UTMOS = True
    print("UTMOS model loaded", flush=True)
except Exception as e:
    print(f"WARNING: UTMOS not available: {e}", flush=True)
    HAS_UTMOS = False


def compute_utmos(audio):
    """Compute UTMOS score for audio quality."""
    if not HAS_UTMOS or audio is None:
        return None
    try:
        # UTMOS expects 16kHz
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        resampled = torchaudio.functional.resample(audio, 24000, 16000)
        resampled = resampled.to(snac_device)
        with torch.inference_mode():
            score = utmos_model(resampled, 16000)
        return float(score.item())
    except Exception as e:
        print(f"UTMOS error: {e}")
        return None


# ============================================================================
# Main Benchmark
# ============================================================================
print(f"\n{'='*80}", flush=True)
print(f"  COMPREHENSIVE ORPHEUS TTS BENCHMARK", flush=True)
print(f"  Voices: {len(VOICES)} | Params: {len(PARAM_CONFIGS)} | Utterances: {len(TEST_UTTERANCES)}", flush=True)
print(f"  Total tests: {len(VOICES) * len(PARAM_CONFIGS) * len(TEST_UTTERANCES)}", flush=True)
print(f"{'='*80}", flush=True)

# Check server health
try:
    r = requests.get(f"{SERVER_URL}/health", timeout=5)
    print(f"\nllama-server health: {r.status_code}", flush=True)
except Exception as e:
    print(f"\nERROR: llama-server not reachable at {SERVER_URL}: {e}", flush=True)
    sys.exit(1)

# Warmup
print("\nWarming up (2 generations)...", flush=True)
generate_speech("hello there", "tara", PARAM_CONFIGS["default"])
generate_speech("how are you", "tara", PARAM_CONFIGS["default"])

# Also test with NO voice name (current behavior)
VOICES_WITH_NONE = [None] + VOICES

results = {}
all_scores = []

for voice in VOICES_WITH_NONE:
    voice_label = voice or "NO_VOICE"
    results[voice_label] = {}

    for config_name, params in PARAM_CONFIGS.items():
        results[voice_label][config_name] = []

        print(f"\n--- Voice: {voice_label} | Config: {config_name} ---", flush=True)

        scores = []
        rtfs = []
        gen_times = []

        for i, text in enumerate(TEST_UTTERANCES):
            audio, gen_time, rtf = generate_speech(text, voice, params)

            if audio is not None and audio.numel() > 0:
                metrics = compute_audio_metrics(audio)
                utmos_score = compute_utmos(audio)

                # Save audio sample
                voice_dir = OUTPUT_DIR / voice_label / config_name
                voice_dir.mkdir(parents=True, exist_ok=True)
                save_audio(audio, voice_dir / f"sample_{i:02d}.wav")

                result = {
                    "text": text[:50],
                    "utmos": utmos_score,
                    "rtf": rtf,
                    "gen_time_ms": gen_time * 1000,
                    **metrics,
                }
                results[voice_label][config_name].append(result)

                if utmos_score:
                    scores.append(utmos_score)
                rtfs.append(rtf)
                gen_times.append(gen_time * 1000)

                print(f"  [{i+1}/{len(TEST_UTTERANCES)}] "
                      f"UTMOS={utmos_score:.3f} RTF={rtf:.2f} "
                      f"Gen={gen_time*1000:.0f}ms Dur={metrics['duration']:.1f}s "
                      f"'{text[:30]}...'", flush=True)
            else:
                print(f"  [{i+1}/{len(TEST_UTTERANCES)}] FAILED: '{text[:30]}...'", flush=True)
                results[voice_label][config_name].append({"text": text[:50], "failed": True})

        if scores:
            avg_utmos = np.mean(scores)
            avg_rtf = np.mean(rtfs)
            avg_gen = np.mean(gen_times)
            std_utmos = np.std(scores)

            all_scores.append({
                "voice": voice_label,
                "config": config_name,
                "avg_utmos": float(avg_utmos),
                "std_utmos": float(std_utmos),
                "min_utmos": float(min(scores)),
                "max_utmos": float(max(scores)),
                "avg_rtf": float(avg_rtf),
                "avg_gen_ms": float(avg_gen),
                "n_samples": len(scores),
                "n_failed": len(TEST_UTTERANCES) - len(scores),
            })

            print(f"  Summary: UTMOS={avg_utmos:.3f}+-{std_utmos:.3f} "
                  f"RTF={avg_rtf:.2f} Gen={avg_gen:.0f}ms", flush=True)

# ============================================================================
# Final Rankings
# ============================================================================
print(f"\n{'='*80}", flush=True)
print(f"  FINAL RANKINGS", flush=True)
print(f"{'='*80}", flush=True)

# Sort by average UTMOS
all_scores.sort(key=lambda x: x["avg_utmos"], reverse=True)

print(f"\n{'Rank':<5} {'Voice':<12} {'Config':<12} {'UTMOS':<12} {'RTF':<8} {'Gen(ms)':<10} {'Failed':<8}", flush=True)
print("-" * 70, flush=True)

for rank, entry in enumerate(all_scores, 1):
    print(f"{rank:<5} {entry['voice']:<12} {entry['config']:<12} "
          f"{entry['avg_utmos']:.3f}+-{entry['std_utmos']:.3f}  "
          f"{entry['avg_rtf']:.2f}    {entry['avg_gen_ms']:.0f}      "
          f"{entry['n_failed']}", flush=True)
    if rank == 10:
        print("... (showing top 10)", flush=True)
        break

# Best voice for each config
print(f"\n--- Best Voice per Config ---", flush=True)
for config_name in PARAM_CONFIGS:
    config_scores = [s for s in all_scores if s["config"] == config_name]
    if config_scores:
        best = config_scores[0]
        print(f"  {config_name:<12}: {best['voice']:<12} UTMOS={best['avg_utmos']:.3f}", flush=True)

# Best config for each voice
print(f"\n--- Best Config per Voice ---", flush=True)
for voice in VOICES_WITH_NONE:
    voice_label = voice or "NO_VOICE"
    voice_scores = [s for s in all_scores if s["voice"] == voice_label]
    if voice_scores:
        best = voice_scores[0]
        print(f"  {voice_label:<12}: {best['config']:<12} UTMOS={best['avg_utmos']:.3f}", flush=True)

# Overall winner
if all_scores:
    winner = all_scores[0]
    print(f"\n{'='*80}", flush=True)
    print(f"  WINNER: {winner['voice']} + {winner['config']}", flush=True)
    print(f"  UTMOS: {winner['avg_utmos']:.3f} (range: {winner['min_utmos']:.3f}-{winner['max_utmos']:.3f})", flush=True)
    print(f"  RTF: {winner['avg_rtf']:.2f} | Gen: {winner['avg_gen_ms']:.0f}ms", flush=True)
    print(f"{'='*80}", flush=True)

# Save full results
results_file = OUTPUT_DIR / "benchmark_results.json"
with open(results_file, "w") as f:
    json.dump({
        "rankings": all_scores,
        "details": {k: {ck: cv for ck, cv in v.items()} for k, v in results.items()},
        "config": {
            "voices": VOICES_WITH_NONE,
            "param_configs": PARAM_CONFIGS,
            "test_utterances": TEST_UTTERANCES,
        },
    }, f, indent=2, default=str)

print(f"\nResults saved to {results_file}", flush=True)
print(f"Audio samples saved to {OUTPUT_DIR}/", flush=True)
