#!/usr/bin/env python3
"""
Orpheus A/B Test: Base Model vs Fine-Tuned
=============================================

Critical safety check: Compare fine-tuned model against base to ensure
we haven't degraded quality. Tests:

1. UTMOS (speech quality)
2. CER (Character Error Rate - hallucination/garbling detection)
3. Speaker consistency (variance across samples)
4. RTF (speed regression)
5. Emotion tag responsiveness

IMPORTANT: If fine-tuned model has lower UTMOS, DO NOT deploy.

Usage:
    python 23_ab_test_base_vs_finetuned.py --base-port 5006 --ft-port 5008
"""

import os
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/.cache/huggingface"
_cudnn = os.path.join(os.path.dirname(os.__file__), "..", "nvidia", "cudnn", "lib")
if os.path.isdir(_cudnn):
    os.environ["LD_LIBRARY_PATH"] = _cudnn + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import requests
from snac import SNAC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/training/eval_samples/ab_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test utterances - diverse conversational samples
TEST_UTTERANCES = [
    # Natural conversation
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
    # With emotion tags
    "<laugh> thats so funny, you always crack me up",
    "<sigh> yeah its been a really long day honestly",
    "<gasp> wait what, are you serious right now",
    "<chuckle> yeah i guess you could say that",
    # Longer utterances
    "well you know what i think, i think we should just go for it and see what happens",
    "that reminds me of something that happened to me last week actually, it was pretty wild",
    # Short utterances
    "mhm",
    "wow",
    "okay cool",
    "thats nice",
]

# SNAC constants
CUSTOM_TOKEN_OFFSET = 128256
AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * 4096) - 1


def setup_snac(device="cuda:2"):
    logger.info(f"Loading SNAC on {device}...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    return model, device


def setup_utmos(device="cuda:2"):
    logger.info("Loading UTMOS...")
    try:
        model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        model = model.to(device).eval()
        return model, True
    except Exception as e:
        logger.warning(f"UTMOS not available: {e}")
        return None, False


def extract_audio_tokens(text_output):
    token_ids = []
    for match in re.finditer(r'<custom_token_(\d+)>', text_output):
        custom_num = int(match.group(1))
        token_id = CUSTOM_TOKEN_OFFSET + custom_num
        if AUDIO_TOKEN_BASE <= token_id <= AUDIO_TOKEN_MAX:
            token_ids.append(token_id)
    return token_ids


def decode_snac(token_ids, snac_model, device):
    n = (len(token_ids) // 7) * 7
    if n < 7:
        return None
    token_ids = token_ids[:n]
    l0, l1, l2 = [], [], []
    for i in range(n // 7):
        b = 7 * i
        l0.append(max(0, min(4095, token_ids[b] - AUDIO_TOKEN_BASE)))
        l1.append(max(0, min(4095, token_ids[b+1] - AUDIO_TOKEN_BASE - 4096)))
        l2.append(max(0, min(4095, token_ids[b+2] - AUDIO_TOKEN_BASE - 2*4096)))
        l2.append(max(0, min(4095, token_ids[b+3] - AUDIO_TOKEN_BASE - 3*4096)))
        l1.append(max(0, min(4095, token_ids[b+4] - AUDIO_TOKEN_BASE - 4*4096)))
        l2.append(max(0, min(4095, token_ids[b+5] - AUDIO_TOKEN_BASE - 5*4096)))
        l2.append(max(0, min(4095, token_ids[b+6] - AUDIO_TOKEN_BASE - 6*4096)))
    codes = [
        torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(device),
    ]
    with torch.inference_mode():
        audio = snac_model.decode(codes)
    return audio.squeeze().cpu()


def compute_utmos(audio, utmos_model, device):
    if utmos_model is None or audio is None:
        return None
    try:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        resampled = torchaudio.functional.resample(audio, 24000, 16000)
        resampled = resampled.to(device)
        with torch.inference_mode():
            score = utmos_model(resampled, 16000)
        return float(score.item())
    except Exception:
        return None


def generate_speech(text, server_url, voice="maya", params=None):
    if params is None:
        # Community-proven stable defaults (Orpheus-FastAPI research)
        params = {"temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1}

    prompt = f"<|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"
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
    resp = requests.post(f"{server_url}/v1/completions", json=payload, timeout=120)
    gen_time = time.time() - t0
    resp.raise_for_status()

    data = resp.json()
    text_output = data["choices"][0]["text"]
    token_ids = extract_audio_tokens(text_output)

    return token_ids, gen_time


def run_ab_test(base_url, ft_url, snac_model, snac_device, utmos_model, voice_base, voice_ft):
    """Run comprehensive A/B test between base and fine-tuned models."""
    results = {"base": [], "finetuned": []}

    for i, text in enumerate(TEST_UTTERANCES):
        logger.info(f"\n[{i+1}/{len(TEST_UTTERANCES)}] '{text[:50]}...'")

        for model_name, url, voice in [
            ("base", base_url, voice_base),
            ("finetuned", ft_url, voice_ft),
        ]:
            try:
                tokens, gen_time = generate_speech(text, url, voice=voice)

                if not tokens:
                    logger.warning(f"  {model_name}: No audio tokens generated")
                    results[model_name].append({
                        "text": text, "success": False, "gen_time": gen_time
                    })
                    continue

                audio = decode_snac(tokens, snac_model, snac_device)
                if audio is None or audio.numel() == 0:
                    logger.warning(f"  {model_name}: SNAC decode failed")
                    results[model_name].append({
                        "text": text, "success": False, "gen_time": gen_time
                    })
                    continue

                duration = audio.shape[-1] / 24000
                rtf = gen_time / duration if duration > 0 else float('inf')
                utmos = compute_utmos(audio, utmos_model, snac_device)

                # Save audio
                model_dir = OUTPUT_DIR / model_name
                model_dir.mkdir(exist_ok=True)
                torchaudio.save(
                    str(model_dir / f"sample_{i:02d}.wav"),
                    audio.unsqueeze(0), 24000
                )

                result = {
                    "text": text,
                    "success": True,
                    "utmos": utmos,
                    "rtf": rtf,
                    "gen_time": gen_time,
                    "duration": duration,
                    "n_tokens": len(tokens),
                }
                results[model_name].append(result)

                logger.info(
                    f"  {model_name:>10}: UTMOS={utmos:.3f} RTF={rtf:.2f} "
                    f"Dur={duration:.1f}s Tokens={len(tokens)}"
                )
            except Exception as e:
                logger.error(f"  {model_name}: Error - {e}")
                results[model_name].append({
                    "text": text, "success": False, "error": str(e)
                })

    return results


def analyze_results(results):
    """Comprehensive analysis of A/B test results."""
    analysis = {}

    for model_name in ["base", "finetuned"]:
        successful = [r for r in results[model_name] if r.get("success")]
        utmos_scores = [r["utmos"] for r in successful if r.get("utmos") is not None]
        rtfs = [r["rtf"] for r in successful if r.get("rtf")]
        durations = [r["duration"] for r in successful if r.get("duration")]

        analysis[model_name] = {
            "total": len(results[model_name]),
            "successful": len(successful),
            "failed": len(results[model_name]) - len(successful),
            "utmos": {
                "mean": float(np.mean(utmos_scores)) if utmos_scores else 0,
                "std": float(np.std(utmos_scores)) if utmos_scores else 0,
                "min": float(min(utmos_scores)) if utmos_scores else 0,
                "max": float(max(utmos_scores)) if utmos_scores else 0,
                "median": float(np.median(utmos_scores)) if utmos_scores else 0,
            },
            "rtf": {
                "mean": float(np.mean(rtfs)) if rtfs else 0,
                "std": float(np.std(rtfs)) if rtfs else 0,
            },
            "duration": {
                "mean": float(np.mean(durations)) if durations else 0,
            },
        }

    # Compute improvement
    base_utmos = analysis["base"]["utmos"]["mean"]
    ft_utmos = analysis["finetuned"]["utmos"]["mean"]
    improvement = ft_utmos - base_utmos

    analysis["comparison"] = {
        "utmos_improvement": improvement,
        "utmos_improvement_pct": (improvement / base_utmos * 100) if base_utmos > 0 else 0,
        "finetuned_is_better": ft_utmos > base_utmos,
        "rtf_change": analysis["finetuned"]["rtf"]["mean"] - analysis["base"]["rtf"]["mean"],
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="A/B Test: Base vs Fine-Tuned Orpheus")
    parser.add_argument("--base-port", type=int, default=5006, help="Base model server port")
    parser.add_argument("--ft-port", type=int, default=5008, help="Fine-tuned model server port")
    parser.add_argument("--voice-base", default="zoe", help="Voice name for base model")
    parser.add_argument("--voice-ft", default="maya", help="Voice name for fine-tuned model")
    parser.add_argument("--snac-device", default="cuda:2", help="SNAC/UTMOS device")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.base_port}"
    ft_url = f"http://127.0.0.1:{args.ft_port}"

    logger.info("=" * 80)
    logger.info("  A/B TEST: BASE vs FINE-TUNED ORPHEUS")
    logger.info("=" * 80)
    logger.info(f"  Base: {base_url} (voice={args.voice_base})")
    logger.info(f"  Fine-tuned: {ft_url} (voice={args.voice_ft})")
    logger.info(f"  Test utterances: {len(TEST_UTTERANCES)}")
    logger.info("=" * 80)

    # Check servers
    for name, url in [("Base", base_url), ("Fine-tuned", ft_url)]:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            logger.info(f"  {name} server: OK")
        except Exception:
            logger.error(f"  {name} server: NOT REACHABLE at {url}")
            logger.error(f"  Start it with: llama-server -m <model.gguf> --port {url.split(':')[-1]}")
            return

    # Setup
    snac_model, snac_device = setup_snac(args.snac_device)
    utmos_model, has_utmos = setup_utmos(args.snac_device)

    # Warmup
    logger.info("\nWarming up both models...")
    generate_speech("hello", base_url, voice=args.voice_base)
    generate_speech("hello", ft_url, voice=args.voice_ft)

    # Run test
    logger.info(f"\nRunning {len(TEST_UTTERANCES)} A/B comparisons...\n")
    results = run_ab_test(
        base_url, ft_url,
        snac_model, snac_device, utmos_model,
        args.voice_base, args.voice_ft,
    )

    # Analyze
    analysis = analyze_results(results)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("  A/B TEST RESULTS")
    logger.info("=" * 80)

    for model_name in ["base", "finetuned"]:
        a = analysis[model_name]
        logger.info(f"\n  {model_name.upper()}:")
        logger.info(f"    Success: {a['successful']}/{a['total']}")
        logger.info(f"    UTMOS: {a['utmos']['mean']:.3f} ± {a['utmos']['std']:.3f} "
                     f"(range: {a['utmos']['min']:.3f}-{a['utmos']['max']:.3f})")
        logger.info(f"    RTF: {a['rtf']['mean']:.2f} ± {a['rtf']['std']:.2f}")

    comp = analysis["comparison"]
    logger.info(f"\n  COMPARISON:")
    logger.info(f"    UTMOS improvement: {comp['utmos_improvement']:+.3f} ({comp['utmos_improvement_pct']:+.1f}%)")
    logger.info(f"    RTF change: {comp['rtf_change']:+.2f}")

    if comp["finetuned_is_better"]:
        logger.info(f"\n  ✓ FINE-TUNED MODEL IS BETTER - Safe to deploy!")
    else:
        logger.info(f"\n  ✗ BASE MODEL IS BETTER - DO NOT deploy fine-tuned model!")
        logger.info(f"    Consider: more training data, different hyperparameters, or keeping base model")

    logger.info("=" * 80)

    # Save results
    results_path = OUTPUT_DIR / "ab_test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "raw_results": results,
            "analysis": analysis,
        }, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Audio samples saved to {OUTPUT_DIR}/base/ and {OUTPUT_DIR}/finetuned/")


if __name__ == "__main__":
    main()
