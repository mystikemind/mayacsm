#!/usr/bin/env python3
"""
Voice Comparison: jess vs zoe for Maya Production
==================================================

Generates audio samples with both voices across conversational prompts,
scores with UTMOS, and saves WAV files for subjective listening.

Based on benchmark results:
  zoe + creative: UTMOS 4.283 ± 0.128 (winner)
  jess + expressive: UTMOS 4.208 ± 0.210 (current production)
"""

import sys, os, time, re, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
from maya.engine.tts_orpheus import OrpheusTTSEngine
from maya.engine.audio_post_processor import post_process
from maya.config import TTS

# Test prompts: production-representative conversational speech
TEST_PROMPTS = [
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "<chuckle> thats hilarious, what happened next",
    "hmm yeah that makes sense, ive been thinking about that too",
    "<sigh> that sounds tough, im sorry youre dealing with that",
    "honestly i think thats a great idea, you should definitely go for it",
    "what made you think of that",
    "oh wow i didnt know that, tell me more about it",
    "<laugh> no way, thats amazing",
    "yeah i know what you mean, sometimes things just dont work out",
    "hey there, im maya, its nice to meet you",
    "do you want to grab coffee sometime",
]

# Voice configs to compare
VOICE_CONFIGS = {
    "jess_production": {
        "voice": "jess",
        "temperature": TTS.temperature,  # 0.6
        "top_p": TTS.top_p,              # 0.9
    },
    "zoe_production": {
        "voice": "zoe",
        "temperature": TTS.temperature,
        "top_p": TTS.top_p,
    },
    "zoe_creative": {
        "voice": "zoe",
        "temperature": 0.7,
        "top_p": 0.95,
    },
}


def score_utmos(audio_np, utmos_model, device):
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device))
    return score.item()


def main():
    device = "cuda:2"
    output_dir = "/home/ec2-user/SageMaker/project_maya/audio_voice_comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading TTS Engine + UTMOS...")
    tts = OrpheusTTSEngine(device=device)
    tts.initialize(device=device)
    tts.warmup()

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    print(f"\n{'=' * 100}")
    print(f"  VOICE COMPARISON: jess (current) vs zoe (candidate)")
    print(f"  {len(TEST_PROMPTS)} prompts × {len(VOICE_CONFIGS)} configs = {len(TEST_PROMPTS) * len(VOICE_CONFIGS)} samples")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 100}\n")

    all_results = {}

    for config_name, config in VOICE_CONFIGS.items():
        print(f"\n{'─' * 100}")
        print(f"  Config: {config_name} (voice={config['voice']}, temp={config['temperature']}, top_p={config['top_p']})")
        print(f"{'─' * 100}")

        scores_raw = []
        scores_proc = []
        durations = []

        for i, text in enumerate(TEST_PROMPTS):
            # Override voice in the prompt by directly building it
            voice = config["voice"]
            processed_text = tts.preprocess_text(text)
            prompt = f"<|begin_of_text|><custom_token_3>{voice}: {processed_text}<custom_token_4><custom_token_5>"

            # Generate with custom sampling params
            max_tokens = tts._estimate_max_tokens(processed_text)
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "top_k": TTS.top_k,
                "min_p": TTS.min_p,
                "repeat_penalty": TTS.repeat_penalty,
                "repeat_last_n": TTS.repeat_last_n,
                "stop": ["<custom_token_2>"],
                "stream": False,
            }

            t0 = time.time()
            resp = tts._session.post(
                f"{tts._server_url}/v1/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text_output = data["choices"][0]["text"]
            finish_reason = data["choices"][0].get("finish_reason", "length")
            token_ids = tts._extract_audio_tokens(text_output)
            audio = tts._decode_snac_frames(token_ids)
            gen_time = time.time() - t0

            if audio is None or audio.numel() < 100:
                print(f"  [{i+1:2d}] FAIL: '{text[:40]}' - no audio")
                continue

            audio = audio.cpu()
            # Apply same trimming as production
            from maya.engine.tts_orpheus import _trim_trailing_audio, _apply_natural_release
            audio = _trim_trailing_audio(audio, text=processed_text)
            if finish_reason != "stop":
                tail_samples = min(int(24000 * 0.05), audio.numel())
                tail_rms = torch.sqrt(torch.mean(audio[-tail_samples:] ** 2)).item()
                if tail_rms > 0.02:
                    audio = _apply_natural_release(audio, 24000)

            audio_np = audio.numpy().astype('float32')
            duration = len(audio_np) / 24000

            # Score raw
            raw_score = score_utmos(audio_np, utmos, device)

            # Post-process and score
            processed = post_process(audio_np.copy(), sample_rate=24000)
            proc_score = score_utmos(processed, utmos, device)

            scores_raw.append(raw_score)
            scores_proc.append(proc_score)
            durations.append(duration)

            # Save WAV files for listening
            safe_text = re.sub(r'[^a-z0-9 ]', '', text.lower())[:30].replace(' ', '_')
            wav_path = os.path.join(output_dir, f"{config_name}_{i+1:02d}_{safe_text}.wav")
            sf.write(wav_path, processed, 24000)

            print(
                f"  [{i+1:2d}] Raw={raw_score:.3f} Post={proc_score:.3f} "
                f"| {duration:.1f}s | {gen_time:.2f}s | '{text[:50]}'"
            )

        if scores_proc:
            all_results[config_name] = {
                "raw_mean": np.mean(scores_raw),
                "raw_std": np.std(scores_raw),
                "proc_mean": np.mean(scores_proc),
                "proc_std": np.std(scores_proc),
                "proc_min": np.min(scores_proc),
                "proc_max": np.max(scores_proc),
                "dur_mean": np.mean(durations),
                "n": len(scores_proc),
                "scores_proc": scores_proc,
            }

    # =============================================================================
    # Summary comparison
    # =============================================================================
    print(f"\n{'=' * 100}")
    print(f"  VOICE COMPARISON SUMMARY")
    print(f"{'=' * 100}\n")

    print(f"  {'Config':<20s} | {'Raw UTMOS':>12s} | {'Post UTMOS':>12s} | {'Min':>6s} | {'Max':>6s} | {'Dur':>5s} | {'N':>3s}")
    print(f"  {'─' * 20}-+-{'─' * 12}-+-{'─' * 12}-+-{'─' * 6}-+-{'─' * 6}-+-{'─' * 5}-+-{'─' * 3}")

    for config_name, stats in all_results.items():
        print(
            f"  {config_name:<20s} | "
            f"{stats['raw_mean']:.3f}±{stats['raw_std']:.3f} | "
            f"{stats['proc_mean']:.3f}±{stats['proc_std']:.3f} | "
            f"{stats['proc_min']:.3f} | "
            f"{stats['proc_max']:.3f} | "
            f"{stats['dur_mean']:.1f}s | "
            f"{stats['n']:3d}"
        )

    # Head-to-head comparison
    if "jess_production" in all_results and "zoe_production" in all_results:
        jess = all_results["jess_production"]
        zoe = all_results["zoe_production"]
        diff = zoe["proc_mean"] - jess["proc_mean"]
        print(f"\n  Head-to-head (same sampling params):")
        print(f"    jess: {jess['proc_mean']:.3f} ± {jess['proc_std']:.3f}")
        print(f"    zoe:  {zoe['proc_mean']:.3f} ± {zoe['proc_std']:.3f}")
        print(f"    Difference: {diff:+.3f} ({'zoe wins' if diff > 0 else 'jess wins'})")

    if "zoe_creative" in all_results:
        zoe_c = all_results["zoe_creative"]
        print(f"\n  zoe (creative config): {zoe_c['proc_mean']:.3f} ± {zoe_c['proc_std']:.3f}")

    # Per-prompt comparison
    if "jess_production" in all_results and "zoe_production" in all_results:
        print(f"\n  Per-prompt head-to-head (Post-processed UTMOS):")
        jess_scores = all_results["jess_production"]["scores_proc"]
        zoe_scores = all_results["zoe_production"]["scores_proc"]
        n = min(len(jess_scores), len(zoe_scores))
        jess_wins = 0
        zoe_wins = 0
        for j in range(n):
            winner = "ZOE" if zoe_scores[j] > jess_scores[j] else "JESS"
            if winner == "ZOE":
                zoe_wins += 1
            else:
                jess_wins += 1
            print(
                f"    [{j+1:2d}] jess={jess_scores[j]:.3f} zoe={zoe_scores[j]:.3f} "
                f"→ {winner} ({abs(zoe_scores[j]-jess_scores[j]):+.3f})"
            )
        print(f"\n  Score: jess wins {jess_wins}, zoe wins {zoe_wins} out of {n} prompts")

    print(f"\n  Audio files saved to: {output_dir}")
    print(f"  Listen and compare subjectively!\n")


if __name__ == "__main__":
    main()
