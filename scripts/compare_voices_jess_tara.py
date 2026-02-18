#!/usr/bin/env python3
"""
Voice Comparison: jess vs tara for Maya Production
====================================================

Research ranks tara #1 for conversational realism, jess #3.
This test compares both using production prompts with UTMOS scoring
and saves WAV files for subjective listening.

Uses 2 runs per prompt for stability (stochastic TTS).
All prompts are clean (no emotion tags).
"""

import sys, os, time, re, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
from maya.engine.tts_orpheus import OrpheusTTSEngine, _trim_trailing_audio, _apply_natural_release
from maya.engine.audio_post_processor import post_process
from maya.config import TTS

# Production-representative conversational prompts (no emotion tags)
TEST_PROMPTS = [
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "ha thats hilarious, what happened next",
    "hmm yeah that makes sense, ive been thinking about that too",
    "aww that sounds tough, im sorry youre dealing with that",
    "honestly i think thats a great idea, you should definitely go for it",
    "what made you think of that",
    "oh wow i didnt know that, tell me more about it",
    "ha no way, thats amazing",
    "yeah i know what you mean, sometimes things just dont work out",
    "hey there, im maya, its nice to meet you",
    "ooh thats exciting, when does it start",
    "ugh yeah that sounds frustrating, what are you gonna do about it",
    "do you want to grab coffee sometime",
    "thats really interesting, i never thought about it that way",
]

VOICES = ["jess", "tara"]
RUNS_PER_PROMPT = 2  # Average over 2 runs for stability


def score_utmos(audio_np, utmos_model, device):
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device))
    return score.item()


def generate_one(tts, text, voice):
    """Generate one audio sample with given voice."""
    processed_text = tts.preprocess_text(text)
    prompt = f"<|begin_of_text|><custom_token_3>{voice}: {processed_text}<custom_token_4><custom_token_5>"
    max_tokens = tts._estimate_max_tokens(processed_text)

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": TTS.temperature,
        "top_p": TTS.top_p,
        "top_k": TTS.top_k,
        "min_p": TTS.min_p,
        "repeat_penalty": TTS.repeat_penalty,
        "repeat_last_n": TTS.repeat_last_n,
        "stop": ["<custom_token_2>"],
        "stream": False,
    }

    t0 = time.time()
    resp = tts._session.post(f"{tts._server_url}/v1/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text_output = data["choices"][0]["text"]
    finish_reason = data["choices"][0].get("finish_reason", "length")
    token_ids = tts._extract_audio_tokens(text_output)
    audio = tts._decode_snac_frames(token_ids)
    gen_time = time.time() - t0

    if audio is None or audio.numel() < 100:
        return None, 0

    audio = audio.cpu()
    audio = _trim_trailing_audio(audio, text=processed_text)
    if finish_reason != "stop":
        tail_samples = min(int(24000 * 0.05), audio.numel())
        tail_rms = torch.sqrt(torch.mean(audio[-tail_samples:] ** 2)).item()
        if tail_rms > 0.02:
            audio = _apply_natural_release(audio, 24000)

    return audio.numpy().astype('float32'), gen_time


def main():
    device = "cuda:2"
    output_dir = "/home/ec2-user/SageMaker/project_maya/audio_voice_comparison_tara"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading TTS Engine + UTMOS...")
    tts = OrpheusTTSEngine(device=device)
    tts.initialize(device=device)
    tts.warmup()

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    print(f"\n{'=' * 100}")
    print(f"  VOICE COMPARISON: jess (current #3) vs tara (ranked #1 conversational)")
    print(f"  {len(TEST_PROMPTS)} prompts × {len(VOICES)} voices × {RUNS_PER_PROMPT} runs")
    print(f"  Sampling: temp={TTS.temperature}, top_p={TTS.top_p}, top_k={TTS.top_k}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 100}\n")

    results = {v: {"raw": [], "proc": [], "durations": [], "per_prompt": []} for v in VOICES}

    for i, text in enumerate(TEST_PROMPTS):
        print(f"\n  Prompt [{i+1:2d}/{len(TEST_PROMPTS)}]: '{text[:60]}'")

        for voice in VOICES:
            run_scores = []
            best_audio = None
            best_score = -1

            for run in range(RUNS_PER_PROMPT):
                audio_np, gen_time = generate_one(tts, text, voice)
                if audio_np is None:
                    continue

                processed = post_process(audio_np.copy(), sample_rate=24000)
                score = score_utmos(processed, utmos, device)
                run_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_audio = processed

            if not run_scores:
                print(f"    {voice:>4s}: FAIL (no audio)")
                continue

            avg_score = np.mean(run_scores)
            raw_score = score_utmos(audio_np, utmos, device)  # last run raw

            results[voice]["raw"].append(raw_score)
            results[voice]["proc"].append(avg_score)
            results[voice]["durations"].append(len(audio_np) / 24000)
            results[voice]["per_prompt"].append(avg_score)

            # Save best audio for listening
            safe_text = re.sub(r'[^a-z0-9 ]', '', text.lower())[:30].replace(' ', '_')
            wav_path = os.path.join(output_dir, f"{voice}_{i+1:02d}_{safe_text}.wav")
            sf.write(wav_path, best_audio, 24000)

            scores_str = ", ".join(f"{s:.3f}" for s in run_scores)
            print(f"    {voice:>4s}: {avg_score:.3f} (runs: {scores_str}) | {len(audio_np)/24000:.1f}s")

    # =============================================================================
    # Summary
    # =============================================================================
    print(f"\n{'=' * 100}")
    print(f"  VOICE COMPARISON SUMMARY")
    print(f"{'=' * 100}\n")

    for voice in VOICES:
        r = results[voice]
        if not r["proc"]:
            continue
        print(f"  {voice}:")
        print(f"    UTMOS (post): {np.mean(r['proc']):.3f} ± {np.std(r['proc']):.3f} "
              f"(min={np.min(r['proc']):.3f}, max={np.max(r['proc']):.3f})")
        print(f"    UTMOS (raw):  {np.mean(r['raw']):.3f} ± {np.std(r['raw']):.3f}")
        print(f"    Duration:     {np.mean(r['durations']):.1f}s avg")
        print(f"    Samples:      {len(r['proc'])}")

    # Head-to-head
    if results["jess"]["per_prompt"] and results["tara"]["per_prompt"]:
        n = min(len(results["jess"]["per_prompt"]), len(results["tara"]["per_prompt"]))
        jess_scores = results["jess"]["per_prompt"][:n]
        tara_scores = results["tara"]["per_prompt"][:n]

        diff = np.mean(tara_scores) - np.mean(jess_scores)
        print(f"\n  Head-to-head ({n} prompts):")
        print(f"    jess: {np.mean(jess_scores):.3f} ± {np.std(jess_scores):.3f}")
        print(f"    tara: {np.mean(tara_scores):.3f} ± {np.std(tara_scores):.3f}")
        print(f"    Difference: {diff:+.3f} ({'tara wins' if diff > 0 else 'jess wins'})")

        jess_wins = sum(1 for j, t in zip(jess_scores, tara_scores) if j > t)
        tara_wins = n - jess_wins
        print(f"    Per-prompt: jess wins {jess_wins}, tara wins {tara_wins}")

        print(f"\n  Per-prompt detail:")
        for j in range(n):
            d = tara_scores[j] - jess_scores[j]
            winner = "TARA" if d > 0 else "JESS"
            print(f"    [{j+1:2d}] jess={jess_scores[j]:.3f} tara={tara_scores[j]:.3f} → {winner} ({d:+.3f})")

    # Recommendation
    if results["jess"]["proc"] and results["tara"]["proc"]:
        jess_mean = np.mean(results["jess"]["proc"])
        tara_mean = np.mean(results["tara"]["proc"])
        diff = tara_mean - jess_mean
        print(f"\n  RECOMMENDATION:")
        if diff > 0.05:
            print(f"    Switch to TARA (+{diff:.3f} UTMOS improvement)")
        elif diff > 0:
            print(f"    MARGINAL: tara slightly better (+{diff:.3f}), listen to samples")
        elif diff > -0.05:
            print(f"    KEEP JESS: marginal difference ({diff:+.3f})")
        else:
            print(f"    KEEP JESS: clearly better ({diff:+.3f})")

    print(f"\n  Audio files saved to: {output_dir}")
    print(f"  Listen and compare subjectively!\n")


if __name__ == "__main__":
    main()
