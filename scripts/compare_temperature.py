#!/usr/bin/env python3
"""
Temperature A/B Comparison for Orpheus TTS
=============================================

Research suggests temp=0.6 (current) may be conservative for conversational speech.
Tests 0.6 vs 0.65 vs 0.70 to find the sweet spot for prosodic variety vs stability.
"""

import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maya.engine.tts_orpheus import OrpheusTTSEngine
from maya.engine.audio_post_processor import post_process
from maya.config import TTS

TEST_PROMPTS = [
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "ha thats hilarious, what happened next",
    "hmm yeah that makes sense, ive been thinking about that too",
    "aww that sounds tough, im sorry youre dealing with that",
    "honestly i think thats a great idea, you should definitely go for it",
    "oh wow i didnt know that, tell me more about it",
    "hey there, im maya, its nice to meet you",
    "ooh thats exciting, when does it start",
    "ugh yeah that sounds frustrating, what are you gonna do about it",
]

TEMPERATURES = [0.6, 0.65, 0.70]
RUNS_PER_PROMPT = 2  # Average for stability


def score_utmos(audio_np, utmos_model, device):
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device))
    return score.item()


def generate_with_temp(tts, text, temperature):
    processed = tts.preprocess_text(text)
    prompt = f"<|begin_of_text|><custom_token_3>{TTS.voice}: {processed}<custom_token_4><custom_token_5>"
    max_tokens = tts._estimate_max_tokens(processed)

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": TTS.top_p,
        "top_k": TTS.top_k,
        "min_p": TTS.min_p,
        "repeat_penalty": TTS.repeat_penalty,
        "repeat_last_n": TTS.repeat_last_n,
        "stop": ["<custom_token_2>"],
        "stream": False,
    }

    resp = tts._session.post(f"{tts._server_url}/v1/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text_output = data["choices"][0]["text"]
    token_ids = tts._extract_audio_tokens(text_output)
    audio = tts._decode_snac_frames(token_ids)

    if audio is None or audio.numel() < 100:
        return None

    from maya.engine.tts_orpheus import _trim_trailing_audio
    audio = _trim_trailing_audio(audio.cpu(), text=processed)
    return audio.numpy().astype('float32')


def main():
    device = "cuda:2"
    print("Loading TTS + UTMOS...")
    tts = OrpheusTTSEngine(device=device)
    tts.initialize(device=device)
    tts.warmup()
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    print(f"\n{'=' * 100}")
    print(f"  TEMPERATURE COMPARISON")
    print(f"  Temperatures: {TEMPERATURES}")
    print(f"  {len(TEST_PROMPTS)} prompts × {RUNS_PER_PROMPT} runs")
    print(f"{'=' * 100}\n")

    results = {t: [] for t in TEMPERATURES}

    for i, text in enumerate(TEST_PROMPTS):
        print(f"  [{i+1:2d}/{len(TEST_PROMPTS)}] '{text[:55]}'")
        row = []
        for temp in TEMPERATURES:
            scores = []
            for _ in range(RUNS_PER_PROMPT):
                audio_np = generate_with_temp(tts, text, temp)
                if audio_np is not None:
                    proc = post_process(audio_np.copy(), sample_rate=24000)
                    scores.append(score_utmos(proc, utmos, device))
            if scores:
                avg = np.mean(scores)
                results[temp].append(avg)
                row.append(f"t={temp}:{avg:.3f}")
            else:
                row.append(f"t={temp}:FAIL")
        print(f"         {' | '.join(row)}")

    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}\n")

    for temp in TEMPERATURES:
        s = results[temp]
        if s:
            print(f"  temp={temp}: {np.mean(s):.3f} ± {np.std(s):.3f} (min={np.min(s):.3f}, max={np.max(s):.3f})")

    # Head-to-head
    if results[0.6] and results[0.65]:
        n = min(len(results[0.6]), len(results[0.65]))
        wins_65 = sum(1 for i in range(n) if results[0.65][i] > results[0.6][i])
        print(f"\n  0.65 vs 0.60: {wins_65}/{n} wins, diff={np.mean(results[0.65][:n])-np.mean(results[0.6][:n]):+.3f}")

    if results[0.6] and results[0.70]:
        n = min(len(results[0.6]), len(results[0.70]))
        wins_70 = sum(1 for i in range(n) if results[0.70][i] > results[0.6][i])
        print(f"  0.70 vs 0.60: {wins_70}/{n} wins, diff={np.mean(results[0.70][:n])-np.mean(results[0.6][:n]):+.3f}")

    best_temp = max(TEMPERATURES, key=lambda t: np.mean(results[t]) if results[t] else 0)
    print(f"\n  BEST: temp={best_temp} ({np.mean(results[best_temp]):.3f})")


if __name__ == "__main__":
    main()
