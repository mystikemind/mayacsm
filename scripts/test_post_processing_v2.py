#!/usr/bin/env python3
"""Quick test: raw vs post-processed UTMOS, without logit_bias."""

import sys, os, time, re, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maya.engine.audio_post_processor import post_process

PROMPTS = [
    "yeah im doing pretty good, hows everything with you",
    "oh thats so cool, tell me more about that",
    "hmm yeah that makes sense, ive been thinking about that too",
    "oh really",
    "yeah for sure",
    "hmm thats interesting",
    "wow thats amazing",
    "okay cool, thanks for letting me know",
    "honestly i think thats a great idea, you should definitely go for it",
    "yeah i know what you mean, sometimes things just dont work out",
]

# New params WITHOUT logit_bias
PARAMS = {
    "temperature": 0.6, "top_p": 0.9, "top_k": 50, "min_p": 0.05,
    "repeat_penalty": 1.1, "repeat_last_n": 64,
    "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 3,
    "dry_penalty_last_n": 128,
}

# Old params for comparison
OLD_PARAMS = {"temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1}

AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MIN = AUDIO_TOKEN_BASE
AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * 4096) - 1
CUSTOM_TOKEN_OFFSET = 128256

def generate_audio(text, voice, params):
    import requests
    prompt = f"<|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"
    payload = {"prompt": prompt, "max_tokens": 500, "stop": ["<custom_token_2>"], "stream": False, **params}
    resp = requests.post("http://127.0.0.1:5006/v1/completions", json=payload, timeout=60)
    resp.raise_for_status()
    text_output = resp.json()["choices"][0]["text"]
    token_ids = []
    for match in re.finditer(r'<custom_token_(\d+)>', text_output):
        custom_num = int(match.group(1))
        tid = CUSTOM_TOKEN_OFFSET + custom_num
        if AUDIO_TOKEN_MIN <= tid <= AUDIO_TOKEN_MAX:
            token_ids.append(tid)
    return token_ids

def decode_snac(token_ids, snac, device):
    n = (len(token_ids) // 7) * 7
    if n < 7: return None
    token_ids = token_ids[:n]
    codes = [t - AUDIO_TOKEN_BASE for t in token_ids]
    l0, l1, l2 = [], [], []
    for i in range(n // 7):
        b = 7 * i
        l0.append(max(0, min(4095, codes[b])))
        l1.append(max(0, min(4095, codes[b+1] - 4096)))
        l2.append(max(0, min(4095, codes[b+2] - 2*4096)))
        l2.append(max(0, min(4095, codes[b+3] - 3*4096)))
        l1.append(max(0, min(4095, codes[b+4] - 4*4096)))
        l2.append(max(0, min(4095, codes[b+5] - 5*4096)))
        l2.append(max(0, min(4095, codes[b+6] - 6*4096)))
    snac_codes = [
        torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(device),
        torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(device),
    ]
    with torch.inference_mode():
        audio = snac.decode(snac_codes)
    return audio.squeeze().cpu().numpy()

def score_utmos(audio_np, utmos_model, device):
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
    print("  COMPREHENSIVE A/B TEST: Raw vs Post-Processed, Old vs New Params")
    print("  Voice: jess | 10 prompts | No logit_bias")
    print("=" * 70)

    new_raw, new_proc, old_raw = [], [], []

    for i, text in enumerate(PROMPTS):
        # Generate with new params
        tokens = generate_audio(text, "jess", PARAMS)
        if len(tokens) < 7:
            print(f"  [{i+1}] SKIP: no tokens"); continue
        audio = decode_snac(tokens, snac, device)
        if audio is None:
            print(f"  [{i+1}] SKIP: decode fail"); continue

        dur = len(audio) / 24000
        raw_score = score_utmos(audio, utmos, device)
        proc = post_process(audio, sample_rate=24000)
        proc_score = score_utmos(proc, utmos, device)

        # Generate with old params
        tokens_old = generate_audio(text, "jess", OLD_PARAMS)
        if len(tokens_old) >= 7:
            audio_old = decode_snac(tokens_old, snac, device)
            old_score = score_utmos(audio_old, utmos, device) if audio_old is not None else 0
            old_dur = len(audio_old) / 24000 if audio_old is not None else 0
        else:
            old_score, old_dur = 0, 0

        diff_proc = proc_score - raw_score
        diff_params = raw_score - old_score

        print(f"  [{i+1:2d}] {text[:45]:45s} | New={raw_score:.3f} Post={proc_score:.3f}({diff_proc:+.3f}) "
              f"Old={old_score:.3f} | {dur:.1f}s/{old_dur:.1f}s")

        new_raw.append(raw_score)
        new_proc.append(proc_score)
        if old_score > 0:
            old_raw.append(old_score)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  New params (raw):        {np.mean(new_raw):.3f} ± {np.std(new_raw):.3f}")
    print(f"  New params (processed):  {np.mean(new_proc):.3f} ± {np.std(new_proc):.3f}")
    print(f"  Old params (raw):        {np.mean(old_raw):.3f} ± {np.std(old_raw):.3f}")
    print(f"  New vs Old params:       {np.mean(new_raw) - np.mean(old_raw):+.3f}")
    print(f"  Post-process effect:     {np.mean(new_proc) - np.mean(new_raw):+.3f}")

if __name__ == "__main__":
    main()
