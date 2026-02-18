#!/usr/bin/env python3
"""Test SAFE new params (no DRY, no logit_bias) vs old params."""

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
    "oh my gosh thats hilarious, i cant believe that happened",
    "aww that sounds really tough, im sorry youre dealing with that",
    "that reminds me of something that happened last week",
    "mhm",
    "well you know what i think, lets just go for it",
]

# Safe new params: repeat_last_n + min_p + top_k only
SAFE_PARAMS = {
    "temperature": 0.6, "top_p": 0.9, "top_k": 50, "min_p": 0.05,
    "repeat_penalty": 1.1, "repeat_last_n": 64,
}
OLD_PARAMS = {"temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1}

AUDIO_TOKEN_BASE = 128266
AUDIO_TOKEN_MIN = AUDIO_TOKEN_BASE
AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * 4096) - 1
CUSTOM_TOKEN_OFFSET = 128256

def generate_and_score(text, voice, params, snac, utmos, device):
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

    if len(token_ids) < 7: return 0, 0, 0
    n = (len(token_ids) // 7) * 7
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
    audio_np = audio.squeeze().cpu().numpy()
    dur = len(audio_np) / 24000

    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos(wav, torch.tensor([24000]).to(device))

    # Also score post-processed
    proc = post_process(audio_np, sample_rate=24000)
    wav_p = torch.from_numpy(proc).unsqueeze(0).to(device)
    with torch.inference_mode():
        score_p = utmos(wav_p, torch.tensor([24000]).to(device))

    return score.item(), score_p.item(), dur

def main():
    device = "cuda:2"
    print("Loading SNAC + UTMOS...")
    from snac import SNAC
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    print("\n" + "=" * 70)
    print("  SAFE PARAMS TEST: jess voice, 15 prompts")
    print("  Safe new: temp=0.6 top_p=0.9 top_k=50 min_p=0.05 rep=1.1 rep_last_n=64")
    print("  Old:      temp=0.6 top_p=0.9 rep=1.1")
    print("=" * 70)

    new_raw, new_proc, old_raw = [], [], []

    for i, text in enumerate(PROMPTS):
        s_new, sp_new, d_new = generate_and_score(text, "jess", SAFE_PARAMS, snac, utmos_model, device)
        s_old, _, d_old = generate_and_score(text, "jess", OLD_PARAMS, snac, utmos_model, device)

        diff = s_new - s_old
        print(f"  [{i+1:2d}] {text[:42]:42s} | New={s_new:.3f} Proc={sp_new:.3f} Old={s_old:.3f} ({diff:+.3f}) | {d_new:.1f}s/{d_old:.1f}s")

        if s_new > 0: new_raw.append(s_new)
        if sp_new > 0: new_proc.append(sp_new)
        if s_old > 0: old_raw.append(s_old)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Safe new params (raw):     {np.mean(new_raw):.3f} ± {np.std(new_raw):.3f}")
    print(f"  Safe new + post-process:   {np.mean(new_proc):.3f} ± {np.std(new_proc):.3f}")
    print(f"  Old params (raw):          {np.mean(old_raw):.3f} ± {np.std(old_raw):.3f}")
    print(f"  New vs Old:                {np.mean(new_raw) - np.mean(old_raw):+.3f}")
    print(f"  Post-process effect:       {np.mean(new_proc) - np.mean(new_raw):+.3f}")

if __name__ == "__main__":
    main()
