#!/usr/bin/env python3
"""
Validate that removing emotion tags improves UTMOS quality.

Compares the OLD prompts (with <laugh>, <sigh> etc.) against
the NEW natural text equivalents (ha!, aww, ugh etc.)
"""

import sys, os, time, re, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maya.engine.tts_orpheus import OrpheusTTSEngine
from maya.engine.audio_post_processor import post_process

# Paired prompts: OLD (emotion tags) vs NEW (natural text)
PAIRED_PROMPTS = [
    ("<chuckle> thats hilarious, what happened next",
     "ha thats hilarious, what happened next"),
    ("<sigh> that sounds tough, im sorry youre dealing with that",
     "aww that sounds tough, im sorry youre dealing with that"),
    ("<gasp> oh my gosh i cant believe that happened",
     "oh my gosh i cant believe that happened"),
    ("<laugh> no way, thats amazing",
     "ha no way, thats amazing"),
    ("<chuckle> yeah that reminds me of something funny",
     "ha yeah that reminds me of something funny"),
    ("<sigh> that sounds rough, im sorry youre going through that",
     "aww that sounds rough, im sorry youre going through that"),
    ("<laugh> thats hilarious, what happened next",
     "ha thats hilarious, what happened next"),
    ("<gasp> oh wow i didnt know that",
     "oh wow i didnt know that"),
]


def score_utmos(audio_np, utmos_model, device):
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device))
    return score.item()


def main():
    device = "cuda:2"
    print("Loading TTS + UTMOS...")
    tts = OrpheusTTSEngine(device=device)
    tts.initialize(device=device)
    tts.warmup()

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    print(f"\n{'=' * 100}")
    print(f"  EMOTION TAG REMOVAL VALIDATION")
    print(f"  {len(PAIRED_PROMPTS)} paired prompts × 2 versions")
    print(f"{'=' * 100}\n")

    old_scores = []
    new_scores = []

    for i, (old_text, new_text) in enumerate(PAIRED_PROMPTS):
        # Generate OLD (with emotion tags)
        audio_old = tts.generate(old_text, use_context=False)
        audio_old_np = audio_old.cpu().numpy().astype('float32')
        old_proc = post_process(audio_old_np.copy(), sample_rate=24000)
        old_score = score_utmos(old_proc, utmos, device)

        # Generate NEW (natural text, tags stripped by preprocess_text)
        audio_new = tts.generate(new_text, use_context=False)
        audio_new_np = audio_new.cpu().numpy().astype('float32')
        new_proc = post_process(audio_new_np.copy(), sample_rate=24000)
        new_score = score_utmos(new_proc, utmos, device)

        old_scores.append(old_score)
        new_scores.append(new_score)

        diff = new_score - old_score
        winner = "NEW" if diff > 0 else "OLD"
        print(
            f"  [{i+1:2d}] OLD={old_score:.3f} NEW={new_score:.3f} ({diff:+.3f} {winner})"
            f"\n       OLD: '{old_text[:55]}'"
            f"\n       NEW: '{new_text[:55]}'"
        )

    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}")
    print(f"  OLD (emotion tags): {np.mean(old_scores):.3f} ± {np.std(old_scores):.3f}")
    print(f"  NEW (natural text): {np.mean(new_scores):.3f} ± {np.std(new_scores):.3f}")
    print(f"  Improvement:        {np.mean(new_scores) - np.mean(old_scores):+.3f}")
    new_wins = sum(1 for o, n in zip(old_scores, new_scores) if n > o)
    print(f"  NEW wins: {new_wins}/{len(PAIRED_PROMPTS)}")
    print()


if __name__ == "__main__":
    main()
