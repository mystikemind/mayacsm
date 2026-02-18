#!/usr/bin/env python3
"""
Post-Processing Chain A/B Comparison
======================================

Compares the OLD chain (gentle compression + EQ) vs NEW chain
(de-essing + parallel compression + EQ) using UTMOS scoring.

Also compares streaming modes:
- OLD streaming: HPF + limiter only (stateless)
- NEW streaming: Full chain with reset=False (stateful)
"""

import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maya.engine.tts_orpheus import OrpheusTTSEngine
from maya.engine.audio_post_processor import (
    studio_process, StreamingProcessor, _stateless_stream_process,
)

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
    "thats really interesting, i never thought about it that way",
]


def score_utmos(audio_np, utmos_model, device):
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device))
    return score.item()


def old_studio_process(audio_np, sample_rate=24000):
    """OLD chain for comparison: HPF → 1.5:1 compression → warmth → presence → air → limiter."""
    from pedalboard import (
        Pedalboard, Compressor, HighpassFilter,
        HighShelfFilter, LowShelfFilter, PeakFilter,
        Limiter, Gain,
    )
    chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-22, ratio=1.5, attack_ms=15.0, release_ms=150.0),
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=1.0, q=0.7),
        PeakFilter(cutoff_frequency_hz=3500, gain_db=1.5, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.0, q=0.7),
        Limiter(threshold_db=-1.0, release_ms=50.0),
        Gain(gain_db=-0.5),
    ])
    audio_2d = audio_np[np.newaxis, :]
    return chain(audio_2d, sample_rate).squeeze().astype(np.float32)


def main():
    device = "cuda:2"
    print("Loading TTS + UTMOS...")
    tts = OrpheusTTSEngine(device=device)
    tts.initialize(device=device)
    tts.warmup()
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    print(f"\n{'=' * 100}")
    print(f"  POST-PROCESSING A/B COMPARISON")
    print(f"  OLD: HPF → 1.5:1 comp → warmth → presence → air → limiter")
    print(f"  NEW: HPF → de-ess → parallel comp (Mix) → warmth → presence → air → limiter")
    print(f"  {len(TEST_PROMPTS)} prompts")
    print(f"{'=' * 100}\n")

    raw_scores = []
    old_scores = []
    new_scores = []
    old_stream_scores = []
    new_stream_scores = []

    for i, text in enumerate(TEST_PROMPTS):
        audio = tts.generate(text, use_context=False)
        audio_np = audio.cpu().numpy().astype('float32')

        # Raw (no processing)
        raw = score_utmos(audio_np, utmos, device)
        raw_scores.append(raw)

        # OLD studio chain
        old_proc = old_studio_process(audio_np.copy())
        old = score_utmos(old_proc, utmos, device)
        old_scores.append(old)

        # NEW studio chain (with de-essing + parallel compression)
        new_proc = studio_process(audio_np.copy())
        new = score_utmos(new_proc, utmos, device)
        new_scores.append(new)

        # OLD streaming (stateless HPF + limiter)
        # Simulate chunked streaming
        chunk_size = int(24000 * 0.15)  # 150ms chunks
        old_stream_chunks = []
        for j in range(0, len(audio_np), chunk_size):
            chunk = audio_np[j:j+chunk_size].copy()
            if len(chunk) > 100:
                old_stream_chunks.append(_stateless_stream_process(chunk))
        if old_stream_chunks:
            old_stream_full = np.concatenate(old_stream_chunks)
            old_s = score_utmos(old_stream_full, utmos, device)
        else:
            old_s = 0
        old_stream_scores.append(old_s)

        # NEW streaming (stateful full chain)
        proc = StreamingProcessor()
        new_stream_chunks = []
        for j in range(0, len(audio_np), chunk_size):
            chunk = audio_np[j:j+chunk_size].copy()
            if len(chunk) > 100:
                new_stream_chunks.append(proc.process_chunk(chunk))
        if new_stream_chunks:
            new_stream_full = np.concatenate(new_stream_chunks)
            new_s = score_utmos(new_stream_full, utmos, device)
        else:
            new_s = 0
        new_stream_scores.append(new_s)

        winner_studio = "NEW" if new > old else "OLD"
        winner_stream = "NEW" if new_s > old_s else "OLD"
        print(
            f"  [{i+1:2d}] Raw={raw:.3f} | OLD={old:.3f} NEW={new:.3f} ({winner_studio}) | "
            f"OldStream={old_s:.3f} NewStream={new_s:.3f} ({winner_stream})"
            f"\n       '{text[:55]}'"
        )

    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}\n")

    print(f"  STUDIO (one-shot) processing:")
    print(f"    Raw (no proc):  {np.mean(raw_scores):.3f} ± {np.std(raw_scores):.3f}")
    print(f"    OLD chain:      {np.mean(old_scores):.3f} ± {np.std(old_scores):.3f}")
    print(f"    NEW chain:      {np.mean(new_scores):.3f} ± {np.std(new_scores):.3f}")
    diff_studio = np.mean(new_scores) - np.mean(old_scores)
    print(f"    Improvement:    {diff_studio:+.3f} ({'NEW better' if diff_studio > 0 else 'OLD better'})")
    new_wins = sum(1 for o, n in zip(old_scores, new_scores) if n > o)
    print(f"    NEW wins:       {new_wins}/{len(old_scores)}")

    print(f"\n  STREAMING processing:")
    print(f"    OLD (stateless): {np.mean(old_stream_scores):.3f} ± {np.std(old_stream_scores):.3f}")
    print(f"    NEW (stateful):  {np.mean(new_stream_scores):.3f} ± {np.std(new_stream_scores):.3f}")
    diff_stream = np.mean(new_stream_scores) - np.mean(old_stream_scores)
    print(f"    Improvement:     {diff_stream:+.3f} ({'NEW better' if diff_stream > 0 else 'OLD better'})")
    new_stream_wins = sum(1 for o, n in zip(old_stream_scores, new_stream_scores) if n > o)
    print(f"    NEW wins:        {new_stream_wins}/{len(old_stream_scores)}")

    print()


if __name__ == "__main__":
    main()
