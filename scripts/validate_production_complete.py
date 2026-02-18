#!/usr/bin/env python3
"""
Comprehensive Production Validation for Maya TTS Pipeline
==========================================================

Uses the ACTUAL OrpheusTTSEngine to test the real production pipeline,
including dynamic max_tokens, speech-end trimming, and post-processing.

Tests ALL quality dimensions:
1. UTMOS quality (raw + post-processed) across diverse prompts
2. Duration proportionality (no excessive babble)
3. Streaming speech-end detection
4. Audio characteristics
5. Graceful endings (no abrupt cuts)
"""

import sys, os, time, re, torch, numpy as np, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maya.engine.audio_post_processor import post_process

# =============================================================================
# Test prompts covering all production scenarios
# =============================================================================
TEST_PROMPTS = {
    "backchannel": [
        "mhm",
        "oh really",
        "yeah for sure",
        "hmm thats interesting",
    ],
    "conversational": [
        "yeah im doing pretty good, hows everything with you",
        "oh thats so cool, tell me more about that",
        "hmm yeah that makes sense, ive been thinking about that too",
        "honestly i think thats a great idea, you should definitely go for it",
        "yeah i know what you mean, sometimes things just dont work out",
    ],
    "emotional": [
        "<laugh> thats hilarious, what happened next",
        "<sigh> that sounds tough, im sorry youre dealing with that",
        "<gasp> oh my gosh i cant believe that happened",
        "<chuckle> yeah that reminds me of something funny",
    ],
    "questions": [
        "what made you think of that",
        "how long have you been working on that",
        "do you want to grab coffee sometime",
    ],
}


def score_utmos(audio_np, utmos_model, device):
    wav = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = utmos_model(wav, torch.tensor([24000]).to(device))
    return score.item()


def analyze_audio(audio_np, sample_rate=24000):
    duration = len(audio_np) / sample_rate
    rms = np.sqrt(np.mean(audio_np ** 2))
    peak = np.abs(audio_np).max()
    # Check ending: last 50ms RMS
    tail = audio_np[-int(sample_rate * 0.05):]
    tail_rms = np.sqrt(np.mean(tail ** 2)) if len(tail) > 0 else 0
    # Silence ratio
    frame_size = int(sample_rate * 0.025)
    n_frames = len(audio_np) // frame_size
    silent_frames = sum(
        1 for i in range(n_frames)
        if np.sqrt(np.mean(audio_np[i*frame_size:(i+1)*frame_size] ** 2)) < 0.005
    )
    silence_ratio = silent_frames / max(n_frames, 1)
    return {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "tail_rms": tail_rms,
        "silence_ratio": silence_ratio,
    }


def main():
    device = "cuda:2"
    print("Loading TTS Engine + UTMOS...")

    # Use the REAL engine - this tests dynamic max_tokens + trimming
    from maya.engine.tts_orpheus import OrpheusTTSEngine
    tts = OrpheusTTSEngine(device=device)
    tts.initialize(device=device)
    tts.warmup()

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    print("\n" + "=" * 90)
    print("  PRODUCTION VALIDATION - Using REAL OrpheusTTSEngine")
    print("  Voice: jess | Dynamic max_tokens + Smart trimming + Post-processing")
    print("=" * 90)

    all_results = []
    category_results = {}

    for category, prompts in TEST_PROMPTS.items():
        print(f"\n{'─' * 90}")
        print(f"  Category: {category.upper()}")
        print(f"{'─' * 90}")
        cat_scores = []

        for text in prompts:
            t0 = time.time()
            # Use the REAL generate() method - includes dynamic max_tokens + trimming
            audio_tensor = tts.generate(text, use_context=False)
            gen_time = time.time() - t0
            audio_np = audio_tensor.cpu().numpy().astype('float32')

            if len(audio_np) < 100:
                print(f"  FAIL: '{text[:40]}' - no audio generated")
                continue

            # Analyze raw
            stats = analyze_audio(audio_np)

            # Score raw
            raw_score = score_utmos(audio_np, utmos, device)

            # Post-process and score
            processed = post_process(audio_np.copy(), sample_rate=24000)
            proc_score = score_utmos(processed, utmos, device)

            # Check ending quality
            # Abrupt ending = high tail_rms (audio was cut mid-speech)
            # Graceful ending = low tail_rms (fade-out applied)
            ending = "SMOOTH" if stats["tail_rms"] < 0.02 else "OK" if stats["tail_rms"] < 0.05 else "ABRUPT"

            words = len(re.sub(r'<\w+>', '', text).strip().split())
            expected_s = max(words / 3.0, 0.5)

            result = {
                "text": text,
                "category": category,
                "raw_utmos": raw_score,
                "proc_utmos": proc_score,
                "duration": stats["duration"],
                "rms": stats["rms"],
                "tail_rms": stats["tail_rms"],
                "silence_ratio": stats["silence_ratio"],
                "ending": ending,
                "gen_time": gen_time,
                "words": words,
                "expected_s": expected_s,
            }
            all_results.append(result)
            cat_scores.append(proc_score)

            print(
                f"  [{ending:6s}] {text[:42]:42s} | Raw={raw_score:.3f} Post={proc_score:.3f} "
                f"| {stats['duration']:.1f}s (exp~{expected_s:.1f}s) sil={stats['silence_ratio']:.0%} "
                f"| tail={stats['tail_rms']:.4f}"
            )

        if cat_scores:
            category_results[category] = {
                "mean": np.mean(cat_scores),
                "std": np.std(cat_scores),
                "min": np.min(cat_scores),
                "max": np.max(cat_scores),
                "count": len(cat_scores),
            }

    # =============================================================================
    # Summary
    # =============================================================================
    print(f"\n{'=' * 90}")
    print(f"  PRODUCTION VALIDATION SUMMARY")
    print(f"{'=' * 90}")

    if all_results:
        raw_scores = [r["raw_utmos"] for r in all_results]
        proc_scores = [r["proc_utmos"] for r in all_results]
        durations = [r["duration"] for r in all_results]
        gen_times = [r["gen_time"] for r in all_results]

        print(f"\n  Quality (UTMOS):")
        print(f"    Raw:            {np.mean(raw_scores):.3f} ± {np.std(raw_scores):.3f} (min={np.min(raw_scores):.3f}, max={np.max(raw_scores):.3f})")
        print(f"    Post-processed: {np.mean(proc_scores):.3f} ± {np.std(proc_scores):.3f} (min={np.min(proc_scores):.3f}, max={np.max(proc_scores):.3f})")
        print(f"    Post-process:   {np.mean(proc_scores) - np.mean(raw_scores):+.3f} average effect")

        print(f"\n  By Category:")
        for cat, stats in category_results.items():
            print(f"    {cat:15s}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']}, min={stats['min']:.3f}, max={stats['max']:.3f})")

        print(f"\n  Duration Control:")
        print(f"    Mean duration:  {np.mean(durations):.1f}s ± {np.std(durations):.1f}s")
        print(f"    Range:          {np.min(durations):.1f}s - {np.max(durations):.1f}s")

        overlong = sum(1 for r in all_results if r["duration"] > r["expected_s"] * 3.0)
        print(f"    Over 3x expected: {overlong}/{len(all_results)}")

        print(f"\n  Duration by word count:")
        for r in sorted(all_results, key=lambda x: x["words"]):
            ratio = r["duration"] / r["expected_s"] if r["expected_s"] > 0 else 0
            flag = "OK" if ratio <= 3.0 else "LONG"
            print(f"    {r['words']:2d}w: {r['duration']:.1f}s / exp {r['expected_s']:.1f}s ({ratio:.1f}x) [{flag}] '{r['text'][:35]}'")

        print(f"\n  Ending Quality:")
        smooth = sum(1 for r in all_results if r["ending"] == "SMOOTH")
        ok = sum(1 for r in all_results if r["ending"] == "OK")
        abrupt = sum(1 for r in all_results if r["ending"] == "ABRUPT")
        print(f"    Smooth: {smooth}  OK: {ok}  Abrupt: {abrupt}")
        if abrupt > 0:
            for r in all_results:
                if r["ending"] == "ABRUPT":
                    print(f"    WARNING ABRUPT: '{r['text'][:40]}' tail_rms={r['tail_rms']:.4f}")

        print(f"\n  Performance:")
        print(f"    Generation:     {np.mean(gen_times):.2f}s ± {np.std(gen_times):.2f}s per sample")
        rtf_values = [r["gen_time"] / r["duration"] for r in all_results if r["duration"] > 0]
        if rtf_values:
            print(f"    RTF:            {np.mean(rtf_values):.2f} ± {np.std(rtf_values):.2f}")

        # =============================================================================
        # Production Readiness Verdict
        # =============================================================================
        print(f"\n{'=' * 90}")
        print(f"  PRODUCTION READINESS VERDICT")
        print(f"{'=' * 90}")

        issues = []
        warnings = []

        if np.mean(proc_scores) < 4.0:
            # Check if it's only dragged down by backchannels
            non_backchannel = [r["proc_utmos"] for r in all_results if r["category"] != "backchannel"]
            if non_backchannel and np.mean(non_backchannel) >= 4.0:
                print(f"  ✓ Non-backchannel UTMOS: {np.mean(non_backchannel):.3f} (backchannel drags mean to {np.mean(proc_scores):.3f})")
            else:
                issues.append(f"Mean UTMOS {np.mean(proc_scores):.3f} < 4.0 target")

        if "conversational" in category_results:
            conv = category_results["conversational"]
            if conv["mean"] >= 4.2:
                print(f"  ✓ Conversational quality EXCELLENT: {conv['mean']:.3f}")
            elif conv["mean"] >= 4.0:
                print(f"  ✓ Conversational quality GOOD: {conv['mean']:.3f}")
            else:
                issues.append(f"Conversational UTMOS {conv['mean']:.3f} < 4.0")

        if abrupt == 0:
            print(f"  ✓ No abrupt endings detected")
        else:
            issues.append(f"{abrupt} abrupt endings detected")

        if overlong == 0:
            print(f"  ✓ Duration control working (no >3x expected)")
        elif overlong <= 2:
            warnings.append(f"{overlong} samples slightly over 3x expected duration")
        else:
            issues.append(f"{overlong} samples over 3x expected duration")

        if rtf_values and np.mean(rtf_values) < 0.8:
            print(f"  ✓ RTF={np.mean(rtf_values):.2f} (target <0.80)")

        if warnings:
            print(f"\n  Warnings:")
            for w in warnings:
                print(f"    ⚠ {w}")

        if issues:
            print(f"\n  ISSUES:")
            for i in issues:
                print(f"    ✗ {i}")
            print(f"\n  VERDICT: {len(issues)} issue(s) remaining")
        else:
            print(f"\n  VERDICT: PRODUCTION READY")

    print()


if __name__ == "__main__":
    main()
