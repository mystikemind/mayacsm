#!/usr/bin/env python3
"""
Test TTS patterns to verify which work with our CURRENT fine-tuned model.

This tests whether patterns like "hmm", "yeah", "mhm" cause gibberish
or work correctly with our production model.

The goal is to get EVIDENCE, not assumptions.
"""

import sys
import os
import time

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

def calculate_audio_quality(audio: torch.Tensor, sample_rate: int = 24000) -> dict:
    """Calculate audio quality metrics to detect gibberish."""
    if len(audio) == 0:
        return {"valid": False, "reason": "empty"}

    audio_np = audio.cpu().numpy()

    # 1. RMS level (too low = silence/failure)
    rms = np.sqrt(np.mean(audio_np ** 2))

    # 2. Zero crossing rate (high = noise/gibberish)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np)))) / 2
    zcr = zero_crossings / len(audio_np)

    # 3. Silence ratio (high = failed generation)
    silence_threshold = 0.01
    silence_ratio = np.sum(np.abs(audio_np) < silence_threshold) / len(audio_np)

    # 4. Peak to RMS ratio (crest factor) - very high = clicks/artifacts
    peak = np.max(np.abs(audio_np))
    crest_factor = peak / (rms + 1e-8)

    # 5. Duration check
    duration_ms = len(audio_np) / sample_rate * 1000

    # Determine quality
    issues = []
    if rms < 0.01:
        issues.append("too_quiet")
    if zcr > 0.3:
        issues.append("high_noise")
    if silence_ratio > 0.5:
        issues.append("mostly_silence")
    if crest_factor > 20:
        issues.append("artifacts")
    if duration_ms < 200:
        issues.append("too_short")

    return {
        "valid": len(issues) == 0,
        "rms": rms,
        "zcr": zcr,
        "silence_ratio": silence_ratio,
        "crest_factor": crest_factor,
        "duration_ms": duration_ms,
        "issues": issues
    }


def main():
    print("=" * 70)
    print("TTS PATTERN TESTING")
    print("Testing which patterns work with our CURRENT fine-tuned model")
    print("=" * 70)

    # Load TTS
    print("\nLoading TTS engine...")
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Test patterns - categorized
    test_patterns = {
        "SAFE (proven)": [
            "oh wow thats really cool",
            "aww im sorry to hear that",
            "oh thats interesting tell me more",
            "i totally agree with you",
            "that sounds amazing",
        ],
        "NATURAL DISFLUENCIES (testing)": [
            "hmm thats interesting",
            "yeah i get what you mean",
            "mhm that makes sense",
            "um let me think about that",
            "right i understand",
        ],
        "THINKING PATTERNS (testing)": [
            "hmm",
            "yeah",
            "mhm",
            "well thats a good question",
            "uh huh i see",
        ],
        "MIXED (testing)": [
            "hmm yeah thats cool",
            "oh hmm interesting point",
            "yeah wow thats amazing",
        ],
    }

    results = {}

    for category, patterns in test_patterns.items():
        print(f"\n{'=' * 70}")
        print(f"CATEGORY: {category}")
        print("=" * 70)

        category_results = []

        for text in patterns:
            print(f"\n  Testing: \"{text}\"")

            # Generate audio
            torch.cuda.synchronize()
            start = time.time()

            try:
                # Use generate (not stream) for consistent measurement
                audio_chunks = []
                for chunk in tts.generate_stream(text, use_context=False):
                    audio_chunks.append(chunk)

                if audio_chunks:
                    audio = torch.cat(audio_chunks)
                else:
                    audio = torch.tensor([])

                torch.cuda.synchronize()
                gen_time = (time.time() - start) * 1000

                # Analyze quality
                quality = calculate_audio_quality(audio)

                status = "✅ GOOD" if quality["valid"] else f"❌ ISSUES: {quality['issues']}"
                print(f"    {status}")
                print(f"    Duration: {quality['duration_ms']:.0f}ms, Gen time: {gen_time:.0f}ms")
                print(f"    RMS: {quality['rms']:.3f}, ZCR: {quality['zcr']:.3f}")

                category_results.append({
                    "text": text,
                    "valid": quality["valid"],
                    "quality": quality,
                    "gen_time": gen_time,
                })

            except Exception as e:
                print(f"    ❌ ERROR: {e}")
                category_results.append({
                    "text": text,
                    "valid": False,
                    "error": str(e),
                })

        results[category] = category_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for category, cat_results in results.items():
        valid_count = sum(1 for r in cat_results if r.get("valid", False))
        total = len(cat_results)
        pct = valid_count / total * 100

        status = "✅" if pct >= 80 else "⚠️" if pct >= 50 else "❌"
        print(f"\n  {status} {category}: {valid_count}/{total} ({pct:.0f}%)")

        # Show failures
        failures = [r for r in cat_results if not r.get("valid", False)]
        if failures:
            print("    Failures:")
            for f in failures:
                issues = f.get("quality", {}).get("issues", [f.get("error", "unknown")])
                print(f"      - \"{f['text']}\": {issues}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    natural_results = results.get("NATURAL DISFLUENCIES (testing)", [])
    natural_valid = sum(1 for r in natural_results if r.get("valid", False))

    if natural_valid == len(natural_results):
        print("\n  ✅ NATURAL DISFLUENCIES WORK!")
        print("  The fine-tuned model handles 'hmm', 'yeah', 'mhm' correctly.")
        print("  We can safely use natural speech patterns.")
    elif natural_valid >= len(natural_results) * 0.8:
        print("\n  ⚠️ MOSTLY WORKS")
        print("  Most natural patterns work, some may need filtering.")
    else:
        print("\n  ❌ NATURAL PATTERNS CAUSE ISSUES")
        print("  Should stick to proven 'Oh wow!', 'Aww,' patterns.")


if __name__ == "__main__":
    main()
