#!/usr/bin/env python3
"""
Test latency of CSM TTS generation.

Measures:
- Time to first audio chunk (TTFA) - most critical metric
- Total generation time
- Real-time factor (RTF)
"""

import sys
import time
import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from maya.engine.tts_streaming import StreamingTTSEngine, FAST_CONFIG


def test_latency():
    """Run latency benchmarks."""
    print("=" * 60)
    print("CSM-1B LATENCY BENCHMARK")
    print("=" * 60)

    # Initialize engine
    print("\nInitializing TTS engine...")
    engine = StreamingTTSEngine()
    engine.initialize()
    print("Engine ready.\n")

    # Test cases - from short starters to longer responses
    test_cases = [
        ("Short starter", "hmm let me think", FAST_CONFIG),
        ("Medium starter", "oh thats interesting well", FAST_CONFIG),
        ("Short response", "yeah i think that makes sense", None),
        ("Medium response", "well thats a good question let me explain how that works", None),
        ("Long response", "oh i understand what youre saying and i think the best approach would be to consider all the options before making a decision", None),
    ]

    results = []

    for name, text, config in test_cases:
        print(f"\n--- {name} ---")
        print(f"Input: '{text}'")

        torch.cuda.synchronize()
        start = time.time()

        first_chunk_time = None
        total_samples = 0
        chunk_count = 0

        # Generate with streaming
        for chunk in engine._generate_frames_sync(
            text=engine.preprocess_text(text),
            speaker=0,
            context=[],
            config=config
        ):
            if first_chunk_time is None:
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - start) * 1000

            total_samples += len(chunk)
            chunk_count += 1

        torch.cuda.synchronize()
        total_time = (time.time() - start) * 1000

        duration = total_samples / 24000
        rtf = (total_time / 1000) / duration if duration > 0 else 0

        result = {
            "name": name,
            "text": text[:30],
            "ttfa_ms": first_chunk_time,
            "total_ms": total_time,
            "audio_sec": duration,
            "rtf": rtf,
            "chunks": chunk_count
        }
        results.append(result)

        print(f"  TTFA: {first_chunk_time:.0f}ms")
        print(f"  Total: {total_time:.0f}ms")
        print(f"  Audio: {duration:.2f}s ({chunk_count} chunks)")
        print(f"  RTF: {rtf:.2f}x real-time")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<20} {'TTFA (ms)':<12} {'Total (ms)':<12} {'RTF':<8}")
    print("-" * 52)

    for r in results:
        print(f"{r['name']:<20} {r['ttfa_ms']:<12.0f} {r['total_ms']:<12.0f} {r['rtf']:<8.2f}")

    avg_ttfa = sum(r['ttfa_ms'] for r in results) / len(results)
    avg_rtf = sum(r['rtf'] for r in results) / len(results)

    print("-" * 52)
    print(f"{'AVERAGE':<20} {avg_ttfa:<12.0f} {'':<12} {avg_rtf:<8.2f}")

    # Target check
    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)

    fast_results = [r for r in results if "starter" in r['name'].lower()]
    fast_ttfa = sum(r['ttfa_ms'] for r in fast_results) / len(fast_results) if fast_results else 0

    print(f"Fast starter avg TTFA: {fast_ttfa:.0f}ms (target: 200-400ms)")
    if fast_ttfa < 400:
        print("  [PASS] Fast starters are under 400ms TTFA")
    else:
        print("  [FAIL] Fast starters exceed 400ms target")

    print(f"\nOverall avg RTF: {avg_rtf:.2f}x (target: <0.5x for real-time)")
    if avg_rtf < 0.5:
        print("  [PASS] Faster than real-time")
    elif avg_rtf < 1.0:
        print("  [WARN] Close to real-time, may cause delays")
    else:
        print("  [FAIL] Slower than real-time - will cause delays")

    return results


if __name__ == "__main__":
    test_latency()
