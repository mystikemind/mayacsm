#!/usr/bin/env python3
"""
SENIOR ENGINEER BENCHMARK: Test different codebook configurations

This tests EXACTLY what matters:
1. RTF (must be < 1.0 for smooth audio)
2. Audio quality (intelligibility)
3. First chunk latency
4. Full pipeline latency

Run each config and save audio files for comparison.
"""
import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import torchaudio
import time
import os

os.makedirs("/home/ec2-user/SageMaker/project_maya/tests/outputs/benchmark", exist_ok=True)

# Test sentences of varying lengths
TEST_SENTENCES = [
    "Hello there",
    "How are you doing today?",
    "I think that is a really interesting question.",
    "Let me think about that for a moment, it's quite complex.",
]

def test_codebook_config(num_codebooks: int):
    """Test a specific codebook configuration."""
    print(f"\n{'='*60}")
    print(f"TESTING {num_codebooks} CODEBOOKS")
    print(f"{'='*60}")

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Import fresh to avoid cached state
    from importlib import reload
    import maya.engine.tts_streaming as tts_module
    reload(tts_module)

    from maya.engine.tts_streaming import StreamingTTSEngine, StreamingConfig

    # Create engine and manually set codebooks
    tts = StreamingTTSEngine()

    # Patch the num_codebooks before initialization
    original_init = tts.initialize
    def patched_init():
        original_init()
        # Override after init
        tts._num_codebooks = num_codebooks
        tts._audio_tokenizer.set_num_codebooks(num_codebooks)

    tts.initialize()
    tts._num_codebooks = num_codebooks
    tts._audio_tokenizer.set_num_codebooks(num_codebooks)

    print(f"Configured codebooks: {tts._num_codebooks}")

    config = StreamingConfig(
        initial_batch_size=4,
        batch_size=12,
        max_audio_length_ms=5000,
        temperature=0.8,
        topk=50
    )

    results = []

    for idx, text in enumerate(TEST_SENTENCES):
        print(f"\nTest {idx+1}: '{text[:40]}...'")

        chunks = []
        start = time.time()
        first_chunk_time = None

        for chunk in tts._generate_frames_sync(
            text=tts.preprocess_text(text),
            speaker=0,
            context=[],
            config=config
        ):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - start) * 1000
            chunks.append(chunk)

        total_time = time.time() - start

        if chunks:
            full_audio = torch.cat(chunks)
            total_samples = len(full_audio)
            audio_duration = total_samples / 24000
            rtf = total_time / audio_duration

            # Normalize audio
            full_audio = full_audio - full_audio.mean()
            peak = full_audio.abs().max()
            if peak > 0:
                full_audio = full_audio * (0.7 / peak)

            # Save audio
            output_path = f"/home/ec2-user/SageMaker/project_maya/tests/outputs/benchmark/{num_codebooks}cb_test{idx+1}.wav"
            torchaudio.save(output_path, full_audio.unsqueeze(0).cpu(), 24000)

            status = "✓ REAL-TIME" if rtf < 1.0 else ("~ CLOSE" if rtf < 1.3 else "✗ TOO SLOW")
            print(f"  First chunk: {first_chunk_time:.0f}ms")
            print(f"  Audio: {audio_duration:.2f}s, Gen: {total_time:.2f}s")
            print(f"  RTF: {rtf:.2f}x {status}")
            print(f"  Saved: {output_path}")

            results.append({
                'text': text,
                'rtf': rtf,
                'first_chunk': first_chunk_time,
                'audio_duration': audio_duration,
            })

    # Summary
    if results:
        avg_rtf = sum(r['rtf'] for r in results) / len(results)
        avg_first = sum(r['first_chunk'] for r in results) / len(results)
        print(f"\n--- {num_codebooks} CODEBOOK SUMMARY ---")
        print(f"Average RTF: {avg_rtf:.2f}x")
        print(f"Average first chunk: {avg_first:.0f}ms")
        print(f"Status: {'REAL-TIME CAPABLE' if avg_rtf < 1.0 else 'NEEDS BUFFERING' if avg_rtf < 1.5 else 'TOO SLOW'}")
        return avg_rtf, avg_first

    return None, None


if __name__ == "__main__":
    print("=" * 60)
    print("CODEBOOK BENCHMARK - Finding optimal quality/speed tradeoff")
    print("=" * 60)

    # Test different configurations
    configs_to_test = [32, 24, 16, 12, 8]

    results = {}
    for num_cb in configs_to_test:
        try:
            rtf, first_chunk = test_codebook_config(num_cb)
            results[num_cb] = (rtf, first_chunk)
        except Exception as e:
            print(f"Error testing {num_cb} codebooks: {e}")
            results[num_cb] = (None, None)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Codebooks':<12} {'RTF':<10} {'First Chunk':<15} {'Status'}")
    print("-" * 60)

    for num_cb, (rtf, first_chunk) in results.items():
        if rtf is not None:
            status = "✓ REAL-TIME" if rtf < 1.0 else ("~ BUFFERABLE" if rtf < 1.5 else "✗ TOO SLOW")
            print(f"{num_cb:<12} {rtf:.2f}x{'':<6} {first_chunk:.0f}ms{'':<10} {status}")
        else:
            print(f"{num_cb:<12} ERROR")

    print("\n" + "=" * 60)
    print("RECOMMENDATION:")

    # Find best config
    best_realtime = None
    best_bufferable = None
    for num_cb in sorted(results.keys(), reverse=True):  # Prefer higher quality
        rtf, _ = results[num_cb]
        if rtf is not None:
            if rtf < 1.0 and best_realtime is None:
                best_realtime = num_cb
            if rtf < 1.5 and best_bufferable is None:
                best_bufferable = num_cb

    if best_realtime:
        print(f"For REAL-TIME (no gaps): Use {best_realtime} codebooks")
    if best_bufferable and best_bufferable != best_realtime:
        print(f"For BUFFERED (better quality): Use {best_bufferable} codebooks with 500ms buffer")

    print("\nAudio files saved to: tests/outputs/benchmark/")
    print("Listen to compare quality at different codebook levels.")
    print("=" * 60)
