#!/usr/bin/env python3
"""
Engineering Test Suite - Objective Audio Quality Metrics

Tests what actually matters for real-time voice AI:
1. RTF (Real-Time Factor) - MUST be < 1.0 for smooth playback
2. TTFA (Time to First Audio) - Target < 1 second
3. Chunk discontinuity analysis - Detect clicks/pops
4. Spectrogram analysis - Visual inspection for artifacts
5. Latency breakdown - Where is time being spent?

This is how a real audio engineer would validate changes.
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

import torch
import torchaudio
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import json

OUTPUT_DIR = Path("/home/ec2-user/SageMaker/project_maya/tests/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GenerationMetrics:
    """Metrics from a single generation."""
    text: str
    total_time_ms: float
    first_chunk_time_ms: float
    audio_duration_ms: float
    rtf: float
    num_chunks: int
    chunk_sizes: List[int]
    max_discontinuity: float
    mean_discontinuity: float


def analyze_discontinuities(chunks: List[torch.Tensor]) -> Tuple[float, float, List[float]]:
    """
    Analyze audio discontinuities at chunk boundaries.

    Returns:
        (max_discontinuity, mean_discontinuity, all_discontinuities)
    """
    if len(chunks) < 2:
        return 0.0, 0.0, []

    discontinuities = []
    for i in range(len(chunks) - 1):
        if len(chunks[i]) > 0 and len(chunks[i+1]) > 0:
            end_val = chunks[i][-1].item()
            start_val = chunks[i+1][0].item()
            disc = abs(end_val - start_val)
            discontinuities.append(disc)

    if not discontinuities:
        return 0.0, 0.0, []

    return max(discontinuities), np.mean(discontinuities), discontinuities


def plot_waveform_boundaries(chunks: List[torch.Tensor], output_path: Path, title: str):
    """Plot waveform with chunk boundaries marked."""
    if not chunks:
        return

    full_audio = torch.cat(chunks).cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full waveform
    axes[0].plot(full_audio, linewidth=0.5)
    axes[0].set_title(f'{title} - Full Waveform')
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Amplitude')

    # Mark chunk boundaries
    pos = 0
    colors = ['r', 'g', 'b', 'orange', 'purple']
    for i, chunk in enumerate(chunks):
        axes[0].axvline(x=pos, color=colors[i % len(colors)], alpha=0.5, linestyle='--')
        pos += len(chunk)

    # Zoomed view at first boundary (if exists)
    if len(chunks) >= 2:
        boundary_pos = len(chunks[0])
        window = 500  # samples around boundary
        start = max(0, boundary_pos - window)
        end = min(len(full_audio), boundary_pos + window)

        axes[1].plot(range(start, end), full_audio[start:end], linewidth=1)
        axes[1].axvline(x=boundary_pos, color='red', linestyle='--', label='Chunk boundary')
        axes[1].set_title('Zoomed: First Chunk Boundary')
        axes[1].set_xlabel('Samples')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_spectrogram(audio: torch.Tensor, output_path: Path, title: str, sample_rate: int = 24000):
    """Generate spectrogram to visually inspect for artifacts."""
    audio_np = audio.cpu().numpy()

    fig, ax = plt.subplots(figsize=(14, 4))

    # Compute spectrogram
    spec = ax.specgram(audio_np, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    ax.set_title(f'{title} - Spectrogram')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(0, 8000)  # Focus on speech frequencies

    plt.colorbar(spec[3], ax=ax, label='dB')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_generation_test(tts, text: str, config) -> Tuple[GenerationMetrics, List[torch.Tensor]]:
    """Run a single generation and collect metrics."""
    start_time = time.time()
    first_chunk_time = None
    chunks = []

    for chunk in tts._generate_frames_sync(
        text=tts.preprocess_text(text),
        speaker=0,
        context=[],
        config=config
    ):
        if first_chunk_time is None:
            first_chunk_time = (time.time() - start_time) * 1000
        chunks.append(chunk)

    total_time = (time.time() - start_time) * 1000

    if not chunks:
        return None, []

    total_samples = sum(len(c) for c in chunks)
    audio_duration = total_samples / 24000 * 1000  # ms
    rtf = (total_time / 1000) / (audio_duration / 1000) if audio_duration > 0 else float('inf')

    max_disc, mean_disc, _ = analyze_discontinuities(chunks)

    metrics = GenerationMetrics(
        text=text,
        total_time_ms=total_time,
        first_chunk_time_ms=first_chunk_time or 0,
        audio_duration_ms=audio_duration,
        rtf=rtf,
        num_chunks=len(chunks),
        chunk_sizes=[len(c) for c in chunks],
        max_discontinuity=max_disc,
        mean_discontinuity=mean_disc
    )

    return metrics, chunks


def test_rtf_after_warmup():
    """
    Test RTF after proper warmup - this is the REAL performance metric.

    The warmup is critical because torch.compile needs to trace the graph
    on first few runs. After warmup, RTF should be < 1.0.
    """
    print("=" * 70)
    print("TEST 1: RTF AFTER WARMUP")
    print("=" * 70)
    print("\nThis tests real-world performance after model warmup.")
    print("RTF MUST be < 1.0 for smooth real-time playback.\n")

    from maya.engine.tts_streaming import StreamingTTSEngine, StreamingConfig

    tts = StreamingTTSEngine()
    tts.initialize()  # This includes 3 warmup generations

    config = StreamingConfig(
        initial_batch_size=4,
        batch_size=12,
        max_audio_length_ms=10000,
        temperature=0.8,
        topk=50
    )

    # Test sentences of varying length
    test_sentences = [
        "Hi there!",
        "How are you doing today?",
        "I think that's a really interesting question.",
        "Well, let me think about that for a moment. There are several ways we could approach this problem.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    ]

    results = []

    for i, text in enumerate(test_sentences):
        print(f"Test {i+1}/{len(test_sentences)}: '{text[:40]}...'")

        metrics, chunks = run_generation_test(tts, text, config)

        if metrics:
            results.append(metrics)
            rtf_status = "✓ PASS" if metrics.rtf < 1.0 else "✗ FAIL"
            print(f"  RTF: {metrics.rtf:.3f}x {rtf_status}")
            print(f"  TTFA: {metrics.first_chunk_time_ms:.0f}ms")
            print(f"  Audio: {metrics.audio_duration_ms:.0f}ms")
            print(f"  Max discontinuity: {metrics.max_discontinuity:.6f}")

            if chunks:
                full_audio = torch.cat(chunks)

                # Save waveform plot
                plot_waveform_boundaries(
                    chunks,
                    OUTPUT_DIR / f"waveform_test{i+1}.png",
                    f"Test {i+1}"
                )

                # Save spectrogram
                plot_spectrogram(
                    full_audio,
                    OUTPUT_DIR / f"spectrogram_test{i+1}.png",
                    f"Test {i+1}"
                )

                # Save audio
                torchaudio.save(
                    str(OUTPUT_DIR / f"audio_test{i+1}.wav"),
                    full_audio.unsqueeze(0).cpu(),
                    24000
                )
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        avg_rtf = np.mean([r.rtf for r in results])
        avg_ttfa = np.mean([r.first_chunk_time_ms for r in results])
        max_disc = max(r.max_discontinuity for r in results)

        print(f"\nAverage RTF: {avg_rtf:.3f}x", end="")
        if avg_rtf < 1.0:
            print(" ✓ GOOD - Real-time capable")
        elif avg_rtf < 1.5:
            print(" ⚠ WARNING - May have occasional stutters")
        else:
            print(" ✗ FAIL - Will have audio cuts")

        print(f"Average TTFA: {avg_ttfa:.0f}ms", end="")
        if avg_ttfa < 500:
            print(" ✓ EXCELLENT")
        elif avg_ttfa < 1000:
            print(" ✓ GOOD")
        elif avg_ttfa < 2000:
            print(" ⚠ ACCEPTABLE")
        else:
            print(" ✗ TOO SLOW")

        print(f"Max discontinuity: {max_disc:.6f}", end="")
        if max_disc < 0.01:
            print(" ✓ EXCELLENT - No audible clicks")
        elif max_disc < 0.05:
            print(" ✓ GOOD")
        elif max_disc < 0.1:
            print(" ⚠ MAY HAVE MINOR ARTIFACTS")
        else:
            print(" ✗ LIKELY AUDIBLE CLICKS")

        # Save results to JSON
        results_dict = {
            "summary": {
                "avg_rtf": avg_rtf,
                "avg_ttfa_ms": avg_ttfa,
                "max_discontinuity": max_disc,
                "rtf_pass": avg_rtf < 1.0,
            },
            "tests": [
                {
                    "text": r.text,
                    "rtf": r.rtf,
                    "ttfa_ms": r.first_chunk_time_ms,
                    "audio_ms": r.audio_duration_ms,
                    "max_discontinuity": r.max_discontinuity,
                }
                for r in results
            ]
        }

        with open(OUTPUT_DIR / "test_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("- test_results.json (metrics)")
        print("- waveform_*.png (chunk boundary analysis)")
        print("- spectrogram_*.png (artifact detection)")
        print("- audio_*.wav (listening test)")

    return results


def test_latency_breakdown():
    """
    Break down latency by component to identify bottlenecks.
    """
    print("\n" + "=" * 70)
    print("TEST 2: LATENCY BREAKDOWN")
    print("=" * 70)
    print("\nMeasuring time spent in each pipeline component.\n")

    from maya.engine.vad import VADEngine
    from maya.engine.stt import STTEngine
    from maya.engine.llm import LLMEngine
    from maya.engine.tts_streaming import StreamingTTSEngine, StreamingConfig

    # Generate test audio (simulate user speech)
    print("Generating test audio...")
    sample_rate = 24000
    duration = 2.0  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Simple sine wave as placeholder (real test would use recorded speech)
    test_audio = 0.5 * torch.sin(2 * 3.14159 * 440 * t)

    # Initialize components
    print("Initializing components...")
    vad = VADEngine()
    vad.initialize()

    stt = STTEngine()
    stt.initialize()

    llm = LLMEngine()
    llm.initialize()

    tts = StreamingTTSEngine()
    tts.initialize()

    # Measure each component
    print("\nRunning latency measurements...\n")

    # STT
    torch.cuda.synchronize()
    start = time.time()
    transcript = stt.transcribe(test_audio)
    torch.cuda.synchronize()
    stt_time = (time.time() - start) * 1000
    print(f"STT: {stt_time:.0f}ms - '{transcript[:50] if transcript else 'N/A'}...'")

    # LLM
    test_input = "Hello, how are you today?"
    torch.cuda.synchronize()
    start = time.time()
    response = llm.generate(test_input)
    torch.cuda.synchronize()
    llm_time = (time.time() - start) * 1000
    print(f"LLM: {llm_time:.0f}ms - '{response[:50]}...'")

    # TTS (first chunk only)
    config = StreamingConfig(
        initial_batch_size=4,
        batch_size=12,
        max_audio_length_ms=5000,
        temperature=0.8,
        topk=50
    )

    torch.cuda.synchronize()
    start = time.time()
    first_chunk_time = None
    for chunk in tts._generate_frames_sync(
        text=tts.preprocess_text(response),
        speaker=0,
        context=[],
        config=config
    ):
        if first_chunk_time is None:
            torch.cuda.synchronize()
            first_chunk_time = (time.time() - start) * 1000
        # Continue to complete generation
    torch.cuda.synchronize()
    tts_total = (time.time() - start) * 1000

    print(f"TTS first chunk: {first_chunk_time:.0f}ms")
    print(f"TTS total: {tts_total:.0f}ms")

    # Total pipeline estimate
    total = stt_time + llm_time + first_chunk_time
    print(f"\n--- ESTIMATED PIPELINE LATENCY ---")
    print(f"STT + LLM + TTS(first): {total:.0f}ms")

    if total < 1500:
        print("✓ Under 1.5 second target")
    elif total < 2000:
        print("⚠ Under 2 second acceptable limit")
    else:
        print("✗ Over 2 seconds - needs optimization")

    return {
        "stt_ms": stt_time,
        "llm_ms": llm_time,
        "tts_first_chunk_ms": first_chunk_time,
        "tts_total_ms": tts_total,
        "total_estimated_ms": total
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MAYA ENGINEERING TEST SUITE")
    print("=" * 70)
    print("Testing objective metrics that matter for real-time voice AI")
    print("=" * 70 + "\n")

    # Run tests
    rtf_results = test_rtf_after_warmup()
    latency_results = test_latency_breakdown()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Check waveform plots for chunk boundary smoothness")
    print("2. Check spectrograms for artifacts (vertical lines = clicks)")
    print("3. Listen to audio files for subjective quality")
    print("4. If RTF > 1.0, optimize or use smaller chunks")
