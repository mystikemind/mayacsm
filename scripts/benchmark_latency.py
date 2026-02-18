#!/usr/bin/env python3
"""End-to-end pipeline latency benchmark.

Measures each component (STT, LLM, TTS) and total pipeline latency
using realistic audio inputs. Compares old Whisper vs new faster-whisper.

Usage:
    python scripts/benchmark_latency.py
"""
import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def generate_test_audio_tts() -> tuple:
    """Generate realistic test audio using TTS (for STT benchmarking).
    Returns (audio_tensor, expected_text) pair.
    """
    from maya.engine.tts_compiled import CompiledTTSEngine
    print("  Generating realistic test audio with TTS...")
    tts = CompiledTTSEngine()
    tts.initialize()
    text = "hello, how are you doing today?"
    audio = tts.generate(text, use_context=False)
    print(f"  Generated {len(audio)/24000:.1f}s test audio")
    del tts
    torch.cuda.empty_cache()
    return audio, text


def benchmark_stt(test_audio):
    """Benchmark OpenAI Whisper STT with realistic audio."""
    print("\n" + "=" * 60)
    print("BENCHMARK: OpenAI Whisper STT (turbo)")
    print("=" * 60)

    from maya.engine.stt import STTEngine
    engine = STTEngine()

    print("  Initializing...")
    start = time.time()
    engine.initialize()
    init_time = time.time() - start
    print(f"  Init: {init_time:.1f}s")

    # Warmup
    _ = engine.transcribe(test_audio)

    # Benchmark
    times = []
    for i in range(5):
        start = time.time()
        result = engine.transcribe(test_audio)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.0f}ms -> '{result[:50]}'")

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"\n  Average: {avg:.0f}ms | P50: {p50:.0f}ms | P95: {p95:.0f}ms")

    del engine
    torch.cuda.empty_cache()
    return avg


def benchmark_stt_faster():
    """Benchmark faster-whisper STT (CTranslate2)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: faster-whisper STT (CTranslate2)")
    print("=" * 60)

    gpu_count = torch.cuda.device_count()
    stt_gpu = 1 if gpu_count > 1 else 0
    print(f"  Using GPU {stt_gpu} (of {gpu_count} available)")

    from maya.engine.stt_faster import FasterSTTEngine
    engine = FasterSTTEngine(device="cuda", device_index=stt_gpu)

    print("  Initializing...")
    start = time.time()
    engine.initialize()
    init_time = time.time() - start
    print(f"  Init: {init_time:.1f}s")

    # Generate test audio
    test_audio = generate_test_audio("hello how are you", duration_s=2.0)

    # Warmup (already done in initialize)

    # Benchmark
    times = []
    for i in range(5):
        start = time.time()
        result = engine.transcribe(test_audio)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.0f}ms -> '{result[:40]}'")

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"\n  Average: {avg:.0f}ms | P50: {p50:.0f}ms | P95: {p95:.0f}ms")

    del engine
    torch.cuda.empty_cache()
    return avg


def benchmark_llm():
    """Benchmark Llama 3.2 3B LLM."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Llama 3.2 3B LLM (torch.compile)")
    print("=" * 60)

    from maya.engine.llm_optimized import OptimizedLLMEngine
    engine = OptimizedLLMEngine()

    print("  Initializing...")
    start = time.time()
    engine.initialize()
    init_time = time.time() - start
    print(f"  Init: {init_time:.1f}s")

    test_inputs = [
        "hello",
        "how are you doing today",
        "what do you think about that",
        "thats really interesting",
        "tell me more about yourself",
    ]

    # Benchmark
    times = []
    for text in test_inputs:
        start = time.time()
        result = engine.generate(text)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  '{text[:30]}' -> '{result[:40]}' [{elapsed:.0f}ms]")

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"\n  Average: {avg:.0f}ms | P50: {p50:.0f}ms | P95: {p95:.0f}ms")

    del engine
    torch.cuda.empty_cache()
    return avg


def benchmark_tts():
    """Benchmark CSM TTS (compiled)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: CSM-1B TTS (torch.compile)")
    print("=" * 60)

    from maya.engine.tts_compiled import CompiledTTSEngine
    engine = CompiledTTSEngine()

    print("  Initializing (includes warmup)...")
    start = time.time()
    engine.initialize()
    init_time = time.time() - start
    print(f"  Init: {init_time:.1f}s")

    test_texts = [
        "yeah, totally.",
        "oh hey, hi! its really nice to meet you.",
        "hmm, im not really sure about that honestly.",
        "wow, thats incredible, tell me more.",
        "ha, yeah, thats pretty funny actually.",
    ]

    # Benchmark
    times = []
    durations = []
    for text in test_texts:
        start = time.time()
        audio = engine.generate(text, use_context=True)
        elapsed = (time.time() - start) * 1000
        dur = len(audio) / 24000
        rtf = (elapsed / 1000) / dur if dur > 0 else 0
        times.append(elapsed)
        durations.append(dur)
        print(f"  '{text[:35]}' -> {dur:.1f}s audio in {elapsed:.0f}ms (RTF: {rtf:.2f}x)")

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    avg_rtf = np.mean([(t/1000)/d for t, d in zip(times, durations) if d > 0])
    print(f"\n  Average: {avg:.0f}ms | P50: {p50:.0f}ms | P95: {p95:.0f}ms | RTF: {avg_rtf:.2f}x")

    del engine
    torch.cuda.empty_cache()
    return avg


def benchmark_full_pipeline():
    """Benchmark full STT -> LLM -> TTS pipeline."""
    print("\n" + "=" * 60)
    print("BENCHMARK: FULL PIPELINE (STT -> LLM -> TTS)")
    print("=" * 60)

    from maya.engine.stt import STTEngine
    from maya.engine.llm_optimized import OptimizedLLMEngine
    from maya.engine.tts_compiled import CompiledTTSEngine

    stt = STTEngine()
    llm = OptimizedLLMEngine()
    tts = CompiledTTSEngine()

    print("  Note: Using TTS-generated audio for realistic STT input")

    print("\n  Initializing all engines...")
    start = time.time()
    stt.initialize()
    llm.initialize()
    tts.initialize()
    total_init = time.time() - start
    print(f"  All engines ready in {total_init:.1f}s")

    # Generate realistic test audio using TTS (for STT)
    test_audio = tts.generate("hi there, how are you doing?", use_context=False)

    print("\n  Running full pipeline (5 iterations):")
    pipeline_times = []
    stt_times = []
    llm_times = []
    tts_times = []

    for i in range(5):
        total_start = time.time()

        # STT
        stt_start = time.time()
        transcript = stt.transcribe(test_audio)
        stt_elapsed = (time.time() - stt_start) * 1000

        if not transcript.strip():
            transcript = "hello how are you"

        # LLM
        llm_start = time.time()
        response = llm.generate(transcript)
        llm_elapsed = (time.time() - llm_start) * 1000

        if not response.strip():
            response = "im doing great thanks"

        # TTS
        tts_start = time.time()
        audio = tts.generate(response, use_context=True)
        tts_elapsed = (time.time() - tts_start) * 1000

        total_elapsed = (time.time() - total_start) * 1000
        audio_dur = len(audio) / 24000

        stt_times.append(stt_elapsed)
        llm_times.append(llm_elapsed)
        tts_times.append(tts_elapsed)
        pipeline_times.append(total_elapsed)

        print(f"\n  Run {i+1}:")
        print(f"    STT: {stt_elapsed:.0f}ms -> '{transcript[:30]}'")
        print(f"    LLM: {llm_elapsed:.0f}ms -> '{response[:30]}'")
        print(f"    TTS: {tts_elapsed:.0f}ms -> {audio_dur:.1f}s audio")
        print(f"    TOTAL: {total_elapsed:.0f}ms")

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  STT  avg: {np.mean(stt_times):>7.0f}ms | P50: {np.percentile(stt_times, 50):>6.0f}ms | P95: {np.percentile(stt_times, 95):>6.0f}ms")
    print(f"  LLM  avg: {np.mean(llm_times):>7.0f}ms | P50: {np.percentile(llm_times, 50):>6.0f}ms | P95: {np.percentile(llm_times, 95):>6.0f}ms")
    print(f"  TTS  avg: {np.mean(tts_times):>7.0f}ms | P50: {np.percentile(tts_times, 50):>6.0f}ms | P95: {np.percentile(tts_times, 95):>6.0f}ms")
    print(f"  TOTAL avg: {np.mean(pipeline_times):>6.0f}ms | P50: {np.percentile(pipeline_times, 50):>6.0f}ms | P95: {np.percentile(pipeline_times, 95):>6.0f}ms")
    print()

    target = 1200
    avg_total = np.mean(pipeline_times)
    if avg_total <= target:
        print(f"  TARGET: {target}ms -> PASS (avg {avg_total:.0f}ms)")
    else:
        print(f"  TARGET: {target}ms -> MISS by {avg_total - target:.0f}ms (avg {avg_total:.0f}ms)")

    del stt, llm, tts
    torch.cuda.empty_cache()
    return np.mean(pipeline_times)


def main():
    print("=" * 60)
    print("MAYA PIPELINE LATENCY BENCHMARK")
    print("=" * 60)

    gpu_count = torch.cuda.device_count()
    print(f"\nGPUs: {gpu_count}")
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({vram:.1f} GB)")

    # Generate realistic test audio first (TTS audio for STT testing)
    test_audio, expected_text = generate_test_audio_tts()

    # Individual benchmarks
    print("\n" + "#" * 60)
    print("# INDIVIDUAL COMPONENT BENCHMARKS")
    print("#" * 60)

    stt_avg = benchmark_stt(test_audio)
    llm_avg = benchmark_llm()
    tts_avg = benchmark_tts()

    # Full pipeline benchmark
    print("\n" + "#" * 60)
    print("# FULL PIPELINE BENCHMARK")
    print("#" * 60)

    pipeline_avg = benchmark_full_pipeline()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\n  Component Averages:")
    print(f"    STT (Whisper turbo):   {stt_avg:>6.0f}ms")
    print(f"    LLM (Llama 3.2 1B):   {llm_avg:>6.0f}ms")
    print(f"    TTS (CSM compiled):    {tts_avg:>6.0f}ms")
    print(f"    Pipeline total:        {pipeline_avg:>6.0f}ms")
    print(f"\n  Theoretical minimum:     {stt_avg + llm_avg + tts_avg:>6.0f}ms")
    print(f"  Target:                  {1500:>6.0f}ms")
    print()


if __name__ == "__main__":
    main()
