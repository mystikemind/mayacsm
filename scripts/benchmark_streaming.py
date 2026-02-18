#!/usr/bin/env python3
"""Benchmark: Streaming TTS pipeline vs non-streaming.

Measures time-to-first-audio with the new streaming TTS,
comparing against the previous full-generation approach.

Usage:
    python scripts/benchmark_streaming.py

    # Or with cuDNN path explicitly set:
    LD_LIBRARY_PATH=/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH python scripts/benchmark_streaming.py
"""
import sys
import os

# Set cuDNN 9 library path for faster-whisper (CTranslate2)
# Note: This should be set BEFORE Python starts for full effect,
# but we add it here as a fallback for convenience.
_cudnn_path = "/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/nvidia/cudnn/lib"
if os.path.exists(_cudnn_path):
    _ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _cudnn_path not in _ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_cudnn_path}:{_ld_path}"

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def benchmark_streaming_tts():
    """Benchmark TRUE streaming TTS (time to first chunk)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: STREAMING TTS (first chunk latency)")
    print("=" * 60)

    from maya.engine.tts_compiled import CompiledTTSEngine
    engine = CompiledTTSEngine()

    print("  Initializing (includes warmup)...")
    start = time.time()
    engine.initialize()
    print(f"  Init: {time.time()-start:.1f}s")

    test_texts = [
        "yeah, totally.",
        "oh hey, hi! its really nice to meet you.",
        "hmm, im not really sure about that honestly.",
        "wow, thats incredible, tell me more.",
        "ha, yeah, thats pretty funny actually.",
    ]

    print("\n  --- Streaming mode (generate_stream) ---")
    first_chunk_times = []
    total_times = []
    total_durations = []

    for text in test_texts:
        start = time.time()
        first_chunk_time = None
        chunks = []
        total_samples = 0

        for chunk in engine.generate_stream(text, use_context=True):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - start) * 1000
            chunks.append(chunk)
            total_samples += len(chunk)

        total_time = (time.time() - start) * 1000
        duration = total_samples / 24000

        first_chunk_times.append(first_chunk_time)
        total_times.append(total_time)
        total_durations.append(duration)

        num_chunks = len(chunks)
        print(f"  '{text[:35]}' -> first={first_chunk_time:.0f}ms, total={total_time:.0f}ms, "
              f"{duration:.1f}s audio, {num_chunks} chunks")

    avg_first = np.mean(first_chunk_times)
    avg_total = np.mean(total_times)
    p50_first = np.percentile(first_chunk_times, 50)
    p95_first = np.percentile(first_chunk_times, 95)

    print(f"\n  First chunk - Avg: {avg_first:.0f}ms | P50: {p50_first:.0f}ms | P95: {p95_first:.0f}ms")
    print(f"  Total gen   - Avg: {avg_total:.0f}ms")

    print("\n  --- Non-streaming mode (generate) ---")
    gen_times = []
    gen_durations = []

    for text in test_texts:
        start = time.time()
        audio = engine.generate(text, use_context=True)
        elapsed = (time.time() - start) * 1000
        duration = len(audio) / 24000

        gen_times.append(elapsed)
        gen_durations.append(duration)
        print(f"  '{text[:35]}' -> {elapsed:.0f}ms, {duration:.1f}s audio")

    avg_gen = np.mean(gen_times)
    print(f"\n  Non-streaming avg: {avg_gen:.0f}ms")

    print(f"\n  IMPROVEMENT: First audio {avg_gen - avg_first:.0f}ms sooner with streaming")
    print(f"  Streaming first chunk: {avg_first:.0f}ms vs Non-streaming: {avg_gen:.0f}ms")

    del engine
    torch.cuda.empty_cache()
    return avg_first, avg_total


def benchmark_full_streaming_pipeline():
    """Benchmark full pipeline with streaming TTS."""
    print("\n" + "=" * 60)
    print("BENCHMARK: FULL PIPELINE WITH STREAMING TTS")
    print("STT -> LLM -> TTS (streaming, first chunk)")
    print("=" * 60)

    from maya.engine.stt_faster import FasterSTTEngine
    from maya.engine.llm_optimized import OptimizedLLMEngine
    from maya.engine.tts_compiled import CompiledTTSEngine

    stt = FasterSTTEngine()
    llm = OptimizedLLMEngine()
    tts = CompiledTTSEngine()

    print("  Initializing all engines...")
    start = time.time()
    stt.initialize()
    llm.initialize()
    tts.initialize()
    print(f"  All engines ready in {time.time()-start:.1f}s")

    # Generate realistic test audio
    print("  Generating test audio for STT...")
    test_audio = tts.generate("hi there, how are you doing?", use_context=False)
    print(f"  Test audio: {len(test_audio)/24000:.1f}s")

    print("\n  Running full streaming pipeline (5 iterations):")
    first_audio_times = []
    total_times = []
    stt_times = []
    llm_times = []
    tts_first_times = []

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

        # TTS (streaming - measure first chunk)
        tts_start = time.time()
        first_chunk_time = None
        all_chunks = []

        for chunk in tts.generate_stream(response, use_context=True):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - tts_start) * 1000
            all_chunks.append(chunk)

        tts_total = (time.time() - tts_start) * 1000

        # First audio = STT + LLM + TTS first chunk
        first_audio = stt_elapsed + llm_elapsed + first_chunk_time
        total_elapsed = (time.time() - total_start) * 1000

        total_samples = sum(len(c) for c in all_chunks)
        audio_dur = total_samples / 24000

        stt_times.append(stt_elapsed)
        llm_times.append(llm_elapsed)
        tts_first_times.append(first_chunk_time)
        first_audio_times.append(first_audio)
        total_times.append(total_elapsed)

        print(f"\n  Run {i+1}:")
        print(f"    STT: {stt_elapsed:.0f}ms -> '{transcript[:30]}'")
        print(f"    LLM: {llm_elapsed:.0f}ms -> '{response[:30]}'")
        print(f"    TTS first chunk: {first_chunk_time:.0f}ms")
        print(f"    FIRST AUDIO: {first_audio:.0f}ms (STT+LLM+first_chunk)")
        print(f"    TTS total: {tts_total:.0f}ms -> {audio_dur:.1f}s audio, {len(all_chunks)} chunks")
        print(f"    TOTAL: {total_elapsed:.0f}ms")

    print("\n" + "=" * 60)
    print("STREAMING PIPELINE SUMMARY")
    print("=" * 60)

    print(f"  STT          avg: {np.mean(stt_times):>6.0f}ms")
    print(f"  LLM          avg: {np.mean(llm_times):>6.0f}ms")
    print(f"  TTS 1st chunk avg: {np.mean(tts_first_times):>5.0f}ms")
    print(f"  FIRST AUDIO   avg: {np.mean(first_audio_times):>5.0f}ms | P50: {np.percentile(first_audio_times, 50):>5.0f}ms | P95: {np.percentile(first_audio_times, 95):>5.0f}ms")
    print(f"  TOTAL         avg: {np.mean(total_times):>5.0f}ms")

    target = 800
    avg_first = np.mean(first_audio_times)
    print(f"\n  TARGET (first audio): {target}ms -> {'PASS' if avg_first <= target else 'MISS'} (avg {avg_first:.0f}ms)")

    previous_baseline = 1346
    improvement = previous_baseline - avg_first
    print(f"  IMPROVEMENT: {improvement:.0f}ms faster than previous {previous_baseline}ms baseline")
    print(f"  Reduction: {improvement/previous_baseline*100:.0f}%")

    del stt, llm, tts
    torch.cuda.empty_cache()
    return avg_first


def main():
    print("=" * 60)
    print("MAYA STREAMING PIPELINE BENCHMARK")
    print("=" * 60)

    gpu_count = torch.cuda.device_count()
    print(f"\nGPUs: {gpu_count}")
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({vram:.1f} GB)")

    # Individual TTS streaming benchmark
    avg_first_chunk, avg_total = benchmark_streaming_tts()

    # Full pipeline with streaming
    avg_first_audio = benchmark_full_streaming_pipeline()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\n  TTS first chunk:     {avg_first_chunk:>6.0f}ms")
    print(f"  TTS total gen:       {avg_total:>6.0f}ms")
    print(f"  Pipeline first audio: {avg_first_audio:>5.0f}ms")
    print(f"\n  Previous (non-streaming): ~1346ms")
    print(f"  Current (streaming):       ~{avg_first_audio:.0f}ms")
    print(f"  Improvement:               ~{1346-avg_first_audio:.0f}ms ({(1346-avg_first_audio)/1346*100:.0f}%)")


if __name__ == "__main__":
    main()
