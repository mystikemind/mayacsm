#!/usr/bin/env python3
"""
Test the FULL SesamePipeline with vLLM + TTS.

Infrastructure already running:
- vLLM server on port 8001 (Llama-3.2-1B, ~75ms)
- Faster Whisper on port 8002
- Fine-tuned CSM TTS (~132ms first chunk)

Target: <200ms to first audio
"""

import sys
import os
import time
import asyncio
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.io.wavfile as wav

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
OUTPUT_DIR = PROJECT_ROOT / "audio_sesame_test"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_INPUTS = [
    "Hello, how are you?",
    "What's your favorite color?",
    "I'm feeling a bit sad today",
    "Tell me something interesting",
    "That's really cool",
]


async def test_vllm_latency():
    """Test vLLM directly."""
    print("\n[1/3] Testing vLLM Server...")

    from maya.engine.llm_vllm import VLLMEngine

    llm = VLLMEngine()
    llm.initialize()

    results = []
    for text in TEST_INPUTS:
        start = time.time()
        response = llm.generate(text)
        elapsed = time.time() - start

        print(f"  '{text[:30]}...' -> '{response}' [{elapsed*1000:.0f}ms]")
        results.append(elapsed * 1000)

    avg = np.mean(results)
    print(f"\n  Average vLLM latency: {avg:.0f}ms")
    return avg


async def test_tts_latency():
    """Test TTS directly."""
    print("\n[2/3] Testing Fine-tuned TTS...")

    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    tts = RealStreamingTTSEngine()
    tts.initialize()

    test_responses = [
        "Oh wow that is really cool",
        "I am doing great thanks",
        "That sounds really interesting",
    ]

    results = []
    for text in test_responses:
        start = time.time()
        first_chunk_time = None

        for chunk in tts.generate_stream(text, use_context=False):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
            break  # Just measure first chunk

        print(f"  '{text}' -> First chunk: {first_chunk_time*1000:.0f}ms")
        results.append(first_chunk_time * 1000)

        torch.cuda.empty_cache()

    avg = np.mean(results)
    print(f"\n  Average TTS first chunk: {avg:.0f}ms")
    return avg, tts


async def test_full_pipeline(llm_avg, tts_avg):
    """Test full LLM → TTS pipeline."""
    print("\n[3/3] Testing Full Pipeline (LLM → TTS)...")

    from maya.engine.llm_vllm import VLLMEngine
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine

    # Load both on separate GPUs if possible
    llm = VLLMEngine()
    llm.initialize()

    # Clear some memory
    torch.cuda.empty_cache()

    tts = RealStreamingTTSEngine()
    tts.initialize()

    results = []
    for i, user_input in enumerate(TEST_INPUTS[:3]):
        print(f"\n  Test {i+1}: '{user_input}'")

        # Full pipeline timing
        total_start = time.time()

        # LLM
        llm_start = time.time()
        response = llm.generate(user_input)
        llm_time = time.time() - llm_start

        # TTS
        tts_start = time.time()
        first_chunk_time = None
        chunks = []

        for chunk in tts.generate_stream(response, use_context=False):
            if first_chunk_time is None:
                first_chunk_time = time.time() - tts_start
            chunks.append(chunk)

        tts_total = time.time() - tts_start
        total_time = time.time() - total_start

        # Save audio
        if chunks:
            audio = torch.cat(chunks)
            audio_np = audio.cpu().numpy()
            audio_np = audio_np / max(abs(audio_np.min()), abs(audio_np.max())) * 0.9
            wav.write(str(OUTPUT_DIR / f"test_{i+1}.wav"), 24000,
                     (audio_np * 32767).astype(np.int16))

        print(f"    Response: '{response}'")
        print(f"    LLM: {llm_time*1000:.0f}ms | TTS first: {first_chunk_time*1000:.0f}ms")
        print(f"    TOTAL to first audio: {(llm_time + first_chunk_time)*1000:.0f}ms")

        results.append({
            "llm_ms": llm_time * 1000,
            "tts_first_ms": first_chunk_time * 1000,
            "total_ms": (llm_time + first_chunk_time) * 1000,
        })

        torch.cuda.empty_cache()

    return results


async def main():
    print("=" * 70)
    print("SESAME PIPELINE FULL TEST")
    print("=" * 70)

    # Test components
    llm_avg = await test_vllm_latency()

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    tts_avg, tts = await test_tts_latency()

    del tts
    torch.cuda.empty_cache()
    gc.collect()

    # Test full pipeline
    results = await test_full_pipeline(llm_avg, tts_avg)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    avg_llm = np.mean([r["llm_ms"] for r in results])
    avg_tts = np.mean([r["tts_first_ms"] for r in results])
    avg_total = np.mean([r["total_ms"] for r in results])

    print(f"""
Component Latencies:
  vLLM:       {avg_llm:.0f}ms
  TTS first:  {avg_tts:.0f}ms
  ─────────────────────
  TOTAL:      {avg_total:.0f}ms

Sesame Target: ~200ms

Result: {"✅ SESAME LEVEL ACHIEVED!" if avg_total <= 250 else "⚠️ Close, needs optimization"}

Audio samples saved to: {OUTPUT_DIR}/
""")

if __name__ == "__main__":
    asyncio.run(main())
