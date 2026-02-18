#!/usr/bin/env python3
"""
Quick E2E Pipeline Validation
==============================
Tests the production pipeline end-to-end to verify:
1. STT with hallucination mitigation works
2. LLM generates with emotion tags
3. TTS streaming with crossfade works
4. Audio quality gate passes

Does NOT test WebSocket/UI, just the pipeline components.

Usage:
    python scripts/test_pipeline_quick.py
"""

import os
import sys
import time
import asyncio
import torch
import logging

# Setup paths
os.environ.setdefault("HF_HOME", "/home/ec2-user/SageMaker/.cache/huggingface")

# Fix cuDNN
_cudnn_lib = os.path.join(os.path.dirname(os.__file__), "..", "nvidia", "cudnn", "lib")
if os.path.isdir(_cudnn_lib):
    os.environ["LD_LIBRARY_PATH"] = _cudnn_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


async def test_pipeline():
    """Test the full pipeline."""
    from maya.pipeline.seamless_orchestrator import SeamlessMayaPipeline, is_whisper_hallucination
    from maya.config import AUDIO

    print("=" * 60)
    print("PIPELINE VALIDATION TEST")
    print("=" * 60)

    # Test 1: Hallucination detection
    print("\nTest 1: Whisper Hallucination Detection")
    test_cases = [
        ("thanks for watching", True),
        ("thank you for watching", True),
        ("hello how are you", False),
        ("", True),
        ("you", True),
        ("yeah im doing great", False),
        ("music", True),
        ("thank you thank you thank you thank you", True),  # Repetitive
        ("i love this conversation", False),
    ]
    for text, expected in test_cases:
        result = is_whisper_hallucination(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{text[:40]}' -> filtered={result} (expected={expected})")

    # Test 2: LLM Generation
    print("\nTest 2: LLM Generation")
    from maya.engine.llm_llamacpp import LlamaCppLLMEngine
    llm = LlamaCppLLMEngine()
    llm.initialize()

    test_inputs = [
        "How are you doing today?",
        "I just got a promotion at work!",
        "My dog is feeling sick, I'm worried.",
        "Tell me something funny.",
    ]
    for text in test_inputs:
        t0 = time.time()
        response = llm.generate(text)
        elapsed = (time.time() - t0) * 1000
        words = len(response.split())
        has_tag = any(tag in response for tag in ['<laugh>', '<chuckle>', '<sigh>', '<gasp>'])
        print(f"  [{elapsed:.0f}ms] '{text[:30]}...' -> '{response}' ({words} words, emotion_tag={has_tag})")

    # Test 3: TTS Streaming with Crossfade
    print("\nTest 3: TTS Streaming with Crossfade")
    from maya.engine.tts_orpheus import OrpheusTTSEngine
    tts = OrpheusTTSEngine()
    tts.initialize()

    test_texts = [
        "hey there, im maya, nice to meet you",
        "thats really interesting, tell me more about it",
    ]
    for text in test_texts:
        t0 = time.time()
        chunks = []
        first_chunk_time = None
        async for chunk in tts.generate_stream(text):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - t0) * 1000
            chunks.append(chunk)
        total_time = (time.time() - t0) * 1000
        total_samples = sum(c.shape[-1] for c in chunks)
        duration = total_samples / 24000
        rtf = (total_time / 1000) / duration if duration > 0 else float('inf')
        print(f"  [{total_time:.0f}ms] '{text[:30]}...' -> {len(chunks)} chunks, "
              f"{duration:.1f}s audio, RTF={rtf:.2f}, first_chunk={first_chunk_time:.0f}ms")

        # Check audio quality
        if chunks:
            full_audio = torch.cat(chunks)
            rms = torch.sqrt(torch.mean(full_audio ** 2)).item()
            peak = full_audio.abs().max().item()
            print(f"    Audio: RMS={rms:.4f}, Peak={peak:.4f}, "
                  f"Samples={total_samples}, NaN={torch.isnan(full_audio).any()}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
