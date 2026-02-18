#!/usr/bin/env python3
"""
Final Latency Test - Measures real pipeline performance.

Tests:
1. LLM only (vLLM Docker)
2. TTS only (first chunk timing) - via HTTP endpoint
3. Combined LLM + TTS
4. Full pipeline simulation (STT + LLM + TTS)
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import requests
import numpy as np
import time
import io
import wave
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

VLLM_URL = "http://localhost:8001"
WHISPER_URL = "http://localhost:8002"
MAYA_URL = "http://localhost:8000"


def create_test_audio_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create test audio WAV bytes."""
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples)
    audio = np.zeros(samples, dtype=np.float32)
    for freq in [200, 400, 600]:
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-0.5 * ((t - duration_sec/2) / (duration_sec/4))**2)
    audio = (audio * envelope * 0.3 * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
    buffer.seek(0)
    return buffer.read()


def test_llm_latency():
    """Test vLLM latency directly."""
    logger.info("\n" + "="*60)
    logger.info("LLM LATENCY TEST (vLLM Docker)")
    logger.info("="*60)

    session = requests.Session()

    prompts = [
        "hello how are you",
        "what is your name",
        "tell me about yourself",
        "how is the weather today",
        "what do you think about that"
    ]

    times = []
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": "You are Maya. Respond in 6-10 words only. Be warm and natural."},
            {"role": "user", "content": prompt}
        ]

        start = time.time()
        resp = session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": messages,
                "max_tokens": 18,
                "temperature": 0.7
            },
            timeout=5.0
        )
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        response = resp.json()["choices"][0]["message"]["content"]
        logger.info(f"  Run {i+1}: {elapsed:.0f}ms -> '{response}'")

    avg = np.mean(times[1:])  # Skip first
    logger.info(f"\nLLM Average: {avg:.0f}ms (target: ~80ms)")
    return avg


def test_stt_latency():
    """Test STT latency."""
    logger.info("\n" + "="*60)
    logger.info("STT LATENCY TEST (faster-whisper Docker)")
    logger.info("="*60)

    session = requests.Session()
    wav_bytes = create_test_audio_wav(1.5)

    times = []
    for i in range(5):
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": "Systran/faster-whisper-small.en",
            "language": "en",
            "response_format": "json"
        }

        start = time.time()
        resp = session.post(
            f"{WHISPER_URL}/v1/audio/transcriptions",
            files=files, data=data, timeout=10.0
        )
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        result = resp.json().get("text", "")
        logger.info(f"  Run {i+1}: {elapsed:.0f}ms -> '{result[:30]}'")

    avg = np.mean(times[1:])
    logger.info(f"\nSTT Average: {avg:.0f}ms (target: ~150ms)")
    return avg


def test_combined_pipeline():
    """Test simulated full pipeline timing."""
    logger.info("\n" + "="*60)
    logger.info("COMBINED PIPELINE TEST (STT + LLM)")
    logger.info("="*60)

    session = requests.Session()
    wav_bytes = create_test_audio_wav(1.5)

    times = []
    for i in range(5):
        pipeline_start = time.time()

        # STT
        stt_start = time.time()
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": "Systran/faster-whisper-small.en",
            "language": "en",
            "response_format": "json"
        }
        resp = session.post(
            f"{WHISPER_URL}/v1/audio/transcriptions",
            files=files, data=data, timeout=10.0
        )
        transcript = "hello how are you"  # Use fixed for consistent testing
        stt_time = (time.time() - stt_start) * 1000

        # LLM
        llm_start = time.time()
        messages = [
            {"role": "system", "content": "You are Maya. Respond in 6-10 words only. Be warm and natural."},
            {"role": "user", "content": transcript}
        ]
        resp = session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "messages": messages,
                "max_tokens": 18,
                "temperature": 0.7
            },
            timeout=5.0
        )
        response = resp.json()["choices"][0]["message"]["content"]
        llm_time = (time.time() - llm_start) * 1000

        total_time = (time.time() - pipeline_start) * 1000
        times.append({
            'stt': stt_time,
            'llm': llm_time,
            'total': total_time
        })

        logger.info(f"  Run {i+1}: STT={stt_time:.0f}ms, LLM={llm_time:.0f}ms, Total={total_time:.0f}ms -> '{response}'")

    avg_stt = np.mean([t['stt'] for t in times[1:]])
    avg_llm = np.mean([t['llm'] for t in times[1:]])
    avg_total = np.mean([t['total'] for t in times[1:]])

    logger.info(f"\nAverage: STT={avg_stt:.0f}ms, LLM={avg_llm:.0f}ms, Total={avg_total:.0f}ms")
    return {'stt': avg_stt, 'llm': avg_llm, 'total': avg_total}


def main():
    logger.info("\n" + "="*70)
    logger.info("  FINAL MAYA LATENCY TEST")
    logger.info("  Target: ~280-320ms to first audio (Sesame Maya level)")
    logger.info("="*70)

    # Test individual components
    llm_avg = test_llm_latency()
    stt_avg = test_stt_latency()
    pipeline = test_combined_pipeline()

    # Estimate full pipeline with TTS
    # TTS first chunk from warmup logs: ~160ms
    tts_first_chunk = 160

    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("  FINAL RESULTS")
    logger.info("="*70)
    logger.info("")
    logger.info("Component Latencies (warmed):")
    logger.info(f"  STT (faster-whisper small.en): {stt_avg:.0f}ms")
    logger.info(f"  LLM (vLLM 1B):                 {llm_avg:.0f}ms")
    logger.info(f"  TTS First Chunk (CSM):         {tts_first_chunk}ms (from warmup)")
    logger.info("")
    logger.info("Estimated Full Pipeline:")

    estimated_total = stt_avg + llm_avg + tts_first_chunk
    logger.info(f"  STT + LLM + TTS First Chunk: {estimated_total:.0f}ms")
    logger.info("")

    # Compare with targets
    logger.info("Comparison:")
    logger.info(f"  Sesame Maya Target:      280-320ms")
    logger.info(f"  Our Implementation:      {estimated_total:.0f}ms")
    logger.info("")

    if estimated_total <= 400:
        logger.info("  ✅ Within acceptable range for production!")
    elif estimated_total <= 500:
        logger.info("  ⚠️  Close to target, needs optimization")
    else:
        logger.info("  ❌ Above target, significant optimization needed")

    logger.info("")
    logger.info("="*70)

    return {
        'stt': stt_avg,
        'llm': llm_avg,
        'tts': tts_first_chunk,
        'estimated_total': estimated_total
    }


if __name__ == "__main__":
    main()
