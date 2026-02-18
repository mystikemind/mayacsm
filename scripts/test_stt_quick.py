#!/usr/bin/env python3
"""Quick STT latency test."""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import numpy as np
import time
import requests
import io
import wave

WHISPER_URL = "http://localhost:8002"

def create_test_audio(duration_sec: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Create test audio."""
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples)
    audio = np.zeros(samples, dtype=np.float32)
    for freq in [200, 400, 600]:
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-0.5 * ((t - duration_sec/2) / (duration_sec/4))**2)
    audio = audio * envelope * 0.3
    return audio

def audio_to_wav_bytes(audio_np: np.ndarray, sample_rate: int = 16000) -> bytes:
    audio_int16 = (audio_np * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    return buffer.read()

def test_stt():
    print("="*60)
    print("QUICK STT LATENCY TEST")
    print("="*60)

    session = requests.Session()

    # Check health
    resp = session.get(f"{WHISPER_URL}/health", timeout=2.0)
    print(f"Server health: {resp.status_code}")

    # Test with different audio lengths
    for duration in [0.5, 1.0, 1.5, 2.0]:
        audio = create_test_audio(duration)
        wav_bytes = audio_to_wav_bytes(audio)

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": "Systran/faster-whisper-tiny.en",
            "language": "en",
            "response_format": "json"
        }

        times = []
        for i in range(3):
            start = time.time()
            resp = session.post(
                f"{WHISPER_URL}/v1/audio/transcriptions",
                files=files, data=data, timeout=10.0
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            result = resp.json().get("text", "")

        avg = np.mean(times)
        print(f"  {duration}s audio: {avg:.0f}ms avg -> '{result[:30]}...'")

    print("")
    print("Note: These are synthetic tones, so Whisper may hallucinate.")
    print("Real speech would give different results.")

if __name__ == "__main__":
    test_stt()
