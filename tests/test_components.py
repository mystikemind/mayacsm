#!/usr/bin/env python3
"""
Test individual Maya components.
"""

import torch
import asyncio
import time
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')


def test_vad():
    """Test VAD engine."""
    print("\n" + "="*50)
    print("Testing VAD Engine")
    print("="*50)

    from maya.engine import VADEngine

    vad = VADEngine()
    vad.initialize()

    # Test with silence
    silence = torch.zeros(2400)  # 100ms at 24kHz
    result = vad.process(silence)
    print(f"Silence: is_speech={result.is_speech}, confidence={result.confidence:.3f}")

    # Test with noise (simulates speech)
    speech = torch.randn(2400) * 0.5
    result = vad.process(speech)
    print(f"Noise: is_speech={result.is_speech}, confidence={result.confidence:.3f}")

    print("VAD: OK ✓")


def test_stt():
    """Test STT engine."""
    print("\n" + "="*50)
    print("Testing STT Engine")
    print("="*50)

    from maya.engine import STTEngine

    stt = STTEngine()
    stt.initialize()

    # Test with silence (should return empty or minimal)
    silence = torch.zeros(24000 * 2)  # 2 seconds
    result = stt.transcribe(silence)
    print(f"Silence transcript: '{result}'")

    print(f"STT latency: {stt.average_latency_ms:.0f}ms")
    print("STT: OK ✓")


def test_llm():
    """Test LLM engine."""
    print("\n" + "="*50)
    print("Testing LLM Engine")
    print("="*50)

    from maya.engine import LLMEngine

    llm = LLMEngine()
    llm.initialize()

    # Test generation
    start = time.time()
    response = llm.generate("Hello, how are you?")
    elapsed = (time.time() - start) * 1000

    print(f"Response: '{response}'")
    print(f"LLM latency: {elapsed:.0f}ms")
    print("LLM: OK ✓")


def test_tts():
    """Test TTS engine."""
    print("\n" + "="*50)
    print("Testing TTS Engine")
    print("="*50)

    from maya.engine import TTSEngine
    import torchaudio

    tts = TTSEngine()
    tts.initialize()

    # Test generation
    start = time.time()
    audio = tts.generate("Hello! I'm Maya, your voice assistant.")
    elapsed = (time.time() - start) * 1000

    duration = len(audio) / 24000
    print(f"Generated {duration:.2f}s audio in {elapsed:.0f}ms")
    print(f"RTF: {elapsed/1000/duration:.2f}x")

    # Save test output
    output_path = "/home/ec2-user/SageMaker/project_maya/tests/test_output.wav"
    torchaudio.save(output_path, audio.unsqueeze(0).cpu(), 24000)
    print(f"Saved to {output_path}")

    print("TTS: OK ✓")


def test_filler():
    """Test filler system."""
    print("\n" + "="*50)
    print("Testing Filler System")
    print("="*50)

    from maya.conversation import FillerSystem
    from maya.engine import TTSEngine

    async def run():
        tts = TTSEngine()
        tts.initialize()

        fillers = FillerSystem()
        await fillers.initialize(tts_engine=tts)

        # Get a thinking filler
        audio, text = fillers.get_thinking_filler()
        duration = len(audio) / 24000
        print(f"Thinking filler: '{text}' ({duration:.1f}s)")

        # Get a backchannel
        audio, text = fillers.get_backchannel()
        duration = len(audio) / 24000
        print(f"Backchannel: '{text}' ({duration:.1f}s)")

        print(f"Total fillers: {fillers.filler_count}")

    asyncio.run(run())
    print("Fillers: OK ✓")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MAYA COMPONENT TESTS")
    print("="*60)

    tests = [
        ("VAD", test_vad),
        ("STT", test_stt),
        ("TTS", test_tts),  # TTS before LLM to save VRAM
        ("LLM", test_llm),
        ("Filler", test_filler),
    ]

    failed = []

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

        # Clear CUDA cache between tests
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    else:
        print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    main()
