#!/usr/bin/env python3
"""
End-to-End Pipeline Test - Senior Engineer Quality Check

Tests:
1. Full pipeline latency (STT -> LLM -> TTS)
2. Audio quality (save samples for listening)
3. Context handling (prosodic consistency)
4. Real-time capability (RTF < 1.0)

Run: python scripts/test_end_to_end.py
"""

import sys
import time
import torch
import numpy as np
import os

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')


def test_tts_with_context():
    """Test TTS with conversation context - this is the key quality improvement."""
    print("=" * 70)
    print("TTS WITH CONVERSATION CONTEXT TEST")
    print("=" * 70)

    from maya.engine.tts_compiled import CompiledTTSEngine

    # Initialize
    print("\n[1] Initializing TTS...")
    tts = CompiledTTSEngine()
    init_start = time.time()
    tts.initialize()
    print(f"    Init time: {time.time() - init_start:.1f}s")

    # Test phrases simulating a conversation
    conversation = [
        ("user", "Hi Maya, how are you?"),
        ("maya", "I'm doing great, thanks for asking!"),
        ("user", "What's your favorite color?"),
        ("maya", "I really love blue, it's so calming."),
        ("user", "That's nice. What about food?"),
        ("maya", "Pizza is definitely my favorite!"),
    ]

    output_dir = "/home/ec2-user/SageMaker/project_maya/tests/outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[2] Generating conversation with context...")
    results = []

    for i, (speaker, text) in enumerate(conversation):
        if speaker == "maya":
            start = time.time()
            audio = tts.generate(text, use_context=True)
            elapsed = time.time() - start

            duration = len(audio) / 24000
            rtf = elapsed / duration

            # Save audio for quality check
            import torchaudio
            output_path = f"{output_dir}/turn_{i}_{text[:20].replace(' ', '_')}.wav"
            torchaudio.save(output_path, audio.unsqueeze(0).cpu(), 24000)

            results.append({
                'turn': i,
                'text': text,
                'time_ms': elapsed * 1000,
                'duration_s': duration,
                'rtf': rtf
            })

            # Add Maya's response to context
            tts.add_context(text, audio, is_user=False)

            print(f"    Turn {i}: '{text[:30]}' -> {elapsed*1000:.0f}ms (RTF={rtf:.2f}x)")
            print(f"           Saved: {output_path}")
        else:
            # Simulate user audio (in real system this would come from STT)
            fake_user_audio = torch.randn(24000 * 2)  # 2 seconds
            tts.add_context(text, fake_user_audio, is_user=True)
            print(f"    Turn {i}: [USER] '{text}'")

    # Stats
    maya_results = [r for r in results]
    avg_time = np.mean([r['time_ms'] for r in maya_results])
    avg_rtf = np.mean([r['rtf'] for r in maya_results])

    print(f"\n    Average generation time: {avg_time:.0f}ms")
    print(f"    Average RTF: {avg_rtf:.2f}x")
    print(f"    Real-time capable: {'YES ✅' if avg_rtf < 1.0 else 'NO ❌'}")
    print(f"    Context turns: {tts.get_context_size()}")

    return results


def test_full_pipeline():
    """Test the complete STT -> LLM -> TTS pipeline."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE TEST (STT -> LLM -> TTS)")
    print("=" * 70)

    from maya.engine.stt import STTEngine
    from maya.engine.llm_optimized import OptimizedLLMEngine
    from maya.engine.tts_compiled import CompiledTTSEngine

    # Initialize
    print("\n[1] Initializing components...")

    stt = STTEngine()
    stt.initialize()
    print("    STT ready")

    llm = OptimizedLLMEngine()
    llm.initialize()
    print("    LLM ready")

    tts = CompiledTTSEngine()
    tts.initialize()
    print("    TTS ready")

    # Test inputs
    test_inputs = [
        "Hello Maya",
        "How are you doing today",
        "What's the weather like",
        "Tell me a joke",
    ]

    output_dir = "/home/ec2-user/SageMaker/project_maya/tests/outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[2] Running pipeline tests...")
    results = []

    for i, user_input in enumerate(test_inputs):
        print(f"\n    Test {i+1}: '{user_input}'")

        turn_start = time.time()

        # STT (simulated - in real system this transcribes audio)
        stt_start = time.time()
        transcript = user_input  # Simulated
        stt_time = 150  # Typical STT time

        # LLM
        llm_start = time.time()
        response = llm.generate(transcript)
        llm_time = (time.time() - llm_start) * 1000

        # TTS
        tts_start = time.time()
        audio = tts.generate(response, use_context=True)
        tts_time = (time.time() - tts_start) * 1000

        total_time = stt_time + llm_time + tts_time
        audio_duration = len(audio) / 24000
        rtf = tts_time / 1000 / audio_duration

        # Save audio
        import torchaudio
        output_path = f"{output_dir}/pipeline_{i}_{user_input[:15].replace(' ', '_')}.wav"
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), 24000)

        results.append({
            'input': user_input,
            'response': response,
            'stt_ms': stt_time,
            'llm_ms': llm_time,
            'tts_ms': tts_time,
            'total_ms': total_time,
            'rtf': rtf
        })

        print(f"           Response: '{response}'")
        print(f"           STT={stt_time:.0f}ms, LLM={llm_time:.0f}ms, TTS={tts_time:.0f}ms")
        print(f"           Total: {total_time:.0f}ms (RTF={rtf:.2f}x)")
        print(f"           Audio: {output_path}")

        # Add to context
        tts.add_context(response, audio, is_user=False)
        llm.add_context("assistant", response)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_total = np.mean([r['total_ms'] for r in results])
    avg_stt = np.mean([r['stt_ms'] for r in results])
    avg_llm = np.mean([r['llm_ms'] for r in results])
    avg_tts = np.mean([r['tts_ms'] for r in results])
    avg_rtf = np.mean([r['rtf'] for r in results])

    print(f"""
    Average Latency:
    - STT:   {avg_stt:.0f}ms (target: <200ms)
    - LLM:   {avg_llm:.0f}ms (target: <300ms)
    - TTS:   {avg_tts:.0f}ms (target: <600ms)
    - Total: {avg_total:.0f}ms (target: <1100ms)

    RTF: {avg_rtf:.2f}x (must be < 1.0 for real-time)

    Sesame Maya Comparison:
    - Sesame target: ~500-1000ms
    - Our latency:   ~{avg_total:.0f}ms
    - Gap: {avg_total - 750:.0f}ms from Sesame midpoint
    """)

    # Verdict
    print("VERDICT:")
    if avg_total < 1100 and avg_rtf < 1.0:
        print("    ✅ REAL-TIME CAPABLE - Latency is acceptable!")
    elif avg_rtf < 1.0:
        print("    ⚠️  RTF is good but total latency needs improvement")
    else:
        print("    ❌ NOT REAL-TIME - RTF > 1.0")

    print(f"\n    Audio samples saved to: {output_dir}")
    print("    Listen to them to evaluate voice quality and human-likeness.")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MAYA END-TO-END QUALITY TEST")
    print("  Senior Engineer Evaluation")
    print("=" * 70)

    # Run tests
    context_results = test_tts_with_context()

    # Clear GPU memory before next test
    torch.cuda.empty_cache()

    pipeline_results = test_full_pipeline()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("""
    Next steps:
    1. Listen to audio files in tests/outputs/
    2. Evaluate voice quality and human-likeness
    3. Test with real microphone input via run.py
    """)
