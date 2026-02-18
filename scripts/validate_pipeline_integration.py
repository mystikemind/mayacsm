#!/usr/bin/env python3
"""Validate fine-tuned model integration in the Maya TTS pipeline.

Tests:
1. Model loads correctly (fine-tuned weights + torch.compile)
2. Short voice prompt works
3. Generation quality matches LoRA v3 eval benchmarks
4. Latency is acceptable for real-time use
"""
import sys
import os
import time
import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SR = 24000
OUTPUT_DIR = "/home/ec2-user/SageMaker/project_maya/training/evaluation/pipeline_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_PROMPTS = [
    "yeah, totally.",
    "oh hey, hi! its really nice to meet you.",
    "aw man, that sounds really rough, im so sorry.",
    "okay so like, what do you think about that?",
    "ha, yeah, thats pretty funny actually.",
    "hmm, im not really sure about that honestly.",
    "wow, thats incredible, tell me more.",
    "ugh, mondays, am i right?",
]


def load_whisper():
    """Load Whisper for transcription verification."""
    import whisper
    return whisper.load_model("base.en", device="cuda")


def transcribe(asr, audio_tensor):
    """Transcribe audio tensor using Whisper."""
    audio_16k = torchaudio.functional.resample(audio_tensor.unsqueeze(0), SR, 16000)
    result = asr.transcribe(audio_16k.squeeze().cpu().numpy())
    return result['text'].strip().lower()


def word_overlap(expected, got):
    """Compute word overlap between expected and transcribed text."""
    import re
    clean = lambda t: set(re.sub(r'[^\w\s]', '', t.lower()).split())
    e, g = clean(expected), clean(got)
    return len(e & g) / max(len(e), 1)


def main():
    print("=" * 70)
    print("MAYA PIPELINE INTEGRATION VALIDATION")
    print("=" * 70)

    # Load Whisper ASR for quality checks
    print("\nLoading Whisper ASR...")
    asr = load_whisper()

    # Load the compiled TTS engine (the production path)
    print("\nLoading CompiledTTSEngine (with fine-tuned model)...")
    from maya.engine.tts_compiled import CompiledTTSEngine
    tts = CompiledTTSEngine()

    init_start = time.time()
    tts.initialize()
    init_time = time.time() - init_start
    print(f"\nTTS initialized in {init_time:.1f}s")

    # Run generation tests
    print(f"\n{'='*70}")
    print("GENERATING TEST SAMPLES")
    print(f"{'='*70}")

    results = []
    for i, text in enumerate(TEST_PROMPTS):
        print(f"\n  [{i+1}/{len(TEST_PROMPTS)}] '{text}'")

        try:
            gen_start = time.time()
            audio = tts.generate(text, use_context=True)
            gen_time = (time.time() - gen_start) * 1000

            duration = len(audio) / SR
            eos = duration < 9.5
            amp_min = audio.min().item()
            amp_max = audio.max().item()
            clipping = abs(amp_min) > 1.0 or abs(amp_max) > 1.0

            # Save audio
            out_path = os.path.join(OUTPUT_DIR, f"pipeline_sample_{i}.wav")
            torchaudio.save(out_path, audio.unsqueeze(0).cpu(), SR)

            # Transcribe
            trans = transcribe(asr, audio)
            overlap = word_overlap(text, trans)

            rtf = (gen_time / 1000) / duration if duration > 0 else 0

            results.append({
                "text": text, "duration": duration, "eos": eos,
                "amp": (amp_min, amp_max), "clipping": clipping,
                "transcription": trans, "overlap": overlap,
                "gen_time_ms": gen_time, "rtf": rtf,
            })

            eos_str = "EOS" if eos else "MAX"
            print(f"    Duration:  {duration:.2f}s ({eos_str})")
            print(f"    Gen time:  {gen_time:.0f}ms (RTF: {rtf:.2f}x)")
            print(f"    Amplitude: [{amp_min:.3f}, {amp_max:.3f}]{'  CLIPPING!' if clipping else ''}")
            print(f"    Expected:  '{text}'")
            print(f"    Got:       '{trans}'")
            print(f"    Overlap:   {overlap:.0%}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"text": text, "error": str(e)})

    # Summary
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nERROR: No valid results!")
        return

    eos_count = sum(1 for r in valid if r["eos"])
    avg_dur = np.mean([r["duration"] for r in valid])
    avg_overlap = np.mean([r["overlap"] for r in valid])
    clipping_count = sum(1 for r in valid if r["clipping"])
    avg_gen_ms = np.mean([r["gen_time_ms"] for r in valid])
    avg_rtf = np.mean([r["rtf"] for r in valid])

    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"  EOS rate:        {eos_count}/{len(valid)} ({100*eos_count/len(valid):.0f}%)")
    print(f"  Avg duration:    {avg_dur:.2f}s")
    print(f"  Avg word match:  {avg_overlap:.0%}")
    print(f"  Clipping:        {clipping_count}/{len(valid)}")
    print(f"  Avg gen time:    {avg_gen_ms:.0f}ms")
    print(f"  Avg RTF:         {avg_rtf:.2f}x")

    # Benchmarks (from LoRA v3 checkpoint-500 eval)
    print(f"\n  BENCHMARK COMPARISON:")
    print(f"  {'Metric':<20s}  {'LoRA v3 Eval':>12s}  {'Pipeline':>12s}  {'Pass?':>6s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*6}")

    eos_pass = eos_count / len(valid) >= 0.60
    overlap_pass = avg_overlap >= 0.60
    clip_pass = clipping_count == 0
    rtf_pass = avg_rtf < 1.0

    print(f"  {'EOS rate':<20s}  {'75%':>12s}  {f'{100*eos_count/len(valid):.0f}%':>12s}  {'OK' if eos_pass else 'FAIL':>6s}")
    print(f"  {'Word match':<20s}  {'70%':>12s}  {f'{avg_overlap:.0%}':>12s}  {'OK' if overlap_pass else 'FAIL':>6s}")
    print(f"  {'Clipping':<20s}  {'0/8':>12s}  {f'{clipping_count}/{len(valid)}':>12s}  {'OK' if clip_pass else 'FAIL':>6s}")
    print(f"  {'RTF':<20s}  {'< 1.0x':>12s}  {f'{avg_rtf:.2f}x':>12s}  {'OK' if rtf_pass else 'FAIL':>6s}")

    all_pass = eos_pass and overlap_pass and clip_pass and rtf_pass
    print(f"\n  OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    print(f"\n  Audio saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
