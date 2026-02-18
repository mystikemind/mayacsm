#!/usr/bin/env python3
"""
DEFINITIVE QUALITY TEST - Find the Best Path to Sesame-Level Voice

This script tests ALL viable approaches to determine the best path forward:
1. Base CSM + Long Voice Prompt (158s) - Zero-shot approach
2. Base CSM + Short Human Voice Prompt (10s) - What worked before
3. keanteng/sesame-csm-elise - Pre-trained fine-tuned model
4. Orpheus-3B-ft - Alternative high-quality model

Output: Ranked comparison with audio samples and metrics
"""

import sys
import os
import time
import json
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
OUTPUT_DIR = PROJECT_ROOT / "audio_definitive_test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Test phrases covering different scenarios
TEST_PHRASES = [
    # Basic greetings
    ("greeting", "hi there how are you doing today"),
    # Emotional - happy
    ("happy", "oh wow thats amazing i love it so much"),
    # Emotional - sad
    ("sad", "oh no im so sorry to hear that"),
    # Thinking/filler sounds (problematic before)
    ("thinking", "hmm let me think about that"),
    # Question (was problematic)
    ("question", "wait what do you mean by that"),
    # Longer response
    ("longer", "yeah that makes a lot of sense actually i totally agree with you on that"),
    # Natural conversation
    ("natural", "you know what i was just thinking the same thing"),
]

print("=" * 70)
print("DEFINITIVE QUALITY TEST")
print("Testing all viable paths to Sesame-level voice quality")
print("=" * 70)


def calculate_audio_metrics(audio_np):
    """Calculate quality metrics for audio."""
    metrics = {}

    # Basic stats
    metrics["duration"] = len(audio_np) / 24000
    metrics["peak"] = float(np.abs(audio_np).max())
    metrics["rms"] = float(np.sqrt(np.mean(audio_np ** 2)))

    # Click detection
    diff = np.abs(np.diff(audio_np))
    metrics["clicks"] = int(np.sum(diff > 0.3))
    metrics["harsh_clicks"] = int(np.sum(diff > 0.5))

    # Silence ratio (potential gibberish indicator)
    silence_threshold = 0.01
    silence_samples = np.sum(np.abs(audio_np) < silence_threshold)
    metrics["silence_ratio"] = float(silence_samples / len(audio_np))

    return metrics


def test_base_csm_with_voice_prompt(voice_prompt_path: str, name: str):
    """Test base CSM with a voice prompt."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Voice Prompt: {voice_prompt_path}")
    print("=" * 60)

    from generator import Generator, Segment
    from models import Model

    # Load model
    print("Loading base CSM-1B...")
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device="cuda", dtype=torch.bfloat16)
    generator = Generator(model)

    # Load voice prompt
    vp_data = torch.load(voice_prompt_path, weights_only=False)

    if isinstance(vp_data, dict):
        vp_audio = vp_data.get("audio", torch.zeros(24000))
        vp_text = vp_data.get("text", "")[:500]  # Truncate if too long
        print(f"Voice prompt duration: {len(vp_audio)/24000:.1f}s")
    else:
        vp_audio = vp_data.audio if hasattr(vp_data, 'audio') else torch.zeros(24000)
        vp_text = vp_data.text[:500] if hasattr(vp_data, 'text') else ""

    # Create voice prompt segment
    # Need to tokenize the audio for CSM
    from moshi.models import loaders
    from huggingface_hub import hf_hub_download
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cuda")
    mimi.set_num_codebooks(32)
    mimi.eval()

    vp_audio_tensor = vp_audio.float().unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        vp_tokens = mimi.encode(vp_audio_tensor).squeeze(0)

    voice_prompt = Segment(
        text=vp_text,
        speaker=0,
        audio=vp_audio.cpu(),
        audio_tokens=vp_tokens.cpu(),
    )

    # Generate samples
    results = []
    output_subdir = OUTPUT_DIR / name.replace(" ", "_").lower()
    output_subdir.mkdir(exist_ok=True)

    for phrase_name, phrase_text in TEST_PHRASES:
        print(f"\n  Generating: '{phrase_text[:40]}...'")

        try:
            start = time.time()

            # Generate with voice prompt as context
            audio = generator.generate(
                text=phrase_text,
                speaker=0,
                context=[voice_prompt],
                max_audio_length_ms=8000,
                temperature=0.8,
                topk=50,
            )

            gen_time = time.time() - start
            audio_np = audio.cpu().float().numpy()

            # Calculate metrics
            metrics = calculate_audio_metrics(audio_np)
            metrics["generation_time"] = gen_time
            metrics["phrase"] = phrase_name
            metrics["text"] = phrase_text

            # Save audio
            output_path = output_subdir / f"{phrase_name}.wav"
            wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))

            print(f"    Duration: {metrics['duration']:.1f}s, Clicks: {metrics['clicks']}, Time: {gen_time:.1f}s")

            results.append(metrics)

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"phrase": phrase_name, "error": str(e)})

        torch.cuda.empty_cache()

    # Cleanup
    del model, generator, mimi
    torch.cuda.empty_cache()

    return results


def test_elise_model():
    """Test the pre-trained keanteng/sesame-csm-elise model."""
    print(f"\n{'='*60}")
    print("Testing: keanteng/sesame-csm-elise (Pre-trained)")
    print("=" * 60)

    try:
        from transformers import CsmForConditionalGeneration, AutoProcessor

        print("Loading keanteng/sesame-csm-elise...")
        model_id = "keanteng/sesame-csm-elise"

        processor = AutoProcessor.from_pretrained(model_id)
        model = CsmForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        results = []
        output_subdir = OUTPUT_DIR / "elise_pretrained"
        output_subdir.mkdir(exist_ok=True)

        for phrase_name, phrase_text in TEST_PHRASES:
            print(f"\n  Generating: '{phrase_text[:40]}...'")

            try:
                start = time.time()

                conversation = [
                    {"role": "0", "content": [{"type": "text", "text": phrase_text}]}
                ]

                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True,
                ).to("cuda")

                audio = model.generate(**inputs, output_audio=True)
                gen_time = time.time() - start

                audio_np = audio[0].to(torch.float32).cpu().numpy()

                # Normalize
                if audio_np.max() > 1.0:
                    audio_np = audio_np / 32768.0

                metrics = calculate_audio_metrics(audio_np)
                metrics["generation_time"] = gen_time
                metrics["phrase"] = phrase_name
                metrics["text"] = phrase_text

                output_path = output_subdir / f"{phrase_name}.wav"
                wav.write(str(output_path), 24000, (audio_np * 32767).astype(np.int16))

                print(f"    Duration: {metrics['duration']:.1f}s, Clicks: {metrics['clicks']}, Time: {gen_time:.1f}s")

                results.append(metrics)

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({"phrase": phrase_name, "error": str(e)})

            torch.cuda.empty_cache()

        del model, processor
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"  Failed to load Elise model: {e}")
        return [{"error": str(e)}]


def test_orpheus():
    """Test Orpheus-3B-ft model."""
    print(f"\n{'='*60}")
    print("Testing: Orpheus-3B-ft (Alternative)")
    print("=" * 60)

    try:
        # Check if orpheus is available
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading Orpheus-3B-ft...")
        model_id = "canopylabs/orpheus-3b-0.1-ft"

        # Orpheus uses a different architecture - need special handling
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        results = []
        output_subdir = OUTPUT_DIR / "orpheus_3b"
        output_subdir.mkdir(exist_ok=True)

        for phrase_name, phrase_text in TEST_PHRASES[:3]:  # Test fewer due to different architecture
            print(f"\n  Generating: '{phrase_text[:40]}...'")

            try:
                start = time.time()

                # Orpheus uses special tokens for speech synthesis
                prompt = f"<|audio|>{phrase_text}<|/audio|>"
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    temperature=0.7,
                    do_sample=True,
                )

                gen_time = time.time() - start

                # Orpheus outputs need decoding - this is a simplified version
                # Full implementation would need the audio decoder
                print(f"    Generated tokens in {gen_time:.1f}s (audio decoding not implemented)")

                results.append({
                    "phrase": phrase_name,
                    "text": phrase_text,
                    "generation_time": gen_time,
                    "note": "Token generation only - needs audio decoder"
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({"phrase": phrase_name, "error": str(e)})

            torch.cuda.empty_cache()

        del model, tokenizer
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"  Orpheus test skipped: {e}")
        return [{"error": str(e), "note": "Orpheus requires special setup"}]


def main():
    all_results = {}

    # Test 1: Base CSM + Long Voice Prompt (158s)
    print("\n" + "=" * 70)
    print("TEST 1: Base CSM + Long Voice Prompt (158s)")
    print("=" * 70)
    vp_path = "assets/voice_prompt/maya_voice_prompt_sesame.pt"
    if Path(vp_path).exists():
        all_results["csm_long_vp"] = test_base_csm_with_voice_prompt(vp_path, "CSM + Long VP (158s)")
    else:
        print(f"  Skipped: {vp_path} not found")

    # Test 2: Base CSM + Short Human Voice Prompt (10s)
    print("\n" + "=" * 70)
    print("TEST 2: Base CSM + Short Human Voice Prompt (10s)")
    print("=" * 70)
    vp_path = "assets/voice_prompt/maya_voice_prompt_human.pt"
    if Path(vp_path).exists():
        all_results["csm_human_vp"] = test_base_csm_with_voice_prompt(vp_path, "CSM + Human VP (10s)")
    else:
        print(f"  Skipped: {vp_path} not found")

    # Test 3: Pre-trained Elise model
    print("\n" + "=" * 70)
    print("TEST 3: keanteng/sesame-csm-elise")
    print("=" * 70)
    all_results["elise"] = test_elise_model()

    # Test 4: Orpheus (optional, different architecture)
    print("\n" + "=" * 70)
    print("TEST 4: Orpheus-3B-ft (Alternative)")
    print("=" * 70)
    all_results["orpheus"] = test_orpheus()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")

        if not results:
            print("  No results")
            continue

        # Calculate averages
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            avg_duration = np.mean([r["duration"] for r in valid_results if "duration" in r])
            avg_clicks = np.mean([r["clicks"] for r in valid_results if "clicks" in r])
            avg_time = np.mean([r["generation_time"] for r in valid_results if "generation_time" in r])

            print(f"  Successful: {len(valid_results)}/{len(results)} phrases")
            print(f"  Avg Duration: {avg_duration:.1f}s")
            print(f"  Avg Clicks: {avg_clicks:.0f}")
            print(f"  Avg Gen Time: {avg_time:.1f}s")

        errors = [r for r in results if "error" in r]
        if errors:
            print(f"  Errors: {len(errors)}")

    # Save detailed results
    results_path = OUTPUT_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nDetailed results saved to: {results_path}")
    print(f"Audio samples saved to: {OUTPUT_DIR}/")
    print("\nListen to the samples and compare quality!")
    print("=" * 70)


if __name__ == "__main__":
    main()
