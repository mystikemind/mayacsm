#!/usr/bin/env python3
"""
Head-to-head comparison: Base CSM-1B vs Fine-tuned CSM-1B

Tests whether our LoRA fine-tuning on 2h of Expresso read speech
has HURT the base model's natural conversational capabilities.

The base CSM-1B was trained on ~1M hours of conversation by Sesame.
Our fine-tuning may have overwritten its naturalness with flat read speech.
"""

import sys
import os
import time
import json
import torch
import torchaudio

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

OUTPUT_DIR = "/home/ec2-user/SageMaker/project_maya/audio_comparison_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test sentences designed to reveal naturalness differences
TEST_SENTENCES = [
    "hmm, thats actually really interesting, tell me more",
    "oh yeah, i totally get what you mean",
    "aww, that sounds tough, im here for you",
    "haha, no way, thats hilarious",
    "well, i think you should just go for it, you know?",
    "yeah, im doing pretty good, thanks for asking",
    "oh really? i didnt know that, thats cool",
    "hmm, let me think about that for a second",
]


def load_base_model(device="cuda:2"):
    """Load base CSM-1B without fine-tuning."""
    from models import Model
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    from transformers import AutoTokenizer
    from tokenizers.processors import TemplateProcessing

    print("=" * 60)
    print("LOADING BASE CSM-1B (no fine-tuning)")
    print("=" * 60)

    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(bos, tokenizer.bos_token_id), (eos, tokenizer.eos_token_id)],
    )

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(32)

    return model, tokenizer, mimi, device


def load_finetuned_model(device="cuda:3"):
    """Load fine-tuned CSM-1B."""
    from models import Model
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    from transformers import AutoTokenizer
    from tokenizers.processors import TemplateProcessing

    print("=" * 60)
    print("LOADING FINE-TUNED CSM-1B")
    print("=" * 60)

    FINETUNED = '/home/ec2-user/SageMaker/project_maya/training/checkpoints/csm_maya_correct/best_model/model_merged.pt'

    model = Model.from_pretrained("sesame/csm-1b")
    if os.path.exists(FINETUNED):
        state_dict = torch.load(FINETUNED, map_location=device, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print("Fine-tuned weights loaded")
    else:
        print(f"WARNING: Fine-tuned model not found at {FINETUNED}")

    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(bos, tokenizer.bos_token_id), (eos, tokenizer.eos_token_id)],
    )

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(32)

    return model, tokenizer, mimi, device


def load_voice_context(tokenizer, mimi, device):
    """Load voice context from training data."""
    TRAINING_DATA = '/home/ec2-user/SageMaker/project_maya/training/data/csm_ready_ex04'
    train_json = os.path.join(TRAINING_DATA, 'train.json')

    contexts = []
    if os.path.exists(train_json):
        with open(train_json) as f:
            train_samples = json.load(f)

        default_samples = [s for s in train_samples if s.get("style") == "default"][:5]
        total_dur = 0
        for sample in default_samples:
            if total_dur >= 10:
                break
            audio_path = os.path.join(TRAINING_DATA, sample["path"])
            if not os.path.exists(audio_path):
                continue
            audio, sr = torchaudio.load(audio_path)
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)
            audio = audio.mean(dim=0) if audio.dim() > 1 and audio.size(0) > 1 else audio.squeeze(0)
            audio = audio.to(device)

            text = sample["text"]
            if text.startswith("["):
                text = text.split("]", 1)[-1].strip()

            contexts.append({"text": text, "audio": audio, "speaker": 0})
            total_dur += sample.get("duration", len(audio) / 24000)
            print(f"  Context: '{text[:50]}...' ({len(audio)/24000:.1f}s)")

    return contexts


def tokenize_text_segment(tokenizer, text, speaker, device):
    text_tokens = tokenizer.encode(f"[{speaker}]{text}")
    text_frame = torch.zeros(len(text_tokens), 33).long()
    text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
    text_frame[:, -1] = torch.tensor(text_tokens)
    text_frame_mask[:, -1] = True
    return text_frame.to(device), text_frame_mask.to(device)


def tokenize_audio(mimi, audio, device):
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = audio.clone().detach().to(device)
    audio_tokens = mimi.encode(audio.unsqueeze(0).unsqueeze(0))[0]
    eos_frame = torch.zeros(audio_tokens.size(0), 1).to(device)
    audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
    audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(device)
    audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(device)
    audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
    audio_frame_mask[:, :-1] = True
    return audio_frame, audio_frame_mask


@torch.inference_mode()
def generate_audio(model, tokenizer, mimi, device, text, contexts=None,
                   temperature=1.0, topk=60, depth_temp=0.7, max_frames=75):
    """Generate audio with given model."""
    from moshi.utils.compile import no_cuda_graph

    model.reset_caches()

    tokens_list, masks_list = [], []

    # Add context if provided
    if contexts:
        for ctx in contexts:
            t, m = tokenize_text_segment(tokenizer, ctx["text"], ctx["speaker"], device)
            tokens_list.append(t)
            masks_list.append(m)
            at, am = tokenize_audio(mimi, ctx["audio"], device)
            tokens_list.append(at)
            masks_list.append(am)

    # Add text to generate
    gen_t, gen_m = tokenize_text_segment(tokenizer, text, 0, device)
    tokens_list.append(gen_t)
    masks_list.append(gen_m)

    prompt_tokens = torch.cat(tokens_list, dim=0).long().to(device)
    prompt_mask = torch.cat(masks_list, dim=0).bool().to(device)

    curr_tokens = prompt_tokens.unsqueeze(0)
    curr_mask = prompt_mask.unsqueeze(0)
    curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(device)

    frames = []
    start = time.time()

    with no_cuda_graph():
        for _ in range(max_frames):
            sample = model.generate_frame(
                curr_tokens, curr_mask, curr_pos,
                temperature=temperature, topk=topk,
                depth_decoder_temperature=depth_temp
            )
            if torch.all(sample == 0):
                break
            frames.append(sample.clone())
            curr_tokens = torch.cat(
                [sample, torch.zeros(1, 1).long().to(device)], dim=1
            ).unsqueeze(1)
            curr_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

    gen_time = time.time() - start

    if not frames:
        return torch.zeros(24000).to(device), gen_time

    stacked = torch.stack(frames).permute(1, 2, 0)
    audio = mimi.decode(stacked).squeeze(0).squeeze(0)

    # Peak normalize
    peak = audio.abs().max()
    if peak > 1e-6:
        audio = audio * (0.8 / peak)

    return audio, gen_time


def analyze_audio(audio, sr=24000, label=""):
    """Analyze audio quality metrics."""
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio

    import numpy as np

    rms = np.sqrt(np.mean(audio_np ** 2))
    peak = np.abs(audio_np).max()
    duration = len(audio_np) / sr

    # Pitch variation (simple zero-crossing rate as proxy)
    zcr = np.sum(np.abs(np.diff(np.sign(audio_np)))) / (2 * len(audio_np))

    # Dynamic range
    if rms > 0:
        crest_factor = peak / rms
    else:
        crest_factor = 0

    # Silence ratio (proportion of near-silence frames)
    frame_size = int(0.025 * sr)  # 25ms frames
    silence_count = 0
    total_frames = 0
    for i in range(0, len(audio_np) - frame_size, frame_size):
        frame = audio_np[i:i+frame_size]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        total_frames += 1
        if frame_rms < 0.01:
            silence_count += 1

    silence_ratio = silence_count / max(total_frames, 1)

    print(f"  {label}")
    print(f"    Duration: {duration:.2f}s")
    print(f"    RMS: {rms:.4f}, Peak: {peak:.4f}")
    print(f"    ZCR: {zcr:.4f} (higher = more fricatives/natural)")
    print(f"    Crest factor: {crest_factor:.2f} (higher = more dynamic)")
    print(f"    Silence ratio: {silence_ratio:.1%} (pauses/breathing)")

    return {
        "duration": duration,
        "rms": float(rms),
        "peak": float(peak),
        "zcr": float(zcr),
        "crest_factor": float(crest_factor),
        "silence_ratio": float(silence_ratio),
    }


def main():
    print("=" * 70)
    print("CSM-1B COMPARISON: BASE vs FINE-TUNED")
    print("Testing if fine-tuning on 2h Expresso HURT base naturalness")
    print("=" * 70)

    # Load both models on different GPUs
    base_model, base_tok, base_mimi, base_dev = load_base_model(device="cuda:2")
    ft_model, ft_tok, ft_mimi, ft_dev = load_finetuned_model(device="cuda:3")

    # Setup KV caches
    base_model.setup_caches(1)
    ft_model.setup_caches(1)

    # Load voice context (same for both)
    print("\nLoading voice context...")
    base_ctx = load_voice_context(base_tok, base_mimi, base_dev)
    ft_ctx = load_voice_context(ft_tok, ft_mimi, ft_dev)

    # Warmup
    print("\nWarming up models...")
    for _ in range(3):
        generate_audio(base_model, base_tok, base_mimi, base_dev, "hello", base_ctx, max_frames=25)
        generate_audio(ft_model, ft_tok, ft_mimi, ft_dev, "hello", ft_ctx, max_frames=25)

    results = {"base": [], "finetuned": []}

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON SAMPLES")
    print("=" * 70)

    for i, text in enumerate(TEST_SENTENCES):
        print(f"\n--- Sentence {i+1}: '{text}' ---")

        # Generate with base model
        base_audio, base_time = generate_audio(
            base_model, base_tok, base_mimi, base_dev, text, base_ctx,
            temperature=1.0, topk=60, depth_temp=0.7
        )
        base_path = os.path.join(OUTPUT_DIR, f"base_{i+1:02d}.wav")
        torchaudio.save(base_path, base_audio.unsqueeze(0).cpu(), 24000)
        base_metrics = analyze_audio(base_audio, label=f"BASE (gen: {base_time*1000:.0f}ms)")

        # Generate with fine-tuned model
        ft_audio, ft_time = generate_audio(
            ft_model, ft_tok, ft_mimi, ft_dev, text, ft_ctx,
            temperature=1.0, topk=60, depth_temp=0.7
        )
        ft_path = os.path.join(OUTPUT_DIR, f"finetuned_{i+1:02d}.wav")
        torchaudio.save(ft_path, ft_audio.unsqueeze(0).cpu(), 24000)
        ft_metrics = analyze_audio(ft_audio, label=f"FINE-TUNED (gen: {ft_time*1000:.0f}ms)")

        results["base"].append({"text": text, "time_ms": base_time * 1000, **base_metrics})
        results["finetuned"].append({"text": text, "time_ms": ft_time * 1000, **ft_metrics})

    # Also test WITHOUT context (to see raw model capability)
    print("\n" + "=" * 70)
    print("TESTING WITHOUT CONTEXT (raw model capability)")
    print("=" * 70)

    no_ctx_texts = [
        "hey there, how are you doing today?",
        "oh wow, thats amazing, im so happy for you!",
    ]

    for i, text in enumerate(no_ctx_texts):
        print(f"\n--- No-context {i+1}: '{text}' ---")

        base_audio, base_time = generate_audio(
            base_model, base_tok, base_mimi, base_dev, text, None,
            temperature=1.0, topk=60, depth_temp=0.7
        )
        base_path = os.path.join(OUTPUT_DIR, f"base_noctx_{i+1:02d}.wav")
        torchaudio.save(base_path, base_audio.unsqueeze(0).cpu(), 24000)
        analyze_audio(base_audio, label=f"BASE no-ctx (gen: {base_time*1000:.0f}ms)")

        ft_audio, ft_time = generate_audio(
            ft_model, ft_tok, ft_mimi, ft_dev, text, None,
            temperature=1.0, topk=60, depth_temp=0.7
        )
        ft_path = os.path.join(OUTPUT_DIR, f"finetuned_noctx_{i+1:02d}.wav")
        torchaudio.save(ft_path, ft_audio.unsqueeze(0).cpu(), 24000)
        analyze_audio(ft_audio, label=f"FINE-TUNED no-ctx (gen: {ft_time*1000:.0f}ms)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    import numpy as np

    for model_name in ["base", "finetuned"]:
        metrics = results[model_name]
        avg_time = np.mean([m["time_ms"] for m in metrics])
        avg_zcr = np.mean([m["zcr"] for m in metrics])
        avg_crest = np.mean([m["crest_factor"] for m in metrics])
        avg_silence = np.mean([m["silence_ratio"] for m in metrics])
        avg_duration = np.mean([m["duration"] for m in metrics])

        print(f"\n{model_name.upper()}:")
        print(f"  Avg gen time: {avg_time:.0f}ms")
        print(f"  Avg duration: {avg_duration:.2f}s")
        print(f"  Avg ZCR: {avg_zcr:.4f}")
        print(f"  Avg crest factor: {avg_crest:.2f}")
        print(f"  Avg silence ratio: {avg_silence:.1%}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudio samples saved to: {OUTPUT_DIR}/")
    print(f"Results saved to: {results_path}")
    print("\nLISTEN to the audio files to compare naturalness!")
    print("Key things to listen for:")
    print("  - Does the base model have more natural pauses/breathing?")
    print("  - Does the fine-tuned model sound more 'read aloud'?")
    print("  - Which has more pitch variation?")
    print("  - Which sounds more like a real conversation?")


if __name__ == "__main__":
    main()
