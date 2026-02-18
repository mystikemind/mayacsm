#!/usr/bin/env python3
"""
Test SOTA inference improvements: RAS + EDT + Min-P
Compare standard vs enhanced sampling on base CSM-1B and decoder-only model.

Tests:
1. Standard sampling (original CSM)
2. Enhanced sampling (RAS + EDT + Min-P)
3. A/B comparison with quality metrics

Uses GPU 1 (free) to not interfere with training on GPU 2.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import torch.nn.functional as F
import torchaudio
import time
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
OUTPUT_DIR = PROJECT_ROOT / "audio_inference_test"
OUTPUT_DIR.mkdir(exist_ok=True)


# Test sentences covering different speech patterns
TEST_SENTENCES = [
    # Emotional variety
    "Oh wow, I didn't expect that at all!",
    "Hmm, let me think about that for a second.",
    "That's absolutely amazing, I'm so happy for you!",
    "I'm really sorry to hear that happened.",
    # Disfluencies and natural speech patterns
    "So, like, the thing is... it's complicated, you know?",
    "Well, um, I guess we could try that approach.",
    "Right, right, okay so basically what happened was...",
    # Short utterances
    "Definitely!",
    "Oh no, really?",
    "I see what you mean.",
    # Longer/complex
    "Hey there! So I was thinking about what you said yesterday, and honestly it makes a lot of sense.",
    "You know what, let's just go for it and see what happens!",
]


def load_generator(checkpoint_path=None, device='cuda'):
    """Load CSM Generator (not just Model)."""
    from models import Model
    from generator import Generator

    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)

    if checkpoint_path:
        cp = Path(checkpoint_path)
        merged = cp / "model_merged.pt"
        if merged.exists():
            logger.info(f"Loading merged model from {merged}")
            state = torch.load(merged, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
        else:
            # Load components
            for name in ['decoder', 'projection', 'audio_head']:
                fpath = cp / f"{name}.pt"
                if fpath.exists():
                    logger.info(f"Loading {name} from {fpath}")
                    state = torch.load(fpath, map_location=device, weights_only=True)
                    getattr(model, name).load_state_dict(state)
        logger.info("Checkpoint loaded")

    gen = Generator(model)
    return gen


def load_voice_context(device='cuda'):
    """Load Maya voice prompt as Segment."""
    from generator import Segment

    vp_path = PROJECT_ROOT / "assets" / "voice_prompt" / "maya_voice_prompt.pt"
    if vp_path.exists():
        vp = torch.load(vp_path, map_location=device, weights_only=False)
        if isinstance(vp, dict) and "audio" in vp:
            audio = vp["audio"]
            if audio.dim() > 1:
                audio = audio.squeeze(0)
            text = vp.get("text", "Hey, how's it going?")
            segment = Segment(speaker=0, text=text, audio=audio.to(device))
            logger.info(f"Voice context: {len(audio)/24000:.1f}s")
            return [segment]
    return []


def compute_metrics(audio: torch.Tensor, sr: int = 24000) -> dict:
    """Compute audio quality metrics."""
    audio_np = audio.cpu().numpy().astype(np.float64)

    # Dynamic range (dB)
    rms = np.sqrt(np.mean(audio_np**2))
    peak = np.max(np.abs(audio_np)) + 1e-10
    dr_db = 20 * np.log10(peak / (rms + 1e-10))

    # Silence ratio
    frame_len = int(0.025 * sr)  # 25ms frames
    hop = int(0.01 * sr)         # 10ms hop
    silence_frames = 0
    total_frames = 0
    for i in range(0, len(audio_np) - frame_len, hop):
        frame = audio_np[i:i+frame_len]
        energy = np.sqrt(np.mean(frame**2))
        total_frames += 1
        if energy < 0.01:
            silence_frames += 1
    silence_pct = silence_frames / max(total_frames, 1) * 100

    # Duration
    duration = len(audio_np) / sr

    # Spectral centroid (brightness indicator)
    if len(audio_np) > 512:
        fft = np.fft.rfft(audio_np[:min(len(audio_np), sr)])
        freqs = np.fft.rfftfreq(min(len(audio_np), sr), 1/sr)
        magnitude = np.abs(fft)
        centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
    else:
        centroid = 0

    # Check for repetition artifacts (variance of short-term features)
    if len(audio_np) > sr:
        # Compute frame-level RMS variance
        frame_rms = []
        for i in range(0, len(audio_np) - frame_len, hop):
            frame = audio_np[i:i+frame_len]
            frame_rms.append(np.sqrt(np.mean(frame**2)))
        rms_variance = np.var(frame_rms) if frame_rms else 0
    else:
        rms_variance = 0

    return {
        'dr_db': dr_db,
        'silence_pct': silence_pct,
        'duration': duration,
        'centroid': centroid,
        'rms_variance': rms_variance,
    }


def _multinomial_sample_one(probs):
    """Multinomial sampling without CUDA sync."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def generate_standard(gen, text, context, device='cuda'):
    """Generate with standard CSM sampling."""
    from generator import Segment
    import re

    text = text.strip().lower()
    text = re.sub(r"[^\w\s'$]", "", text)
    text = re.sub(r"\s+", " ", text)

    start = time.time()
    audio = gen.generate(
        text=text,
        speaker=0,
        context=context,
        max_audio_length_ms=8000,
        temperature=0.9,
        topk=50,
    )
    elapsed = time.time() - start
    return audio, elapsed


@torch.inference_mode()
def generate_enhanced(gen, text, context, device='cuda'):
    """Generate with enhanced sampling (RAS + EDT + Min-P)."""
    from generator import Segment
    from models import Model, _index_causal_mask, sample_topk
    import re

    text = text.strip().lower()
    text = re.sub(r"[^\w\s'$]", "", text)
    text = re.sub(r"\s+", " ", text)

    model = gen._model
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod

    dtype = next(model.parameters()).dtype

    # Build tokens using generator's tokenization
    model.reset_caches()

    # Tokenize context + text using Generator's actual API
    tokens_list = []
    tokens_mask_list = []

    for segment in context:
        seg_tokens, seg_mask = gen._tokenize_segment(segment)
        tokens_list.append(seg_tokens)
        tokens_mask_list.append(seg_mask)

    # Generation text (text-only, no audio)
    gen_tokens, gen_mask = gen._tokenize_text_segment(text, 0)
    tokens_list.append(gen_tokens)
    tokens_mask_list.append(gen_mask)

    prompt_tokens = torch.cat(tokens_list, dim=0).long().to(device)
    prompt_tokens_mask = torch.cat(tokens_mask_list, dim=0).bool().to(device)

    # Truncate if needed
    max_gen_len = 100  # 8 seconds
    max_ctx_len = 2048 - max_gen_len
    if prompt_tokens.size(0) >= max_ctx_len:
        prompt_tokens = prompt_tokens[-max_ctx_len:]
        prompt_tokens_mask = prompt_tokens_mask[-max_ctx_len:]

    curr_tokens = prompt_tokens.unsqueeze(0)
    curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
    curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(device)

    zero_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    zero_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

    recent_c0_tokens = []
    samples = []

    # Config
    temperature = 0.9
    topk = 50
    min_p = 0.1
    edt_enabled = False  # Disabled: causes excess silence
    edt_base = 0.8
    edt_sensitivity = 0.1
    ras_window = 10
    ras_threshold = 0.1
    decoder_temp = 0.7

    start = time.time()

    for i in range(max_gen_len):
            # Backbone forward
            curr_backbone_mask = _index_causal_mask(model.backbone_causal_mask, curr_pos)
            embeds = model._embed_tokens(curr_tokens)
            masked_embeds = embeds * curr_tokens_mask.unsqueeze(-1)
            h = masked_embeds.sum(dim=2)
            h = model.backbone(h, input_pos=curr_pos, mask=curr_backbone_mask).to(dtype=dtype)

            last_h = h[:, -1, :]
            c0_logits = model.codebook0_head(last_h)

            # EDT: Dynamic temperature based on entropy
            temp = temperature
            if edt_enabled:
                probs_for_entropy = F.softmax(c0_logits, dim=-1)
                entropy = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-10)).sum(dim=-1)
                temp = temperature * (edt_base ** (edt_sensitivity / (entropy.item() + 1e-8)))
                temp = max(temp, temperature * 0.6)  # Floor at 60% of base temp

            # Apply temperature
            scaled_logits = c0_logits / temp

            # Min-P filtering
            if min_p > 0:
                probs = F.softmax(scaled_logits, dim=-1)
                max_prob = probs.max(dim=-1, keepdim=True)[0]
                threshold = min_p * max_prob
                scaled_logits = scaled_logits.masked_fill(probs < threshold, float('-inf'))

            # Top-K filtering
            indices_to_remove = scaled_logits < torch.topk(scaled_logits, topk)[0][..., -1, None]
            scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample
            probs = F.softmax(scaled_logits, dim=-1)
            c0_sample = _multinomial_sample_one(probs)

            # RAS check
            if len(recent_c0_tokens) >= ras_window:
                token_val = c0_sample.item()
                window_tokens = recent_c0_tokens[-ras_window:]
                rep_ratio = sum(1 for t in window_tokens if t == token_val) / ras_window
                if rep_ratio > ras_threshold:
                    # Resample from full distribution
                    full_probs = F.softmax(c0_logits / temperature, dim=-1)
                    c0_sample = _multinomial_sample_one(full_probs)

            recent_c0_tokens.append(c0_sample.item())

            # Depth decoder
            c0_embed = model._embed_audio(0, c0_sample)
            curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
            curr_sample = c0_sample.clone()
            curr_dec_pos = torch.arange(0, curr_h.size(1), device=device).unsqueeze(0)

            model.decoder.reset_caches()
            for cb in range(1, model.config.audio_num_codebooks):
                curr_decoder_mask = _index_causal_mask(model.decoder_causal_mask, curr_dec_pos)
                decoder_h = model.decoder(
                    model.projection(curr_h), input_pos=curr_dec_pos, mask=curr_decoder_mask
                ).to(dtype=dtype)
                ci_logits = torch.mm(decoder_h[:, -1, :], model.audio_head[cb - 1])
                ci_sample = sample_topk(ci_logits, topk, decoder_temp)
                ci_embed = model._embed_audio(cb, ci_sample)
                curr_h = ci_embed
                curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                curr_dec_pos = curr_dec_pos[:, -1:] + 1

            # Check EOS
            if torch.all(curr_sample == 0):
                break

            samples.append(curr_sample)

            # Update state
            curr_tokens = torch.cat([curr_sample, zero_token], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(curr_sample).bool(), zero_mask], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

    elapsed = time.time() - start

    # Decode audio
    if samples:
        stacked = torch.stack(samples).permute(1, 2, 0)  # (batch, codebooks, frames)
        audio = gen._audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)
    else:
        audio = torch.zeros(1, device=device)

    return audio, elapsed


def main():
    device = 'cuda'
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Load base model
    logger.info("Loading base CSM-1B...")
    gen = load_generator(device=device)
    context = load_voice_context(device)

    logger.info("="*70)
    logger.info("STANDARD vs ENHANCED INFERENCE COMPARISON")
    logger.info("="*70)

    # Warmup
    logger.info("Warming up...")
    _ = gen.generate(text="warmup", speaker=0, context=context, max_audio_length_ms=2000)

    # Check if enhanced generation methods are available
    try:
        # Quick test that Model internals are accessible
        model = gen._model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        _ = model.backbone_causal_mask
        enhanced_available = True
        logger.info("Enhanced generation: AVAILABLE")
    except Exception as e:
        enhanced_available = False
        logger.warning(f"Enhanced generation NOT available: {e}")

    results = {'standard': [], 'enhanced': []}

    for idx, text in enumerate(TEST_SENTENCES):
        logger.info(f"\n--- Sentence {idx+1}/{len(TEST_SENTENCES)}: \"{text[:50]}...\" ---")

        # Standard generation
        audio_std, time_std = generate_standard(gen, text, context, device)
        metrics_std = compute_metrics(audio_std)
        results['standard'].append({
            'text': text, 'time': time_std, **metrics_std
        })

        # Save audio
        torchaudio.save(
            str(OUTPUT_DIR / f"std_{idx:02d}.wav"),
            audio_std.unsqueeze(0).cpu().float(), 24000
        )

        logger.info(f"  Standard:  {time_std:.1f}s | {metrics_std['duration']:.2f}s | "
                    f"DR={metrics_std['dr_db']:.1f}dB | Silence={metrics_std['silence_pct']:.1f}% | "
                    f"Centroid={metrics_std['centroid']:.0f}Hz | RMSVar={metrics_std['rms_variance']:.6f}")

        if enhanced_available:
            # Enhanced generation
            try:
                audio_enh, time_enh = generate_enhanced(gen, text, context, device)
                metrics_enh = compute_metrics(audio_enh)
                results['enhanced'].append({
                    'text': text, 'time': time_enh, **metrics_enh
                })

                torchaudio.save(
                    str(OUTPUT_DIR / f"enh_{idx:02d}.wav"),
                    audio_enh.unsqueeze(0).cpu().float(), 24000
                )

                logger.info(f"  Enhanced:  {time_enh:.1f}s | {metrics_enh['duration']:.2f}s | "
                           f"DR={metrics_enh['dr_db']:.1f}dB | Silence={metrics_enh['silence_pct']:.1f}% | "
                           f"Centroid={metrics_enh['centroid']:.0f}Hz | RMSVar={metrics_enh['rms_variance']:.6f}")
            except Exception as e:
                logger.error(f"  Enhanced FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)

    for mode in ['standard', 'enhanced']:
        if not results[mode]:
            continue
        avg_dr = np.mean([r['dr_db'] for r in results[mode]])
        avg_silence = np.mean([r['silence_pct'] for r in results[mode]])
        avg_time = np.mean([r['time'] for r in results[mode]])
        avg_dur = np.mean([r['duration'] for r in results[mode]])
        avg_centroid = np.mean([r['centroid'] for r in results[mode]])
        avg_rmsvar = np.mean([r['rms_variance'] for r in results[mode]])

        logger.info(f"\n{mode.upper()}:")
        logger.info(f"  Avg DR:       {avg_dr:.1f} dB")
        logger.info(f"  Avg Silence:  {avg_silence:.1f}%")
        logger.info(f"  Avg GenTime:  {avg_time:.2f}s")
        logger.info(f"  Avg Duration: {avg_dur:.2f}s")
        logger.info(f"  Avg Centroid: {avg_centroid:.0f} Hz")
        logger.info(f"  Avg RMSVar:   {avg_rmsvar:.6f}")

    # Quality comparison
    if results['standard'] and results['enhanced']:
        logger.info("\n" + "="*80)
        logger.info("QUALITY DELTA (Enhanced - Standard)")
        logger.info("="*80)

        std_dr = np.mean([r['dr_db'] for r in results['standard']])
        enh_dr = np.mean([r['dr_db'] for r in results['enhanced']])
        std_sil = np.mean([r['silence_pct'] for r in results['standard']])
        enh_sil = np.mean([r['silence_pct'] for r in results['enhanced']])
        std_rms = np.mean([r['rms_variance'] for r in results['standard']])
        enh_rms = np.mean([r['rms_variance'] for r in results['enhanced']])

        logger.info(f"  DR:       {enh_dr - std_dr:+.1f} dB")
        logger.info(f"  Silence:  {enh_sil - std_sil:+.1f}%")
        logger.info(f"  RMSVar:   {enh_rms - std_rms:+.6f} (higher = more dynamic)")

        logger.info("\nAudio saved to: " + str(OUTPUT_DIR))
        logger.info("Listen to std_XX.wav vs enh_XX.wav for perceptual comparison!")


if __name__ == "__main__":
    main()
