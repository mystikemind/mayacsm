#!/usr/bin/env python3
"""
Orpheus 3B Fine-Tuning: Data Preparation Pipeline
===================================================

Prepares Expresso dataset for Orpheus 3B LoRA fine-tuning.

Pipeline:
1. Load Expresso dataset from HuggingFace (ylacombe/expresso)
2. Filter: single female speaker (ex04=Talia), conversational styles
3. Quality filter: duration 2-12s, minimum text length
4. Resample 48kHz → 24kHz
5. SNAC encode → 7-token-per-frame audio codes
6. Remove duplicate frames (compress silence)
7. Build training sequences (Orpheus format)
8. Validate: decode back and verify audio quality
9. Save tokenized dataset

Token format (official Orpheus):
    [BOS, START_HUMAN, text_tokens, END_HUMAN, START_AI, START_SPEECH,
     audio_codes, END_SPEECH, END_AI]

Where:
    BOS = 128000 (<|begin_of_text|>)
    START_HUMAN = 128259 (<custom_token_3>)
    END_HUMAN = 128260 (<custom_token_4>)
    START_AI = 128261 (<custom_token_5>)
    START_SPEECH = 128257 (<custom_token_1>)
    END_SPEECH = 128258 (<custom_token_2>)
    END_AI = 128262 (<custom_token_6>)
    audio_codes = SNAC token IDs (128266-156937)

Usage:
    python 20_prepare_orpheus_data.py [--speaker ex04] [--voice-name maya]
"""

import os
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/.cache/huggingface"
_cudnn = os.path.join(os.path.dirname(os.__file__), "..", "nvidia", "cudnn", "lib")
if os.path.isdir(_cudnn):
    os.environ["LD_LIBRARY_PATH"] = _cudnn + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import argparse
import json
import logging
import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from snac import SNAC
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (Official Orpheus token IDs)
# =============================================================================

BOS_TOKEN = 128000           # <|begin_of_text|>
EOT_TOKEN = 128009           # <|eot_id|>
START_OF_SPEECH = 128257     # <custom_token_1>
END_OF_SPEECH = 128258       # <custom_token_2>
START_OF_HUMAN = 128259      # <custom_token_3>
END_OF_HUMAN = 128260        # <custom_token_4>
START_OF_AI = 128261         # <custom_token_5>
END_OF_AI = 128262           # <custom_token_6>
PAD_TOKEN = 128263           # <custom_token_7>

AUDIO_TOKEN_BASE = 128266    # First audio token (custom_token_10)
AUDIO_TOKEN_MAX = 128266 + (7 * 4096) - 1  # Last audio token

# SNAC codec settings
SNAC_SR = 24000              # SNAC expects 24kHz
EXPRESSO_SR = 48000          # Expresso is 48kHz

# Training constraints
MAX_SEQ_LENGTH = 8192        # Maximum training sequence length
MIN_AUDIO_TOKENS = 70        # ~0.5s minimum (10 frames * 7 tokens)
MAX_AUDIO_TOKENS = 7000      # ~12s maximum (1000 frames * 7 tokens)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Conversational styles to include (drop singing, longform, essentials, narration)
CONVERSATIONAL_STYLES = [
    "default",      # Neutral baseline - most important
    "happy",        # Warm, upbeat
    "sad",          # Empathetic, somber
    "confused",     # Questioning, uncertain
    "laughing",     # Light, amused
    "whisper",      # Intimate, quiet
    "enunciated",   # Clear, emphatic
    "emphasis",     # Stressed/emphasized speech
]

# Duration filter (seconds)
MIN_DURATION = 2.0
MAX_DURATION = 12.0

# Minimum text length (characters)
MIN_TEXT_LENGTH = 10

# Emotion tag mapping (Expresso style → Orpheus emotion tag prefix)
# Only add tags that our LLM actually outputs at inference time.
# Most emotions are carried by the audio itself, not text tags.
STYLE_TO_EMOTION = {
    "default": "",
    "happy": "",           # Let voice carry the emotion naturally
    "sad": "",
    "confused": "",
    "laughing": "<laugh> ",  # LLM outputs <laugh> tags
    "whisper": "",
    "enunciated": "",
    "emphasis": "",
}

# Style tag patterns to strip from Expresso text
# e.g., "[confused] Ready for bed?" → "Ready for bed?"
STYLE_TAG_PATTERN = re.compile(r'^\[(?:confused|default|enunciated|happy|laughing|sad|whisper|emphasis|essentials|singing|longform|narration)\]\s*')


# =============================================================================
# SNAC ENCODING
# =============================================================================

class SNACEncoder:
    """Encodes audio to Orpheus SNAC token format."""

    def __init__(self, device="cuda:2"):
        self.device = device
        logger.info(f"Loading SNAC 24kHz model on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        self.resample = T.Resample(orig_freq=EXPRESSO_SR, new_freq=SNAC_SR)
        logger.info("SNAC encoder ready")

    def encode(self, waveform: np.ndarray, source_sr: int = EXPRESSO_SR) -> list:
        """Encode audio waveform to Orpheus SNAC token IDs.

        Args:
            waveform: numpy array of audio samples
            source_sr: source sample rate (default 48kHz for Expresso)

        Returns:
            List of token IDs (128266+) in 7-per-frame interleaved format
        """
        # Convert to torch tensor
        audio = torch.from_numpy(waveform).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample to 24kHz
        if source_sr != SNAC_SR:
            resample = T.Resample(orig_freq=source_sr, new_freq=SNAC_SR)
            audio = resample(audio)

        # Normalize amplitude
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak * 0.95

        # Encode with SNAC
        audio = audio.unsqueeze(0).to(self.device)  # [1, 1, samples]

        with torch.inference_mode():
            codes = self.model.encode(audio)

        # codes: list of 3 tensors
        #   codes[0]: [1, N]    - Layer 0 (coarsest, 1 code/frame)
        #   codes[1]: [1, 2*N]  - Layer 1 (2 codes/frame)
        #   codes[2]: [1, 4*N]  - Layer 2 (finest, 4 codes/frame)

        n_frames = codes[0].shape[1]

        # Interleave into 7-token-per-frame format
        all_tokens = []
        for i in range(n_frames):
            # Position 0: L0[i] + offset 0*4096
            all_tokens.append(codes[0][0][i].item() + AUDIO_TOKEN_BASE)
            # Position 1: L1[2*i] + offset 1*4096
            all_tokens.append(codes[1][0][2*i].item() + AUDIO_TOKEN_BASE + 4096)
            # Position 2: L2[4*i] + offset 2*4096
            all_tokens.append(codes[2][0][4*i].item() + AUDIO_TOKEN_BASE + (2*4096))
            # Position 3: L2[4*i+1] + offset 3*4096
            all_tokens.append(codes[2][0][4*i+1].item() + AUDIO_TOKEN_BASE + (3*4096))
            # Position 4: L1[2*i+1] + offset 4*4096
            all_tokens.append(codes[1][0][2*i+1].item() + AUDIO_TOKEN_BASE + (4*4096))
            # Position 5: L2[4*i+2] + offset 5*4096
            all_tokens.append(codes[2][0][4*i+2].item() + AUDIO_TOKEN_BASE + (5*4096))
            # Position 6: L2[4*i+3] + offset 6*4096
            all_tokens.append(codes[2][0][4*i+3].item() + AUDIO_TOKEN_BASE + (6*4096))

        return all_tokens

    def decode(self, token_ids: list) -> torch.Tensor:
        """Decode SNAC tokens back to audio (for verification)."""
        n = (len(token_ids) // 7) * 7
        if n < 7:
            return None
        token_ids = token_ids[:n]

        l0, l1, l2 = [], [], []
        for i in range(n // 7):
            b = 7 * i
            l0.append(max(0, min(4095, token_ids[b] - AUDIO_TOKEN_BASE)))
            l1.append(max(0, min(4095, token_ids[b+1] - AUDIO_TOKEN_BASE - 4096)))
            l2.append(max(0, min(4095, token_ids[b+2] - AUDIO_TOKEN_BASE - 2*4096)))
            l2.append(max(0, min(4095, token_ids[b+3] - AUDIO_TOKEN_BASE - 3*4096)))
            l1.append(max(0, min(4095, token_ids[b+4] - AUDIO_TOKEN_BASE - 4*4096)))
            l2.append(max(0, min(4095, token_ids[b+5] - AUDIO_TOKEN_BASE - 5*4096)))
            l2.append(max(0, min(4095, token_ids[b+6] - AUDIO_TOKEN_BASE - 6*4096)))

        codes = [
            torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(self.device),
            torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(self.device),
            torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(self.device),
        ]
        with torch.inference_mode():
            audio = self.model.decode(codes)
        return audio.squeeze().cpu()


def remove_duplicate_frames(token_ids: list) -> list:
    """Remove consecutive duplicate frames to compress silence/sustained sounds.

    Only removes frames where the Layer 0 code (coarsest level) is identical,
    indicating very similar audio content. This is a mild compression that
    removes redundancy without losing quality.
    """
    if len(token_ids) < 7:
        return token_ids

    n = (len(token_ids) // 7) * 7
    token_ids = token_ids[:n]

    result = token_ids[:7]  # Always keep first frame
    removed = 0

    for i in range(7, n, 7):
        current_l0 = token_ids[i]       # Layer 0 code of current frame
        prev_l0 = result[-7]            # Layer 0 code of previous kept frame

        if current_l0 != prev_l0:
            result.extend(token_ids[i:i+7])
        else:
            removed += 1

    return result


# =============================================================================
# DATA LOADING AND FILTERING
# =============================================================================

def load_and_filter_expresso(speaker: str, styles: list, min_dur: float, max_dur: float):
    """Load Expresso dataset from cached Arrow files with direct audio decoding.

    Uses PyArrow + soundfile to avoid torchcodec dependency issues.
    The dataset is already cached from a previous HF download.
    """
    import pyarrow.ipc as ipc
    import io
    import soundfile as sf
    import glob

    logger.info(f"Loading Expresso dataset (speaker={speaker}, styles={len(styles)})...")
    t0 = time.time()

    # Find cached Arrow files
    cache_dir = "/home/ec2-user/SageMaker/.cache/huggingface/datasets/ylacombe___expresso/read/0.0.0"
    arrow_pattern = os.path.join(cache_dir, "**/expresso-train-*.arrow")
    arrow_files = sorted(glob.glob(arrow_pattern, recursive=True))

    if not arrow_files:
        # Try to download metadata first (without audio decoding)
        logger.info("Arrow cache not found, downloading metadata from HuggingFace...")
        _mock_torchcodec()
        from datasets import load_dataset
        ds = load_dataset("ylacombe/expresso", split="train")
        arrow_files = sorted(glob.glob(arrow_pattern, recursive=True))
        if not arrow_files:
            raise FileNotFoundError(f"Could not find Arrow files in {cache_dir}")

    logger.info(f"Found {len(arrow_files)} Arrow shard files")

    # Read all Arrow files and filter
    samples = []
    total_read = 0

    for arrow_file in arrow_files:
        reader = ipc.open_stream(arrow_file)
        table = reader.read_all()
        total_read += table.num_rows

        for i in range(table.num_rows):
            speaker_id = table.column("speaker_id")[i].as_py()
            style = table.column("style")[i].as_py()
            text = table.column("text")[i].as_py()
            sample_id = table.column("id")[i].as_py()

            # Filter speaker
            if speaker_id != speaker:
                continue

            # Filter style
            if style not in styles:
                continue

            # Filter text length
            if len(text.strip()) < MIN_TEXT_LENGTH:
                continue

            # Decode audio to get duration
            audio_struct = table.column("audio")[i].as_py()
            audio_bytes = audio_struct.get("bytes")
            if audio_bytes is None:
                continue

            try:
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                duration = len(audio_array) / sr
            except Exception as e:
                logger.warning(f"Failed to decode audio for {sample_id}: {e}")
                continue

            # Filter duration
            if not (min_dur <= duration <= max_dur):
                continue

            samples.append({
                "text": text,
                "style": style,
                "speaker_id": speaker_id,
                "id": sample_id,
                "audio_array": audio_array,
                "audio_sr": sr,
                "duration": duration,
            })

    elapsed = time.time() - t0
    logger.info(f"Read {total_read} total samples in {elapsed:.1f}s")
    logger.info(f"After all filters: {len(samples)} samples")

    # Log style distribution
    style_counts = Counter(s["style"] for s in samples)
    logger.info("Style distribution:")
    for style, count in style_counts.most_common():
        logger.info(f"  {style}: {count}")

    # Duration stats
    durations = [s["duration"] for s in samples]
    logger.info(f"Duration: mean={np.mean(durations):.2f}s, "
                f"median={np.median(durations):.2f}s, "
                f"total={sum(durations)/3600:.2f}h")

    return samples


def _mock_torchcodec():
    """Mock torchcodec to avoid import errors with datasets library."""
    import types
    import importlib.machinery
    mock = types.ModuleType('torchcodec')
    mock.__spec__ = importlib.machinery.ModuleSpec('torchcodec', None)
    sys.modules['torchcodec'] = mock
    for sub in ['decoders', 'decoders._core', 'decoders._core.video_decoder_ops']:
        name = f'torchcodec.{sub}'
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = m


# =============================================================================
# TOKENIZATION
# =============================================================================

def build_training_sequence(
    text: str,
    audio_tokens: list,
    voice_name: str,
    tokenizer,
    style: str = "default",
) -> dict:
    """Build a single Orpheus training sequence.

    Format:
        [BOS, START_HUMAN, text_tokens, END_HUMAN, START_AI,
         START_SPEECH, audio_tokens, END_SPEECH, END_AI]

    This matches the inference format:
        prompt = "<|begin_of_text|><custom_token_3>voice: text<custom_token_4><custom_token_5>"
        model generates: <custom_token_1>audio_tokens<custom_token_2>

    Args:
        text: transcript text
        audio_tokens: list of SNAC token IDs
        voice_name: voice identifier (e.g., "maya")
        tokenizer: HuggingFace tokenizer
        style: emotion style for potential tag prefix

    Returns:
        dict with input_ids, labels, attention_mask
    """
    # Strip Expresso style tags from text: "[confused] text" → "text"
    text = STYLE_TAG_PATTERN.sub("", text).strip()

    # Add emotion tag prefix if applicable (only for styles our LLM outputs)
    emotion_prefix = STYLE_TO_EMOTION.get(style, "")

    # Lowercase (Orpheus pretraining uses lowercase conversational text)
    text = text.lower()

    # Remove excessive punctuation but keep apostrophes and basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()

    # Build text prompt: "maya: text"
    text_prompt = f"{voice_name}: {emotion_prefix}{text}"

    # Tokenize text (without special tokens - we add them manually)
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=False)

    # Build full sequence
    input_ids = (
        [BOS_TOKEN, START_OF_HUMAN]
        + text_ids
        + [END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
        + audio_tokens
        + [END_OF_SPEECH, END_OF_AI]
    )

    # Validate length
    if len(input_ids) > MAX_SEQ_LENGTH:
        return None

    # Labels = input_ids (full autoregressive training)
    # We train on the complete sequence so the model learns:
    # 1. Text-to-audio mapping
    # 2. Audio token sequences
    # 3. Proper termination (END_OF_SPEECH)
    labels = list(input_ids)

    # Attention mask (all 1s, padding handled by data collator)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare Orpheus fine-tuning data")
    parser.add_argument("--speaker", default="ex04", help="Expresso speaker ID (default: ex04=Talia)")
    parser.add_argument("--voice-name", default="maya", help="Voice name in prompts")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--snac-device", default="cuda:2", help="GPU for SNAC encoding")
    parser.add_argument("--verify-n", type=int, default=5, help="Number of samples to verify by decoding")
    parser.add_argument("--no-dedup", action="store_true", help="Skip duplicate frame removal")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("/home/ec2-user/SageMaker/project_maya/training/data/orpheus_finetune")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_dir = output_dir / "verification_audio"
    verify_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("  ORPHEUS 3B FINE-TUNING: DATA PREPARATION")
    logger.info(f"  Speaker: {args.speaker} | Voice: {args.voice_name}")
    logger.info(f"  Styles: {len(CONVERSATIONAL_STYLES)}")
    logger.info(f"  Duration: {MIN_DURATION}-{MAX_DURATION}s")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 80)

    # Step 1: Load and filter dataset
    logger.info("\n[Step 1/6] Loading and filtering Expresso dataset...")
    ds = load_and_filter_expresso(
        speaker=args.speaker,
        styles=CONVERSATIONAL_STYLES,
        min_dur=MIN_DURATION,
        max_dur=MAX_DURATION,
    )

    # Step 2: Initialize SNAC encoder
    logger.info("\n[Step 2/6] Initializing SNAC encoder...")
    snac = SNACEncoder(device=args.snac_device)

    # Step 3: Load tokenizer
    logger.info("\n[Step 3/6] Loading Orpheus tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface"
    )
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Step 4: Process all samples
    logger.info(f"\n[Step 4/6] Processing {len(ds)} samples (SNAC encode + tokenize)...")

    processed = []
    failed = 0
    too_long = 0
    too_short = 0
    stats = {
        "audio_tokens": [],
        "total_tokens": [],
        "text_tokens": [],
        "durations": [],
    }

    t0 = time.time()
    for idx in range(len(ds)):
        sample = ds[idx]
        text = sample["text"].strip()
        style = sample["style"]
        waveform = sample["audio_array"]
        sr = sample["audio_sr"]
        duration = sample["duration"]

        # SNAC encode
        try:
            audio_tokens = snac.encode(waveform, source_sr=sr)
        except Exception as e:
            logger.warning(f"SNAC encode failed for sample {idx}: {e}")
            failed += 1
            continue

        # Remove duplicate frames
        if not args.no_dedup:
            original_len = len(audio_tokens)
            audio_tokens = remove_duplicate_frames(audio_tokens)
            dedup_ratio = len(audio_tokens) / original_len if original_len > 0 else 1.0

        # Check audio token count
        if len(audio_tokens) < MIN_AUDIO_TOKENS:
            too_short += 1
            continue
        if len(audio_tokens) > MAX_AUDIO_TOKENS:
            too_long += 1
            continue

        # Build training sequence
        seq = build_training_sequence(
            text=text,
            audio_tokens=audio_tokens,
            voice_name=args.voice_name,
            tokenizer=tokenizer,
            style=style,
        )

        if seq is None:
            too_long += 1
            continue

        # Track stats
        n_text = len(seq["input_ids"]) - len(audio_tokens) - 5  # subtract control tokens
        stats["audio_tokens"].append(len(audio_tokens))
        stats["total_tokens"].append(len(seq["input_ids"]))
        stats["text_tokens"].append(n_text)
        stats["durations"].append(duration)

        # Add metadata (not part of training, for reference)
        seq["_text"] = text
        seq["_style"] = style
        seq["_duration"] = duration
        seq["_n_audio_tokens"] = len(audio_tokens)
        seq["_idx"] = idx

        processed.append(seq)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(ds) - idx - 1) / rate
            logger.info(
                f"  [{idx+1}/{len(ds)}] Processed: {len(processed)} | "
                f"Failed: {failed} | Short: {too_short} | Long: {too_long} | "
                f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s"
            )

    elapsed = time.time() - t0
    logger.info(f"\nProcessing complete in {elapsed:.1f}s")
    logger.info(f"  Total processed: {len(processed)}")
    logger.info(f"  Failed (SNAC): {failed}")
    logger.info(f"  Too short (<{MIN_AUDIO_TOKENS} tokens): {too_short}")
    logger.info(f"  Too long (>{MAX_AUDIO_TOKENS} tokens or >{MAX_SEQ_LENGTH} total): {too_long}")

    if not processed:
        logger.error("No samples processed! Check data and parameters.")
        return

    # Step 5: Verify by decoding a few samples
    logger.info(f"\n[Step 5/6] Verifying {args.verify_n} samples by SNAC decode...")
    for i in range(min(args.verify_n, len(processed))):
        sample = processed[i]
        # Extract audio tokens from input_ids (between START_SPEECH and END_SPEECH)
        ids = sample["input_ids"]
        speech_start = ids.index(START_OF_SPEECH) + 1
        speech_end = ids.index(END_OF_SPEECH)
        audio_token_ids = ids[speech_start:speech_end]

        audio = snac.decode(audio_token_ids)
        if audio is not None:
            out_path = verify_dir / f"verify_{i:03d}_{sample['_style']}.wav"
            torchaudio.save(str(out_path), audio.unsqueeze(0), SNAC_SR)
            logger.info(
                f"  Verified sample {i}: style={sample['_style']}, "
                f"text='{sample['_text'][:50]}...', "
                f"audio_tokens={len(audio_token_ids)}, "
                f"duration={audio.shape[-1]/SNAC_SR:.2f}s"
            )

    # Step 6: Save dataset
    logger.info(f"\n[Step 6/6] Saving dataset to {output_dir}...")

    # Save training data (only training columns)
    train_data = []
    for sample in processed:
        train_data.append({
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
            "attention_mask": sample["attention_mask"],
        })

    # Save as JSON Lines (each line is one sample)
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved {len(train_data)} training samples to {train_path}")

    # Save metadata separately
    meta_data = []
    for sample in processed:
        meta_data.append({
            "text": sample["_text"],
            "style": sample["_style"],
            "duration": sample["_duration"],
            "n_audio_tokens": sample["_n_audio_tokens"],
            "n_total_tokens": len(sample["input_ids"]),
            "idx": sample["_idx"],
        })

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    # Save dataset statistics
    stats_summary = {
        "total_samples": len(processed),
        "speaker": args.speaker,
        "voice_name": args.voice_name,
        "styles": CONVERSATIONAL_STYLES,
        "duration_range": [MIN_DURATION, MAX_DURATION],
        "failed": failed,
        "too_short": too_short,
        "too_long": too_long,
        "audio_tokens": {
            "mean": float(np.mean(stats["audio_tokens"])),
            "std": float(np.std(stats["audio_tokens"])),
            "min": int(min(stats["audio_tokens"])),
            "max": int(max(stats["audio_tokens"])),
            "median": float(np.median(stats["audio_tokens"])),
        },
        "total_tokens": {
            "mean": float(np.mean(stats["total_tokens"])),
            "std": float(np.std(stats["total_tokens"])),
            "min": int(min(stats["total_tokens"])),
            "max": int(max(stats["total_tokens"])),
            "median": float(np.median(stats["total_tokens"])),
        },
        "text_tokens": {
            "mean": float(np.mean(stats["text_tokens"])),
            "median": float(np.median(stats["text_tokens"])),
        },
        "durations": {
            "mean": float(np.mean(stats["durations"])),
            "total_hours": float(sum(stats["durations"]) / 3600),
        },
        "style_distribution": dict(Counter(s["_style"] for s in processed)),
    }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("  DATASET SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Samples: {stats_summary['total_samples']}")
    logger.info(f"  Total audio: {stats_summary['durations']['total_hours']:.2f} hours")
    logger.info(f"  Avg duration: {stats_summary['durations']['mean']:.2f}s")
    logger.info(f"  Audio tokens: {stats_summary['audio_tokens']['mean']:.0f} ± {stats_summary['audio_tokens']['std']:.0f}")
    logger.info(f"  Total tokens: {stats_summary['total_tokens']['mean']:.0f} ± {stats_summary['total_tokens']['std']:.0f}")
    logger.info(f"  Style distribution:")
    for style, count in sorted(stats_summary["style_distribution"].items()):
        logger.info(f"    {style}: {count}")
    logger.info(f"\n  Files saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
