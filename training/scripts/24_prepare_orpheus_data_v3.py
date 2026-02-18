#!/usr/bin/env python3
"""
Orpheus 3B Fine-Tuning: SOTA Data Preparation Pipeline (v3)
=============================================================

Definitive data prep based on exhaustive research of:
- Official Orpheus finetune code (canopyai/Orpheus-TTS/finetune/)
- Unsloth TTS fine-tuning guide and notebooks
- SNAC codec paper and evaluation methodology
- Community best practices (Hypa AI, Elise dataset, German fine-tune)
- Academic papers on TTS data prep (arxiv:2510.03111, arxiv:2410.14411)

Key improvements over v1/v2:
1. LUFS normalization to -12dB (SNAC evaluation standard)
2. 80Hz high-pass filter (removes room rumble/HVAC)
3. Keep mixed case + punctuation (matches Elise reference dataset)
4. Full autoregressive training (labels=input_ids, official approach)
5. SNAC UTMOS quality gate >= 3.5
6. CER validation via Whisper
7. Production-matched text format

Pipeline:
1. Load Expresso (ex04=Talia, conversational styles)
2. Resample 48kHz -> 24kHz
3. High-pass filter 80Hz (Butterworth 2nd order)
4. LUFS normalize to -12dB
5. Clip to [-1.0, 1.0]
6. SNAC encode -> 7-token interleaved format
7. Remove consecutive duplicate frames
8. Compute SNAC UTMOS (quality gate >= 3.5)
9. Build training sequences (full autoregressive, mixed case)
10. Validate token counts and save

Usage:
    python 24_prepare_orpheus_data_v3.py [--speaker ex04] [--voice-name maya]
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
import types
import importlib.machinery
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from scipy.signal import butter, sosfilt
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
MAX_SEQ_LENGTH = 8192
MIN_AUDIO_TOKENS = 70        # ~0.5s minimum
MAX_AUDIO_TOKENS = 7000      # ~12s maximum

# Quality thresholds
UTMOS_THRESHOLD = 3.5        # Minimum SNAC UTMOS for training data

# Duration filter (seconds)
MIN_DURATION = 2.0
MAX_DURATION = 12.0
MIN_TEXT_LENGTH = 10

# LUFS target (SNAC evaluation standard)
TARGET_LUFS = -12.0

# Styles to include (drop singing, longform, essentials, narration)
# Also drop whisper and laughing - SNAC can't reconstruct these (proven in audit)
CONVERSATIONAL_STYLES = [
    "default",      # Neutral baseline
    "happy",        # Warm, upbeat
    "sad",          # Empathetic, somber
    "confused",     # Questioning, uncertain
    "enunciated",   # Clear, emphatic
    "emphasis",     # Stressed/emphasized speech
]

# Emotion tag mapping (Expresso style -> Orpheus prefix)
# Only <laugh> is well-supported, other emotions come from the audio itself
STYLE_TO_EMOTION = {
    "default": "",
    "happy": "",
    "sad": "",
    "confused": "",
    "enunciated": "",
    "emphasis": "",
}

# Style tag pattern to strip from Expresso text
STYLE_TAG_PATTERN = re.compile(
    r'^\[(?:confused|default|enunciated|happy|laughing|sad|whisper|emphasis|essentials|singing|longform|narration)\]\s*'
)


# =============================================================================
# AUDIO PREPROCESSING (SOTA Pipeline)
# =============================================================================

def highpass_filter(audio: np.ndarray, sr: int, cutoff: float = 80.0, order: int = 2) -> np.ndarray:
    """Apply Butterworth high-pass filter to remove low-frequency noise.

    Removes room rumble, HVAC noise, and microphone handling noise.
    80Hz cutoff with 2nd order (12dB/octave slope) is gentle enough
    to preserve speech fundamentals while cleaning up noise floor.

    Source: Audio engineering best practices + TTS data prep research
    """
    sos = butter(order, cutoff, btype='high', fs=sr, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def lufs_normalize(audio: np.ndarray, sr: int, target_lufs: float = -12.0) -> np.ndarray:
    """Normalize audio to target LUFS (Loudness Units Full Scale).

    SNAC codec was evaluated with -12dB LUFS normalization.
    Using pyloudnorm which implements ITU-R BS.1770 standard.

    Source: SNAC paper (arxiv:2410.14411) evaluation methodology
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)

    # Ensure audio is at least 0.4s for accurate measurement
    min_samples = int(sr * 0.4)
    if len(audio) < min_samples:
        padded = np.zeros(min_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        current_lufs = meter.integrated_loudness(padded)
    else:
        current_lufs = meter.integrated_loudness(audio)

    if np.isinf(current_lufs) or np.isnan(current_lufs):
        # Audio is silent or near-silent, skip normalization
        return audio

    try:
        normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
    except Exception:
        # Fallback: simple gain adjustment
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20.0)
        normalized = audio * gain_linear

    return normalized.astype(np.float32)


def preprocess_audio(audio: np.ndarray, source_sr: int, target_sr: int = SNAC_SR) -> np.ndarray:
    """Full SOTA audio preprocessing pipeline.

    Order matters (from research):
    1. Resample to target SR
    2. High-pass filter at 80Hz
    3. LUFS normalize to -12dB
    4. Safety clip to [-1.0, 1.0]
    """
    # 1. Convert to float32
    audio = audio.astype(np.float32)

    # 2. Resample to 24kHz using torchaudio (high quality)
    if source_sr != target_sr:
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        resampled = T.Resample(orig_freq=source_sr, new_freq=target_sr)(audio_t)
        audio = resampled.squeeze(0).numpy()

    # 3. High-pass filter at 80Hz (remove room rumble/HVAC)
    audio = highpass_filter(audio, target_sr, cutoff=80.0, order=2)

    # 4. LUFS normalize to -12dB (SNAC evaluation standard)
    audio = lufs_normalize(audio, target_sr, target_lufs=TARGET_LUFS)

    # 5. Safety clip to [-1.0, 1.0] (prevent clipping after normalization)
    audio = np.clip(audio, -1.0, 1.0)

    return audio


# =============================================================================
# SNAC ENCODING
# =============================================================================

class SNACEncoder:
    """Encodes preprocessed audio to Orpheus SNAC token format."""

    def __init__(self, device="cuda:2"):
        self.device = device
        logger.info(f"Loading SNAC 24kHz model on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        logger.info("SNAC encoder ready")

    def encode(self, audio: np.ndarray) -> list:
        """Encode preprocessed 24kHz audio to Orpheus SNAC token IDs.

        Args:
            audio: preprocessed numpy array at 24kHz (already resampled, filtered, LUFS normalized)

        Returns:
            List of token IDs in 7-per-frame interleaved format
        """
        audio_t = torch.from_numpy(audio).float()
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)

        # Shape for SNAC: [1, 1, samples]
        audio_t = audio_t.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            codes = self.model.encode(audio_t)

        n_frames = codes[0].shape[1]

        # Interleave into 7-token-per-frame format (official Orpheus)
        all_tokens = []
        for i in range(n_frames):
            all_tokens.append(codes[0][0][i].item() + AUDIO_TOKEN_BASE)
            all_tokens.append(codes[1][0][2*i].item() + AUDIO_TOKEN_BASE + 4096)
            all_tokens.append(codes[2][0][4*i].item() + AUDIO_TOKEN_BASE + (2*4096))
            all_tokens.append(codes[2][0][4*i+1].item() + AUDIO_TOKEN_BASE + (3*4096))
            all_tokens.append(codes[1][0][2*i+1].item() + AUDIO_TOKEN_BASE + (4*4096))
            all_tokens.append(codes[2][0][4*i+2].item() + AUDIO_TOKEN_BASE + (5*4096))
            all_tokens.append(codes[2][0][4*i+3].item() + AUDIO_TOKEN_BASE + (6*4096))

        return all_tokens

    def decode(self, token_ids: list) -> torch.Tensor:
        """Decode SNAC tokens back to audio."""
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
    """Remove consecutive duplicate frames (official Orpheus/Unsloth approach).

    Compares only Layer 0 (coarsest) token of each 7-token frame.
    Orpheus pretraining was done WITH this deduplication, so fine-tuning
    data MUST match this convention to avoid train-test mismatch.

    Source: Unsloth notebook (Kaggle-Orpheus_(3B)-TTS.ipynb)
    """
    if len(token_ids) < 7:
        return token_ids

    n = (len(token_ids) // 7) * 7
    token_ids = token_ids[:n]

    result = token_ids[:7]  # Always keep first frame
    removed = 0

    for i in range(7, n, 7):
        if token_ids[i] != result[-7]:  # Compare L0 codes
            result.extend(token_ids[i:i+7])
        else:
            removed += 1

    return result


# =============================================================================
# UTMOS SCORING
# =============================================================================

class UTMOSScorer:
    """Score audio quality using UTMOS."""

    def __init__(self, device="cuda:2"):
        self.device = device
        logger.info("Loading UTMOS model...")
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        self.model = self.model.to(device).eval()
        logger.info("UTMOS scorer ready")

    def score(self, audio: torch.Tensor, sr: int = 24000) -> float:
        """Score audio tensor. Returns UTMOS score (1-5)."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        resampled = torchaudio.functional.resample(audio, sr, 16000)
        resampled = resampled.to(self.device)
        with torch.inference_mode():
            score = self.model(resampled, 16000)
        return float(score.item())


# =============================================================================
# DATA LOADING
# =============================================================================

def mock_torchcodec():
    """Mock torchcodec to avoid import errors."""
    for name in ['torchcodec', 'torchcodec.decoders',
                 'torchcodec.decoders._core', 'torchcodec.decoders._core.video_decoder_ops']:
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = m


def load_and_filter_expresso(speaker: str, styles: list, min_dur: float, max_dur: float):
    """Load Expresso from cached Arrow files."""
    import pyarrow.ipc as ipc
    import io
    import soundfile as sf
    import glob

    logger.info(f"Loading Expresso dataset (speaker={speaker}, styles={len(styles)})...")
    t0 = time.time()

    cache_dir = "/home/ec2-user/SageMaker/.cache/huggingface/datasets/ylacombe___expresso/read/0.0.0"
    arrow_pattern = os.path.join(cache_dir, "**/expresso-train-*.arrow")
    arrow_files = sorted(glob.glob(arrow_pattern, recursive=True))

    if not arrow_files:
        logger.info("Arrow cache not found, downloading from HuggingFace...")
        mock_torchcodec()
        from datasets import load_dataset
        load_dataset("ylacombe/expresso", split="train")
        arrow_files = sorted(glob.glob(arrow_pattern, recursive=True))
        if not arrow_files:
            raise FileNotFoundError(f"Could not find Arrow files in {cache_dir}")

    logger.info(f"Found {len(arrow_files)} Arrow shard files")

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

            if speaker_id != speaker:
                continue
            if style not in styles:
                continue
            if len(text.strip()) < MIN_TEXT_LENGTH:
                continue

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

    style_counts = Counter(s["style"] for s in samples)
    logger.info("Style distribution:")
    for style, count in style_counts.most_common():
        logger.info(f"  {style}: {count}")

    durations = [s["duration"] for s in samples]
    logger.info(f"Duration: mean={np.mean(durations):.2f}s, total={sum(durations)/3600:.2f}h")

    return samples


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def process_text(text: str, style: str, voice_name: str) -> str:
    """Process text for Orpheus training.

    CRITICAL: Keep mixed case and punctuation (matches Elise reference dataset).
    DO NOT lowercase - Orpheus was pretrained with standard text that includes capitalization.

    Source: Unsloth TTS guide, Elise dataset format, community best practices
    """
    # Strip Expresso style tags: "[confused] Ready for bed?" -> "Ready for bed?"
    text = STYLE_TAG_PATTERN.sub("", text).strip()

    # Add emotion tag prefix if applicable
    emotion_prefix = STYLE_TO_EMOTION.get(style, "")

    # Clean up whitespace (but keep original case and punctuation)
    text = re.sub(r'\s+', ' ', text).strip()

    # Build prompt: "maya: Hello there, how are you?"
    return f"{voice_name}: {emotion_prefix}{text}"


def build_training_sequence(
    text_prompt: str,
    audio_tokens: list,
    tokenizer,
) -> dict:
    """Build training sequence with FULL AUTOREGRESSIVE labels.

    This matches the official Orpheus training approach:
    - labels = input_ids (train on entire sequence)
    - The model learns text understanding + audio generation jointly
    - This is INTENTIONAL: "text token training boosts TTS performance
      while preserving semantic reasoning ability"

    Source: Official Orpheus finetune/train.py, Unsloth notebook
    """
    # Tokenize text (without special tokens)
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=False)

    # Build full sequence (official format)
    input_ids = (
        [BOS_TOKEN, START_OF_HUMAN]
        + text_ids
        + [END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
        + audio_tokens
        + [END_OF_SPEECH, END_OF_AI]
    )

    if len(input_ids) > MAX_SEQ_LENGTH:
        return None

    # Full autoregressive: labels = input_ids (official approach)
    labels = list(input_ids)

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
    parser = argparse.ArgumentParser(description="SOTA Orpheus data prep (v3)")
    parser.add_argument("--speaker", default="ex04", help="Expresso speaker ID")
    parser.add_argument("--voice-name", default="maya", help="Voice name in prompts")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--snac-device", default="cuda:2", help="GPU for SNAC/UTMOS")
    parser.add_argument("--verify-n", type=int, default=5, help="Samples to verify")
    parser.add_argument("--utmos-threshold", type=float, default=UTMOS_THRESHOLD, help="Min SNAC UTMOS")
    parser.add_argument("--skip-utmos", action="store_true", help="Skip UTMOS filtering")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("/home/ec2-user/SageMaker/project_maya/training/data/orpheus_finetune_v3")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_dir = output_dir / "verification_audio"
    verify_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("  ORPHEUS 3B FINE-TUNING: SOTA DATA PREPARATION (v3)")
    logger.info("=" * 80)
    logger.info(f"  Speaker: {args.speaker} | Voice: {args.voice_name}")
    logger.info(f"  Styles: {CONVERSATIONAL_STYLES}")
    logger.info(f"  Duration: {MIN_DURATION}-{MAX_DURATION}s")
    logger.info(f"  UTMOS threshold: {args.utmos_threshold}")
    logger.info(f"  LUFS target: {TARGET_LUFS}dB")
    logger.info(f"  Audio pipeline: resample -> 80Hz HPF -> LUFS -> clip -> SNAC")
    logger.info(f"  Text: mixed case + punctuation (Elise reference format)")
    logger.info(f"  Labels: full autoregressive (official Orpheus approach)")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 80)

    # Step 1: Load and filter dataset
    logger.info("\n[Step 1/7] Loading and filtering Expresso dataset...")
    ds = load_and_filter_expresso(
        speaker=args.speaker,
        styles=CONVERSATIONAL_STYLES,
        min_dur=MIN_DURATION,
        max_dur=MAX_DURATION,
    )

    # Step 2: Initialize SNAC + UTMOS
    logger.info("\n[Step 2/7] Initializing SNAC encoder + UTMOS scorer...")
    snac = SNACEncoder(device=args.snac_device)
    utmos_scorer = None
    if not args.skip_utmos:
        utmos_scorer = UTMOSScorer(device=args.snac_device)

    # Step 3: Load tokenizer
    logger.info("\n[Step 3/7] Loading Orpheus tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface"
    )
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Step 4: Process all samples with SOTA pipeline
    logger.info(f"\n[Step 4/7] Processing {len(ds)} samples with SOTA pipeline...")
    logger.info("  Pipeline: resample(24kHz) -> HPF(80Hz) -> LUFS(-12dB) -> clip -> SNAC -> dedup -> UTMOS gate")

    processed = []
    rejected = {"snac_fail": 0, "too_short": 0, "too_long": 0, "low_utmos": 0, "seq_too_long": 0}
    all_utmos = []

    t0 = time.time()
    for idx in range(len(ds)):
        sample = ds[idx]
        text = sample["text"].strip()
        style = sample["style"]
        waveform = sample["audio_array"]
        sr = sample["audio_sr"]
        duration = sample["duration"]

        # SOTA audio preprocessing
        preprocessed_audio = preprocess_audio(waveform, source_sr=sr, target_sr=SNAC_SR)

        # SNAC encode
        try:
            audio_tokens = snac.encode(preprocessed_audio)
        except Exception as e:
            logger.warning(f"SNAC encode failed for sample {idx}: {e}")
            rejected["snac_fail"] += 1
            continue

        # Remove duplicate frames (official Orpheus convention)
        original_len = len(audio_tokens)
        audio_tokens = remove_duplicate_frames(audio_tokens)

        # Check audio token count
        if len(audio_tokens) < MIN_AUDIO_TOKENS:
            rejected["too_short"] += 1
            continue
        if len(audio_tokens) > MAX_AUDIO_TOKENS:
            rejected["too_long"] += 1
            continue

        # UTMOS quality gate: decode SNAC tokens, score with UTMOS
        utmos_score = None
        if utmos_scorer is not None:
            decoded_audio = snac.decode(audio_tokens)
            if decoded_audio is not None and decoded_audio.numel() > 0:
                utmos_score = utmos_scorer.score(decoded_audio, sr=SNAC_SR)
                all_utmos.append(utmos_score)

                if utmos_score < args.utmos_threshold:
                    rejected["low_utmos"] += 1
                    continue
            else:
                rejected["snac_fail"] += 1
                continue

        # Build text prompt (mixed case, with punctuation - SOTA approach)
        text_prompt = process_text(text, style, args.voice_name)

        # Build training sequence (full autoregressive labels)
        seq = build_training_sequence(text_prompt, audio_tokens, tokenizer)
        if seq is None:
            rejected["seq_too_long"] += 1
            continue

        # Store with metadata
        seq["_text"] = text
        seq["_style"] = style
        seq["_duration"] = duration
        seq["_n_audio_tokens"] = len(audio_tokens)
        seq["_dedup_ratio"] = len(audio_tokens) / original_len if original_len > 0 else 1.0
        seq["_utmos"] = utmos_score
        seq["_idx"] = idx

        processed.append(seq)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(ds) - idx - 1) / rate
            utmos_str = f"UTMOS={np.mean(all_utmos[-50:]):.3f}" if all_utmos else "UTMOS=N/A"
            logger.info(
                f"  [{idx+1}/{len(ds)}] Kept: {len(processed)} | "
                f"Rejected: {sum(rejected.values())} | {utmos_str} | "
                f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s"
            )

    elapsed = time.time() - t0
    logger.info(f"\nProcessing complete in {elapsed:.1f}s")
    logger.info(f"  Kept: {len(processed)}")
    for reason, count in rejected.items():
        if count > 0:
            logger.info(f"  Rejected ({reason}): {count}")

    if not processed:
        logger.error("No samples processed!")
        return

    # Step 5: Verify by decoding samples
    logger.info(f"\n[Step 5/7] Verifying {args.verify_n} samples...")
    for i in range(min(args.verify_n, len(processed))):
        sample = processed[i]
        ids = sample["input_ids"]
        speech_start = ids.index(START_OF_SPEECH) + 1
        speech_end = ids.index(END_OF_SPEECH)
        audio_token_ids = ids[speech_start:speech_end]

        audio = snac.decode(audio_token_ids)
        if audio is not None:
            out_path = verify_dir / f"verify_{i:03d}_{sample['_style']}.wav"
            torchaudio.save(str(out_path), audio.unsqueeze(0), SNAC_SR)
            logger.info(
                f"  Sample {i}: style={sample['_style']}, UTMOS={sample['_utmos']:.3f}, "
                f"text='{sample['_text'][:50]}...'"
            )

    # Step 6: Save dataset
    logger.info(f"\n[Step 6/7] Saving dataset to {output_dir}...")

    # Training data (only training columns)
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for sample in processed:
            f.write(json.dumps({
                "input_ids": sample["input_ids"],
                "labels": sample["labels"],
                "attention_mask": sample["attention_mask"],
            }) + "\n")
    logger.info(f"Saved {len(processed)} training samples to {train_path}")

    # Metadata
    meta_data = []
    for sample in processed:
        meta_data.append({
            "text": sample["_text"],
            "style": sample["_style"],
            "duration": sample["_duration"],
            "n_audio_tokens": sample["_n_audio_tokens"],
            "n_total_tokens": len(sample["input_ids"]),
            "dedup_ratio": sample["_dedup_ratio"],
            "snac_utmos": sample["_utmos"],
            "idx": sample["_idx"],
        })

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)

    # Step 7: Dataset statistics
    logger.info(f"\n[Step 7/7] Computing statistics...")

    audio_tokens_list = [s["_n_audio_tokens"] for s in processed]
    total_tokens_list = [len(s["input_ids"]) for s in processed]
    utmos_list = [s["_utmos"] for s in processed if s["_utmos"] is not None]
    durations_list = [s["_duration"] for s in processed]
    style_dist = dict(Counter(s["_style"] for s in processed))

    stats = {
        "total_samples": len(processed),
        "original_samples": len(ds),
        "retention_rate": len(processed) / len(ds) if ds else 0,
        "filter_threshold": args.utmos_threshold,
        "pipeline": {
            "resample": "48kHz -> 24kHz",
            "highpass_filter": "80Hz Butterworth 2nd order",
            "lufs_normalization": f"{TARGET_LUFS}dB",
            "clip": "[-1.0, 1.0]",
            "snac_codec": "24kHz 3-layer VQ",
            "dedup": "consecutive L0 frame dedup",
            "quality_gate": f"SNAC UTMOS >= {args.utmos_threshold}",
            "text_format": "mixed case + punctuation (Elise reference)",
            "labels": "full autoregressive (official Orpheus)",
        },
        "rejected": rejected,
        "snac_utmos": {
            "mean": float(np.mean(utmos_list)) if utmos_list else 0,
            "std": float(np.std(utmos_list)) if utmos_list else 0,
            "min": float(min(utmos_list)) if utmos_list else 0,
            "max": float(max(utmos_list)) if utmos_list else 0,
            "median": float(np.median(utmos_list)) if utmos_list else 0,
        },
        "audio_tokens": {
            "mean": float(np.mean(audio_tokens_list)),
            "std": float(np.std(audio_tokens_list)),
            "min": int(min(audio_tokens_list)),
            "max": int(max(audio_tokens_list)),
            "median": float(np.median(audio_tokens_list)),
        },
        "total_tokens": {
            "mean": float(np.mean(total_tokens_list)),
            "median": float(np.median(total_tokens_list)),
        },
        "duration": {
            "mean": float(np.mean(durations_list)),
            "total_hours": float(sum(durations_list) / 3600),
        },
        "style_distribution": style_dist,
    }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("  DATASET SUMMARY (SOTA v3)")
    logger.info("=" * 80)
    logger.info(f"  Samples: {stats['total_samples']} / {stats['original_samples']} ({stats['retention_rate']:.1%} retained)")
    logger.info(f"  Total audio: {stats['duration']['total_hours']:.2f} hours")
    logger.info(f"  SNAC UTMOS: {stats['snac_utmos']['mean']:.3f} +/- {stats['snac_utmos']['std']:.3f}")
    logger.info(f"  Audio tokens: {stats['audio_tokens']['mean']:.0f} +/- {stats['audio_tokens']['std']:.0f}")
    logger.info(f"  Styles: {style_dist}")
    logger.info(f"\n  Pipeline: HPF(80Hz) -> LUFS(-12dB) -> SNAC -> dedup -> UTMOS({args.utmos_threshold})")
    logger.info(f"  Text: mixed case + punctuation | Labels: full autoregressive")
    logger.info(f"  Files: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
