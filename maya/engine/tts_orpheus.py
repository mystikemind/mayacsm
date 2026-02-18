"""
Orpheus 3B TTS Engine for Maya Pipeline

High-performance TTS using Orpheus 3B via llama.cpp GGUF backend.
Benchmark: RTF=0.64 @ 129 tok/s, UTMOS 4.345 (Q4_K_M on A10G)

Architecture:
- llama.cpp server runs Orpheus GGUF model on dedicated GPU
- HTTP API for token generation (streaming + non-streaming)
- SNAC decoder converts tokens to 24kHz audio
- Native emotion tags: <laugh>, <sigh>, <gasp>, <chuckle>

Token format (official Orpheus):
- Special tokens as custom_token_N (N = token_id - 128256):
  - custom_token_1 = START_OF_SPEECH (128257)
  - custom_token_2 = END_OF_SPEECH (128258)
  - custom_token_3 = START_OF_HUMAN (128259)
  - custom_token_4 = END_OF_HUMAN (128260)
  - custom_token_5 = START_OF_AI (128261)
- Audio tokens: custom_token_10 through custom_token_28683
  (token_ids 128266-156937, 7 codebook levels * 4096 codes)
- Frame layout: [L0, L1+4096, L2+8192, L2+12288, L1+16384, L2+20480, L2+24576]

Speed benchmarks on A10G:
  HF Transformers BF16: RTF=2.08, 40 tok/s, 7.7GB VRAM
  llama.cpp Q8_0:       RTF=0.86, 95 tok/s, 3.3GB VRAM
  llama.cpp Q4_K_M:     RTF=0.64, 129 tok/s, ~2.3GB VRAM
"""

import torch
import asyncio
import logging
import time
import re
import json
import subprocess
import signal
import os
from typing import Optional, AsyncIterator, List
from pathlib import Path
from threading import Thread
from snac import SNAC

from ..config import TTS, AUDIO, DEVICES

logger = logging.getLogger(__name__)

# SNAC token constants (from official Orpheus implementation)
AUDIO_TOKEN_BASE = 128266  # 128256 + 10

# Fade settings for smooth chunk boundaries
# Each SNAC chunk is decoded independently, creating waveform discontinuities.
# Short fades (5ms) eliminate clicks without audible artifacts.
# 50ms was too long and caused volume dips between chunks.
FADE_SAMPLES = 120  # 5ms at 24kHz


def _apply_fade_in(chunk: torch.Tensor, fade_samples: int = FADE_SAMPLES) -> torch.Tensor:
    """Apply a fade-in to the beginning of an audio chunk."""
    if chunk.numel() < fade_samples * 2:
        return chunk
    fade = torch.linspace(0.0, 1.0, fade_samples)
    result = chunk.clone()
    result[:fade_samples] = result[:fade_samples] * fade
    return result


def _apply_fade_out(chunk: torch.Tensor, fade_samples: int = FADE_SAMPLES) -> torch.Tensor:
    """Apply a fade-out to the end of an audio chunk."""
    if chunk.numel() < fade_samples * 2:
        return chunk
    fade = torch.linspace(1.0, 0.0, fade_samples)
    result = chunk.clone()
    result[-fade_samples:] = result[-fade_samples:] * fade
    return result


# Natural release duration: 200ms cosine fade prevents "dropped call" feeling.
# Human speech naturally trails off over 150-300ms. 200ms is the sweet spot:
# - Long enough to sound natural (not an abrupt stop)
# - Short enough to not audibly affect the last syllable
NATURAL_RELEASE_MS = 200
NATURAL_RELEASE_SAMPLES = int(24000 * NATURAL_RELEASE_MS / 1000)  # 4800 samples


def _apply_natural_release(audio: torch.Tensor, sample_rate: int = 24000,
                           duration_ms: int = NATURAL_RELEASE_MS) -> torch.Tensor:
    """Apply natural speech release fade-out using cosine curve.

    Prevents the 'dropped call' feeling by creating a smooth, natural
    ending that mimics how human speech naturally trails off.

    Cosine curve advantages over linear/squared:
    - Gentle initial slope: preserves the tail of the last syllable
    - Smooth midpoint: no perceptible inflection during fade
    - Rapid final approach: reaches silence cleanly, no lingering artifacts

    This is ONLY for final audio endings (end of speech, last streaming chunk).
    NOT for inter-chunk boundaries (those use 5ms linear fades).
    """
    fade_samples = min(int(sample_rate * duration_ms / 1000), audio.numel() // 2)
    if fade_samples < 50:
        return audio

    t = torch.linspace(0, 1, fade_samples)
    # Cosine fade: 1→0 with smooth S-curve shape
    fade = 0.5 * (1 + torch.cos(t * torch.pi))

    result = audio.clone()
    result[-fade_samples:] = result[-fade_samples:] * fade
    return result
SNAC_CODEBOOK_SIZE = 4096
CUSTOM_TOKEN_OFFSET = 128256

# Valid audio token range
AUDIO_TOKEN_MIN = AUDIO_TOKEN_BASE
AUDIO_TOKEN_MAX = AUDIO_TOKEN_BASE + (7 * SNAC_CODEBOOK_SIZE) - 1

# Default paths
DEFAULT_GGUF_DIR = "/home/ec2-user/SageMaker/.cache/huggingface/hub/models--QuantFactory--orpheus-3b-0.1-ft-GGUF"
DEFAULT_LLAMA_SERVER = "/home/ec2-user/SageMaker/llama.cpp/build/bin/llama-server"
LLAMA_LIB_PATH = "/home/ec2-user/SageMaker/llama.cpp/build/ggml/src:/home/ec2-user/SageMaker/llama.cpp/build/ggml/src/ggml-cuda"


def _find_gguf(quant: str = "Q4_K_M") -> Optional[str]:
    """Find a GGUF model file in the HF cache."""
    for snapshot_dir in Path(DEFAULT_GGUF_DIR).glob("snapshots/*"):
        for gguf in snapshot_dir.glob(f"*{quant}*"):
            # Resolve symlinks
            resolved = gguf.resolve()
            if resolved.exists():
                return str(gguf)
    return None


def _trim_trailing_audio(audio: torch.Tensor, sample_rate: int = 24000,
                         text: str = "") -> torch.Tensor:
    """
    Smart speech-end detection and trimming for Orpheus TTS output.

    The Orpheus model often generates audio beyond the intended speech:
    - [SPEECH] → [gap] → [babble] → [gap] → [babble] → ...
    This happens because the GGUF EOS token (128009) doesn't match
    Orpheus's END_OF_SPEECH token (128258).

    This function detects where the actual speech content ends using
    a multi-stage energy envelope analysis, then trims with a graceful
    fade-out to prevent abrupt endings.

    Algorithm:
    1. Compute RMS energy in 25ms overlapping windows
    2. Determine "speech energy level" from the main speech content
    3. Find the last sustained speech region (energy above threshold)
    4. Look for a natural low-energy cut point after that
    5. Apply 50ms fade-out for smooth ending

    Uses text word count as a sanity check: trim point shouldn't be
    earlier than expected speech duration (words / 4.0 seconds).
    """
    if audio.numel() < sample_rate * 0.3:
        return audio

    audio_np = audio.cpu().numpy() if audio.is_cuda else audio.numpy()
    total_samples = len(audio_np)
    total_duration = total_samples / sample_rate

    # Compute RMS energy in overlapping windows
    window_ms = 25
    window_samples = int(sample_rate * window_ms / 1000)
    hop_samples = window_samples // 2  # 50% overlap

    energies = []
    for i in range(0, total_samples - window_samples, hop_samples):
        rms = float(torch.sqrt(torch.mean(audio[i:i + window_samples] ** 2)).item())
        energies.append(rms)

    if len(energies) < 10:
        return audio

    import numpy as np
    energies = np.array(energies)

    # Determine speech energy level from the "core" speech region
    # Use the first 60% of audio (most likely contains the main speech)
    core_end = int(len(energies) * 0.6)
    core_energies = energies[:core_end]
    voiced_mask = core_energies > 0.008  # Above noise floor

    if voiced_mask.sum() < 5:
        return audio  # Too little voiced content to analyze

    speech_energy = np.percentile(core_energies[voiced_mask], 50)  # Median speech energy

    # Dynamic threshold: 12% of speech energy level
    # This catches babble (typically 30-70% of speech energy)
    # while not being triggered by natural speech dynamics
    drop_threshold = max(speech_energy * 0.12, 0.008)

    # Find regions of sustained speech vs gaps
    # A "gap" is 150ms+ below threshold
    gap_min_frames = int(0.15 / (hop_samples / sample_rate))

    # Scan forward to find the LAST sustained speech region
    # "Sustained" = at least 100ms above threshold
    sustained_min_frames = int(0.10 / (hop_samples / sample_rate))

    # Build a speech activity map
    is_speech = energies > drop_threshold

    # Find speech regions (consecutive True in is_speech)
    speech_regions = []
    in_speech = False
    region_start = 0

    for i, active in enumerate(is_speech):
        if active and not in_speech:
            region_start = i
            in_speech = True
        elif not active and in_speech:
            if i - region_start >= sustained_min_frames:
                speech_regions.append((region_start, i))
            in_speech = False

    if in_speech:
        if len(is_speech) - region_start >= sustained_min_frames:
            speech_regions.append((region_start, len(is_speech)))

    if not speech_regions:
        return audio

    # The speech content is likely in the first few regions
    # After the main speech, any new regions after a long gap are likely babble
    # "Long gap" = 250ms+ of below-threshold energy
    long_gap_frames = int(0.25 / (hop_samples / sample_rate))

    # Find where babble starts: first long gap after the initial speech content
    # starts that's followed by more audio
    last_speech_region_end = speech_regions[0][1]

    for i in range(1, len(speech_regions)):
        gap = speech_regions[i][0] - speech_regions[i - 1][1]
        if gap >= long_gap_frames:
            # Long gap found. The speech before this gap is the "real" content.
            # But only if we've had enough speech content already
            speech_duration_so_far = speech_regions[i - 1][1] * hop_samples / sample_rate
            # Minimum speech check: at least 0.3s of speech before we consider trimming
            if speech_duration_so_far > 0.3:
                # Check word count: don't trim too early
                clean_text = re.sub(r'<\w+>', '', text).strip() if text else ""
                words = len(clean_text.split()) if clean_text else 0
                min_speech_s = max(words / 4.0, 0.4)  # At least words/4s (generous)
                if speech_duration_so_far >= min_speech_s:
                    last_speech_region_end = speech_regions[i - 1][1]
                    break
        last_speech_region_end = speech_regions[i][1]

    # Convert frame index back to sample position
    speech_end_sample = last_speech_region_end * hop_samples + window_samples

    # Duration-based fallback: if gap detection found that speech goes to the end
    # but audio significantly exceeds expected speech duration, force-find a cut point.
    # This handles continuous babble (no gaps) after speech.
    clean_text = re.sub(r'<\w+>', '', text).strip() if text else ""
    words = len(clean_text.split()) if clean_text else 0
    emotion_tags = len(re.findall(r'<\w+>', text)) if text else 0
    expected_speech_s = max(words / 3.0, 0.5) + emotion_tags * 0.5
    max_reasonable_s = expected_speech_s * 2.0  # 2x expected is the max reasonable

    if speech_end_sample >= total_samples - int(sample_rate * 0.3):
        # Gap detection didn't find an earlier speech end.
        # Check if audio significantly exceeds expected duration.
        if total_duration > max_reasonable_s and max_reasonable_s < total_duration - 0.2:
            # Force-find the best cut point after expected speech end.
            # Scan from expected_speech_s to end, find lowest energy 50ms window.
            force_search_start = int(expected_speech_s * sample_rate)
            force_search_end = total_samples

            if force_search_end > force_search_start + window_samples:
                min_energy = float('inf')
                best_cut = force_search_start
                scan_window = int(sample_rate * 0.05)  # 50ms windows for precision
                for i in range(force_search_start, force_search_end - scan_window,
                               scan_window // 2):
                    rms = float(torch.sqrt(
                        torch.mean(audio[i:i + scan_window] ** 2)
                    ).item())
                    if rms < min_energy:
                        min_energy = rms
                        best_cut = i

                speech_end_sample = best_cut
                logger.debug(
                    f"Duration fallback: force-cut at {best_cut/sample_rate:.2f}s "
                    f"(expected speech: {expected_speech_s:.1f}s, "
                    f"total: {total_duration:.1f}s, energy={min_energy:.4f})"
                )

    # Look for the best cut point: lowest energy in a 200ms window after speech end
    search_start = speech_end_sample
    search_end = min(speech_end_sample + int(sample_rate * 0.2), total_samples)

    if search_end > search_start + window_samples:
        min_energy = float('inf')
        best_cut = speech_end_sample
        for i in range(search_start, search_end - window_samples, hop_samples):
            rms = float(torch.sqrt(torch.mean(audio[i:i + window_samples] ** 2)).item())
            if rms < min_energy:
                min_energy = rms
                best_cut = i
        trim_pos = best_cut
    else:
        trim_pos = speech_end_sample

    # Add 80ms of natural tail
    trim_pos = min(trim_pos + int(sample_rate * 0.08), total_samples)

    # Don't trim if we'd only remove < 150ms (not worth it)
    if total_samples - trim_pos < int(sample_rate * 0.15):
        return audio

    # Don't trim if it would make the audio too short (< 0.3s)
    if trim_pos < int(sample_rate * 0.3):
        return audio

    # Apply natural release: 200ms cosine fade for smooth ending
    # This is the key fix for the "dropped call" feeling.
    # The trim point is after speech content, so the fade is in the
    # post-speech tail region and won't affect the last word.
    trimmed = audio[:trim_pos].clone()
    trimmed = _apply_natural_release(trimmed, sample_rate)

    trimmed_duration = trim_pos / sample_rate
    removed_duration = total_duration - trimmed_duration
    if removed_duration > 0.2:
        logger.debug(
            f"Trimmed {removed_duration:.1f}s trailing audio "
            f"({total_duration:.1f}s → {trimmed_duration:.1f}s)"
        )

    return trimmed


class OrpheusTTSEngine:
    """
    Orpheus 3B TTS Engine with llama.cpp backend.

    Uses llama-server for optimized inference (129 tok/s Q4_K_M on A10G).
    Manages the llama-server lifecycle automatically.
    Drop-in replacement for StreamingTTSEngine.
    """

    SAMPLE_RATE = 24000
    MODEL_ID = "canopylabs/orpheus-3b-0.1-ft"

    def __init__(
        self,
        device: Optional[str] = None,
        gguf_quant: Optional[str] = None,
        server_port: Optional[int] = None,
    ):
        self._snac = None
        self._device = device
        self._gguf_quant = gguf_quant or TTS.gguf_quant
        self._server_port = server_port or TTS.server_port
        self._server_url = f"http://127.0.0.1:{self._server_port}"
        self._server_process = None
        self._initialized = False
        self._voice_ref_audio = None
        self._voice_ref_text = None
        self._context: List = []
        self._session = None  # requests.Session for connection pooling

    def initialize(self, device: Optional[str] = None) -> None:
        """Load SNAC codec and start llama-server."""
        if self._initialized:
            return

        start = time.time()

        if device:
            self._device = device
        if self._device is None:
            self._device = f"cuda:{DEVICES.tts_gpu}" if torch.cuda.is_available() else "cpu"

        # Extract GPU index for llama-server
        if "cuda:" in self._device:
            self._gpu_index = int(self._device.split(":")[-1])
        else:
            self._gpu_index = 0

        device_obj = torch.device(self._device)
        logger.info(f"Initializing Orpheus TTS on {self._device}...")

        # Load SNAC codec (~100MB)
        logger.info("Loading SNAC 24kHz codec...")
        self._snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device_obj).eval()

        # Start llama-server
        self._start_server()

        # Setup HTTP session for connection pooling
        import requests
        self._session = requests.Session()

        elapsed = time.time() - start
        logger.info(f"Orpheus TTS ready ({self._gguf_quant}) in {elapsed:.1f}s")
        self._initialized = True

    def _start_server(self) -> None:
        """Start llama-server as a subprocess."""
        gguf_path = _find_gguf(self._gguf_quant)
        if not gguf_path:
            raise FileNotFoundError(
                f"No GGUF model found for {self._gguf_quant}. "
                f"Download with: huggingface-cli download QuantFactory/orpheus-3b-0.1-ft-GGUF "
                f"orpheus-3b-0.1-ft.{self._gguf_quant}.gguf"
            )

        server_bin = DEFAULT_LLAMA_SERVER
        if not Path(server_bin).exists():
            raise FileNotFoundError(
                f"llama-server not found at {server_bin}. "
                f"Build it: cd /home/ec2-user/SageMaker/llama.cpp && mkdir build && cd build && "
                f"cmake .. -DGGML_CUDA=ON && cmake --build . -j4"
            )

        # Check if server already running on this port
        try:
            import requests
            r = requests.get(f"{self._server_url}/health", timeout=2)
            if r.status_code == 200:
                logger.info(f"llama-server already running on port {self._server_port}")
                return
        except Exception:
            pass

        logger.info(f"Starting llama-server: {gguf_path} on GPU {self._gpu_index}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self._gpu_index)
        env["LD_LIBRARY_PATH"] = LLAMA_LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

        cmd = [
            server_bin,
            "-m", gguf_path,
            "-c", "4096",
            "-ngl", "99",
            "--host", "127.0.0.1",
            "--port", str(self._server_port),
            "-fa", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "-np", "1",  # Single slot for lowest latency
            "--mlock",   # Lock model in memory for stable inference
        ]

        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # Wait for server to be ready
        import requests
        for attempt in range(60):  # 60 seconds max
            time.sleep(1)
            try:
                r = requests.get(f"{self._server_url}/health", timeout=2)
                if r.status_code == 200:
                    logger.info(f"llama-server ready (PID {self._server_process.pid})")
                    return
            except Exception:
                pass

            # Check if process died
            if self._server_process.poll() is not None:
                stdout = self._server_process.stdout.read().decode() if self._server_process.stdout else ""
                raise RuntimeError(f"llama-server exited with code {self._server_process.returncode}: {stdout[-500:]}")

        raise TimeoutError("llama-server failed to start within 60 seconds")

    def _stop_server(self) -> None:
        """Stop the llama-server subprocess."""
        if self._server_process and self._server_process.poll() is None:
            logger.info("Stopping llama-server...")
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                self._server_process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._server_process = None

    def __del__(self):
        self._stop_server()

    def _extract_audio_tokens(self, text_output: str) -> list:
        """Extract audio token IDs from llama.cpp text output.

        llama.cpp outputs extended vocab tokens as <custom_token_N>.
        Audio tokens: N in [10, 28683] => token_id in [128266, 156937].
        """
        token_ids = []
        for match in re.finditer(r'<custom_token_(\d+)>', text_output):
            custom_num = int(match.group(1))
            token_id = CUSTOM_TOKEN_OFFSET + custom_num
            if AUDIO_TOKEN_MIN <= token_id <= AUDIO_TOKEN_MAX:
                token_ids.append(token_id)
        return token_ids

    def _decode_snac_frames(self, token_ids: list) -> Optional[torch.Tensor]:
        """Decode audio token IDs to waveform via SNAC.

        Groups tokens into 7-token frames and decodes all at once.
        """
        n = (len(token_ids) // 7) * 7
        if n < 7:
            return None
        token_ids = token_ids[:n]

        codes = [t - AUDIO_TOKEN_BASE for t in token_ids]
        l0, l1, l2 = [], [], []
        for i in range(n // 7):
            b = 7 * i
            l0.append(max(0, min(4095, codes[b])))
            l1.append(max(0, min(4095, codes[b + 1] - 4096)))
            l2.append(max(0, min(4095, codes[b + 2] - 2 * 4096)))
            l2.append(max(0, min(4095, codes[b + 3] - 3 * 4096)))
            l1.append(max(0, min(4095, codes[b + 4] - 4 * 4096)))
            l2.append(max(0, min(4095, codes[b + 5] - 5 * 4096)))
            l2.append(max(0, min(4095, codes[b + 6] - 6 * 4096)))

        device = next(self._snac.parameters()).device
        snac_codes = [
            torch.tensor(l0, dtype=torch.int32).unsqueeze(0).to(device),
            torch.tensor(l1, dtype=torch.int32).unsqueeze(0).to(device),
            torch.tensor(l2, dtype=torch.int32).unsqueeze(0).to(device),
        ]

        with torch.inference_mode():
            audio = self._snac.decode(snac_codes)

        return audio.squeeze()

    # SNAC sliding window: number of frames to carry over between streaming chunks.
    # SNAC's decoder has convolutions that create inter-frame dependencies.
    # Decoding with left context produces smoother boundaries than independent chunks.
    # 1 frame = 7 tokens = 2048 audio samples = 85ms.
    SNAC_OVERLAP_FRAMES = 1

    def _decode_snac_with_context(
        self, context_tokens: list, new_tokens: list
    ) -> Optional[torch.Tensor]:
        """Decode new SNAC frames with left context for boundary continuity.

        Instead of decoding each streaming chunk independently (which creates
        waveform discontinuities), this method decodes [context + new] together
        and returns only the new portion. The SNAC decoder's convolutions see
        the left context, producing smoother audio at the chunk boundary.

        Based on Orpheus-FastAPI sliding window technique.
        """
        all_tokens = context_tokens + new_tokens
        full_audio = self._decode_snac_frames(all_tokens)
        if full_audio is None:
            return None

        # Skip the context portion to return only new audio
        context_frames = len(context_tokens) // 7
        if context_frames > 0:
            # Each SNAC frame = 2048 audio samples at 24kHz
            context_samples = context_frames * 2048
            if context_samples < full_audio.numel():
                return full_audio[context_samples:]
            else:
                return full_audio

        return full_audio

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for Orpheus.

        - Strip emotion tags (<laugh>, <sigh>, etc.) - they cause UTMOS drops to 1.3-2.9
          The model was designed to infer emotion from context, not from tags.
          Only ~336 laugh examples in 100k+ hours of training data = unreliable.
        - Lowercase for consistent prosody
        - Remove trailing periods (cause unnatural falling intonation on short phrases)
        - Keep question marks (drive important intonation contours)
        - Keep exclamation marks (emotional emphasis)
        - Normalize whitespace
        """
        # Strip any angle-bracket emotion/action tags (safety net)
        text = re.sub(r'<[^>]+>', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove trailing periods (but not ? or !) for natural conversational prosody
        text = re.sub(r'\.$', '', text).strip()
        return text

    def _build_prompt(self, text: str) -> str:
        """Build the Orpheus prompt for llama.cpp.

        Format: <|begin_of_text|><custom_token_3>voice: text<custom_token_4><custom_token_5>
        Where custom_token_3=START_OF_HUMAN, 4=END_OF_HUMAN, 5=START_OF_AI.
        Voice name provides consistent speaker identity (tara=#1 conversational realism).
        """
        voice = TTS.voice
        return f"<|begin_of_text|><custom_token_3>{voice}: {text}<custom_token_4><custom_token_5>"

    def _estimate_max_tokens(self, text: str) -> int:
        """Estimate appropriate max_tokens based on text length.

        The Orpheus GGUF has EOS=128009 (Llama 3's eot_id), NOT 128258
        (END_OF_SPEECH). The model frequently fails to generate END_OF_SPEECH,
        causing audio to run until max_tokens. Dynamic budgeting prevents this.

        Audio math:
        - 7 SNAC tokens = 1 frame = 2048 samples = 85.3ms at 24kHz
        - ~82 tokens per second of audio
        - Conversational speech: ~3 words/second
        - Emotion tags (<laugh>, <sigh>) add ~0.5s each

        Strategy: generous enough to avoid mid-word cuts, but bounded
        to prevent excessive babble. Post-generation trimming (in
        _trim_trailing_audio) handles the precise speech-end detection.

        Observed from production data:
        - 13 words → 4.2s (343 tokens) with natural stop
        - 11 words → 4.9s (399 tokens) with natural stop
        - 9 words → 6.1s (499 tokens) WITHOUT stop (babble)
        """
        # Strip emotion tags for word count but account for their duration
        clean = re.sub(r'<\w+>', '', text).strip()
        words = len(clean.split()) if clean else 0
        emotion_tags = len(re.findall(r'<\w+>', text))

        # Base duration: words / speaking_rate
        estimated_seconds = max(words / 3.0, 0.5)  # At least 0.5s for very short phrases
        estimated_seconds += emotion_tags * 0.5  # Each emotion tag adds ~0.5s

        # Tiered safety margin: tighter for longer text (babble risk is higher)
        # Short (1-3 words): 2.5x - model needs room for natural delivery
        # Medium (4-8 words): 2.0x - more predictable duration
        # Long (9+ words): 1.8x - very predictable, babble is the bigger risk
        if words <= 3:
            margin = 2.5
        elif words <= 8:
            margin = 2.0
        else:
            margin = 1.8
        estimated_seconds *= margin

        # Convert to tokens: ~82 tokens per second + 20 overhead
        max_tokens = int(estimated_seconds * 82) + 20

        # Clamp to reasonable range
        # Min: 80 tokens (~1s) - enough for shortest utterances
        # Max: 420 tokens (~5.1s) - enough for 15-word responses
        # Post-generation trimming handles precise end detection
        max_tokens = max(max_tokens, 80)
        max_tokens = min(max_tokens, 420)

        return max_tokens

    def _build_sampling_params(self, max_tokens: Optional[int] = None, stream: bool = False) -> dict:
        """Build optimized sampling parameters for Orpheus token generation.

        Research-backed parameters:
        - repeat_last_n=64: Prevents penalty from corrupting valid SNAC patterns
          (OuteTTS community: full-window penalty breaks audio)
        - min_p=0.05: Adaptive probability floor prevents garbage tokens
        - top_k=50: Hard ceiling on token diversity
        """
        return {
            "max_tokens": max_tokens or TTS.max_tokens,
            "temperature": TTS.temperature,
            "top_p": TTS.top_p,
            "top_k": TTS.top_k,
            "min_p": TTS.min_p,
            "repeat_penalty": TTS.repeat_penalty,
            "repeat_last_n": TTS.repeat_last_n,
            "stop": ["<custom_token_2>"],  # END_OF_SPEECH
            "stream": stream,
        }

    def _generate_tokens_via_server(
        self, text: str, max_tokens: Optional[int] = None
    ) -> tuple:
        """Generate all tokens via llama-server (non-streaming).

        Returns:
            (token_ids, finish_reason) where finish_reason is "stop" (natural
            END_OF_SPEECH) or "length" (hit max_tokens).
        """
        prompt = self._build_prompt(text)

        # Use dynamic max_tokens if not explicitly provided
        if max_tokens is None:
            max_tokens = self._estimate_max_tokens(text)

        payload = {"prompt": prompt, **self._build_sampling_params(max_tokens, stream=False)}

        resp = self._session.post(
            f"{self._server_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()

        data = resp.json()
        text_output = data["choices"][0]["text"]
        finish_reason = data["choices"][0].get("finish_reason", "length")
        return self._extract_audio_tokens(text_output), finish_reason

    def _generate_tokens_streaming(self, text: str, max_tokens: Optional[int] = None):
        """Generate tokens via llama-server with streaming.

        Yields audio token IDs as they arrive.
        Uses dynamic max_tokens based on text length to prevent runaway generation.
        """
        prompt = self._build_prompt(text)

        # Use dynamic max_tokens if not explicitly provided
        if max_tokens is None:
            max_tokens = self._estimate_max_tokens(text)

        payload = {"prompt": prompt, **self._build_sampling_params(max_tokens, stream=True)}

        resp = self._session.post(
            f"{self._server_url}/v1/completions",
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        buffer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                token_text = data["choices"][0].get("text", "")
                buffer += token_text

                # Extract any complete custom_token patterns
                while True:
                    match = re.search(r'<custom_token_(\d+)>', buffer)
                    if not match:
                        break
                    custom_num = int(match.group(1))
                    token_id = CUSTOM_TOKEN_OFFSET + custom_num
                    if AUDIO_TOKEN_MIN <= token_id <= AUDIO_TOKEN_MAX:
                        yield token_id
                    # Remove the matched pattern from buffer
                    buffer = buffer[match.end():]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def _estimate_max_audio_samples(self, text: str) -> int:
        """Estimate maximum audio samples for speech-end detection in streaming.

        Returns the sample count beyond which generated audio is likely babble.
        Uses a generous estimate to avoid cutting real speech, but bounded
        to prevent excessive post-speech content.

        This is the streaming counterpart to _trim_trailing_audio (non-streaming).
        """
        clean = re.sub(r'<\w+>', '', text).strip()
        words = len(clean.split()) if clean else 0
        emotion_tags = len(re.findall(r'<\w+>', text))

        # Expected speech duration
        speech_seconds = max(words / 3.0, 0.5)
        speech_seconds += emotion_tags * 0.5

        # 2.0x margin: generous enough for natural delivery + breathing pauses
        # but not so generous that 3 seconds of babble gets through
        max_seconds = speech_seconds * 2.0

        # Absolute minimum: 1.2s (very short phrases still need some room)
        max_seconds = max(max_seconds, 1.2)

        return int(max_seconds * self.SAMPLE_RATE)

    async def generate_stream(
        self,
        text: str,
        use_context: bool = True,
        config=None,
    ) -> AsyncIterator[torch.Tensor]:
        """
        Async streaming audio generation with speech-end detection.

        Yields audio chunks as they're generated. Monitors cumulative audio
        duration and energy to detect when speech has ended. Stops early if
        post-speech babble is detected, preventing excessive audio output.

        Compatible with seamless_orchestrator.py interface.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        first_chunk_time = None
        total_audio_samples = 0

        text = self.preprocess_text(text)
        dynamic_max = self._estimate_max_tokens(text)
        max_audio_samples = self._estimate_max_audio_samples(text)
        logger.debug(
            f"Orpheus streaming: '{text[:50]}...' "
            f"(max_tokens={dynamic_max}, max_audio={max_audio_samples/self.SAMPLE_RATE:.1f}s)"
        )

        loop = asyncio.get_event_loop()
        chunk_queue = asyncio.Queue()
        generation_done = asyncio.Event()
        # Signal to stop token generation from the async loop
        stop_generation = asyncio.Event()

        def generate_sync():
            try:
                first_chunk = True
                audio_buffer = []
                frame_buffer = []
                cumulative_samples = 0
                speech_energy_sum = 0.0
                speech_energy_count = 0
                # SNAC sliding window: carry over last frame's tokens
                # for decoder continuity (Orpheus-FastAPI technique)
                context_tokens = []

                for token_id in self._generate_tokens_streaming(text):
                    # Check if async loop requested stop (speech end detected)
                    if stop_generation.is_set():
                        logger.debug("Streaming: stop requested by speech-end detector")
                        break

                    audio_buffer.append(token_id)

                    while len(audio_buffer) >= 7:
                        frame = audio_buffer[:7]
                        audio_buffer = audio_buffer[7:]
                        frame_buffer.append(frame)

                        # First chunk: 2 frames (~150ms audio) for fast first response
                        # Later: 6 frames (~450ms) for better SNAC quality
                        threshold = 2 if first_chunk else 6
                        if len(frame_buffer) >= threshold:
                            new_tokens = [t for f in frame_buffer for t in f]

                            # Decode with left context from previous chunk
                            # SNAC's convolutions create inter-frame dependencies;
                            # context improves the left boundary of each chunk
                            if context_tokens:
                                audio = self._decode_snac_with_context(
                                    context_tokens, new_tokens
                                )
                            else:
                                audio = self._decode_snac_frames(new_tokens)

                            if audio is not None and audio.numel() > 0:
                                chunk_cpu = audio.cpu()

                                # Track energy for speech-end detection
                                chunk_rms = torch.sqrt(torch.mean(chunk_cpu ** 2)).item()
                                if first_chunk or cumulative_samples < max_audio_samples * 0.5:
                                    # Track speech energy from the first half
                                    speech_energy_sum += chunk_rms
                                    speech_energy_count += 1

                                cumulative_samples += chunk_cpu.numel()

                                # Speech-end detection: if we're past max audio duration
                                # and energy has dropped significantly, stop
                                if cumulative_samples > max_audio_samples:
                                    avg_speech_energy = (
                                        speech_energy_sum / max(speech_energy_count, 1)
                                    )
                                    # If current chunk energy is < 25% of speech average,
                                    # we're in a gap/silence - good place to stop
                                    if chunk_rms < avg_speech_energy * 0.25:
                                        logger.info(
                                            f"Streaming speech-end: energy drop at "
                                            f"{cumulative_samples/self.SAMPLE_RATE:.1f}s "
                                            f"(rms={chunk_rms:.4f} < thresh={avg_speech_energy*0.25:.4f})"
                                        )
                                        # Natural release for smooth ending
                                        chunk_cpu = _apply_natural_release(
                                            chunk_cpu, self.SAMPLE_RATE
                                        )
                                        loop.call_soon_threadsafe(
                                            chunk_queue.put_nowait, chunk_cpu
                                        )
                                        loop.call_soon_threadsafe(stop_generation.set)
                                        return  # Exit generation
                                    # Also force-stop if way past limit (1.5x max)
                                    elif cumulative_samples > max_audio_samples * 1.5:
                                        logger.info(
                                            f"Streaming force-stop at "
                                            f"{cumulative_samples/self.SAMPLE_RATE:.1f}s "
                                            f"(1.5x max audio limit)"
                                        )
                                        chunk_cpu = _apply_natural_release(
                                            chunk_cpu, self.SAMPLE_RATE
                                        )
                                        loop.call_soon_threadsafe(
                                            chunk_queue.put_nowait, chunk_cpu
                                        )
                                        return

                                # Normal chunk: apply boundary fades
                                chunk_audio = _apply_fade_out(chunk_cpu)
                                loop.call_soon_threadsafe(
                                    chunk_queue.put_nowait, chunk_audio
                                )
                                first_chunk = False

                            # Save last frame as context for next decode
                            last_frame = frame_buffer[-self.SNAC_OVERLAP_FRAMES:]
                            context_tokens = [t for f in last_frame for t in f]
                            frame_buffer = []

                # Flush remaining frames (this is the LAST chunk)
                if frame_buffer:
                    new_tokens = [t for f in frame_buffer for t in f]
                    if context_tokens:
                        audio = self._decode_snac_with_context(
                            context_tokens, new_tokens
                        )
                    else:
                        audio = self._decode_snac_frames(new_tokens)
                    if audio is not None and audio.numel() > 0:
                        # Apply NATURAL RELEASE to last chunk (200ms cosine)
                        # This is the key anti-"dropped call" fix for streaming.
                        audio = _apply_natural_release(audio.cpu(), self.SAMPLE_RATE)
                        loop.call_soon_threadsafe(
                            chunk_queue.put_nowait, audio
                        )

            except Exception as e:
                logger.error(f"Orpheus generation error: {e}", exc_info=True)
            finally:
                loop.call_soon_threadsafe(generation_done.set)

        thread = Thread(target=generate_sync, daemon=True)
        thread.start()

        is_first_chunk = True
        chunk_count = 0

        while not generation_done.is_set() or not chunk_queue.empty():
            try:
                chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.05)
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    logger.info(
                        f">>> FIRST AUDIO at {first_chunk_time*1000:.0f}ms <<<"
                    )

                chunk_count += 1

                # Apply fade-in to non-first chunks for smooth transitions
                if not is_first_chunk:
                    chunk = _apply_fade_in(chunk)
                is_first_chunk = False

                total_audio_samples += chunk.shape[-1]
                yield chunk

            except asyncio.TimeoutError:
                if generation_done.is_set() and chunk_queue.empty():
                    break
                continue

        elapsed = time.time() - start_time
        duration = total_audio_samples / self.SAMPLE_RATE if total_audio_samples > 0 else 0
        rtf = elapsed / duration if duration > 0 else float('inf')
        fc_ms = first_chunk_time * 1000 if first_chunk_time else 0
        logger.info(
            f"Orpheus stream done: {duration:.1f}s audio in {elapsed:.1f}s "
            f"(RTF={rtf:.2f}, first_chunk={fc_ms:.0f}ms, chunks={chunk_count})"
        )

    def generate(self, text: str, use_context: bool = True) -> torch.Tensor:
        """Non-streaming generation with full-batch SNAC decoding.

        Handles both natural stops (END_OF_SPEECH) and forced stops (max_tokens):
        - Natural stop: trim trailing silence, minimal processing
        - Forced stop: smart speech-end detection + graceful fade-out
        """
        if not self._initialized:
            self.initialize()

        text = self.preprocess_text(text)

        token_ids, finish_reason = self._generate_tokens_via_server(text)
        audio = self._decode_snac_frames(token_ids)

        if audio is not None and audio.numel() > 0:
            audio = audio.cpu()

            # Smart speech-end detection: trims babble and applies graceful fade-out
            audio = _trim_trailing_audio(audio, text=text)

            # If max_tokens was hit (not natural stop), ALWAYS ensure a smooth ending.
            # The trimmer may not have found a good cut point if audio was
            # continuously voiced, so we apply a natural release as last resort.
            if finish_reason != "stop":
                tail_samples = min(int(self.SAMPLE_RATE * 0.05), audio.numel())
                tail_rms = torch.sqrt(
                    torch.mean(audio[-tail_samples:] ** 2)
                ).item() if tail_samples > 0 else 0

                # If tail is still loud (trimmer didn't handle it), apply natural release
                if tail_rms > 0.02:
                    audio = _apply_natural_release(audio, self.SAMPLE_RATE)
                    logger.debug(
                        f"Applied natural release fade-out "
                        f"(finish={finish_reason}, tail_rms={tail_rms:.4f})"
                    )

            return audio
        else:
            return torch.zeros(self.SAMPLE_RATE)

    def generate_fast(self, text: str, use_context: bool = True) -> torch.Tensor:
        """Fast generation for short utterances."""
        if not self._initialized:
            self.initialize()

        text = self.preprocess_text(text)

        # Use dynamic max_tokens, capped at fast limit
        dynamic_max = min(self._estimate_max_tokens(text), TTS.max_tokens_fast)
        token_ids, finish_reason = self._generate_tokens_via_server(text, max_tokens=dynamic_max)
        audio = self._decode_snac_frames(token_ids)

        if audio is not None and audio.numel() > 0:
            audio = audio.cpu()
            # Smart speech-end detection for clean endings
            audio = _trim_trailing_audio(audio, text=text)

            # Natural release if max_tokens hit and tail is still loud
            if finish_reason != "stop":
                tail_samples = min(int(self.SAMPLE_RATE * 0.05), audio.numel())
                tail_rms = torch.sqrt(
                    torch.mean(audio[-tail_samples:] ** 2)
                ).item() if tail_samples > 0 else 0
                if tail_rms > 0.02:
                    audio = _apply_natural_release(audio, self.SAMPLE_RATE)

            return audio
        else:
            return torch.zeros(self.SAMPLE_RATE)

    def set_voice_prompt(self, text: str, audio: torch.Tensor) -> None:
        """Set voice reference (stored for future voice-cloning support)."""
        self._voice_ref_text = text
        self._voice_ref_audio = audio
        logger.info(f"Voice reference stored ({audio.shape[-1]/self.SAMPLE_RATE:.1f}s)")

    def add_context(self, text: str, audio: torch.Tensor, is_user: bool = False) -> None:
        """Add conversation context (stored for future multi-turn support)."""
        self._context.append({"text": text, "audio": audio, "is_user": is_user})
        if len(self._context) > 3:
            self._context = self._context[-3:]

    def clear_context(self) -> None:
        """Clear all conversation context."""
        self._context = []

    def warmup(self) -> None:
        """Warm up the engine with a short generation."""
        if not self._initialized:
            self.initialize()

        logger.info("Warming up Orpheus TTS...")
        start = time.time()

        # Warmup llama-server (triggers CUDA kernel compilation)
        _ = self.generate_fast("hello")

        # Second warmup to stabilize
        _ = self.generate_fast("hey there")

        elapsed = time.time() - start
        logger.info(f"Orpheus warmup done in {elapsed:.1f}s")
