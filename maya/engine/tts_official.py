"""
Official CSM Generator TTS - EXACT same approach as natural test samples.

Uses the OFFICIAL CSM Generator class with fine-tuned model and training
data context. This is PROVEN to produce natural-sounding audio.

Key principles:
1. Use official Generator (not custom streaming)
2. Load fine-tuned model (same as test script)
3. Use training data as context (same as test script)
4. Temperature 0.8, topk 50 (same as test script)
5. Stream output for perceived responsiveness
"""

import torch
import logging
import time
import sys
import os

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

from typing import Optional, List, Generator as PyGenerator
from dataclasses import dataclass
from generator import Generator, Segment
from models import Model

from ..config import TTS as TTS_CONFIG

logger = logging.getLogger(__name__)


class OfficialTTSEngine:
    """
    Production TTS using EXACT same approach as test_finetuned_v2.py.

    This produces natural-sounding audio because it:
    1. Uses official Generator class
    2. Loads fine-tuned model correctly
    3. Uses training data context
    4. Same temperature/topk as working test
    """

    SAMPLE_RATE = 24000
    FIRST_CHUNK_FRAMES = 0  # Not applicable - uses batch generation

    def __init__(self):
        self._generator: Optional[Generator] = None
        self._initialized = False
        self._voice_context: List[Segment] = []
        self._device = f"cuda:{TTS_CONFIG.device_index}"

    def initialize(self) -> None:
        """Initialize exactly like test script."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("LOADING OFFICIAL CSM (test script approach)")
        logger.info("=" * 60)

        start = time.time()
        torch.cuda.set_device(TTS_CONFIG.device_index)

        # Load model EXACTLY like test script
        FINETUNED_MODEL = '/home/ec2-user/SageMaker/project_maya/training/checkpoints/csm_maya_correct/best_model/model_merged.pt'

        logger.info("Loading CSM model...")
        model = Model.from_pretrained("sesame/csm-1b")

        if os.path.exists(FINETUNED_MODEL):
            logger.info("  Loading fine-tuned weights...")
            state_dict = torch.load(FINETUNED_MODEL, map_location="cuda", weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
            logger.info("  Fine-tuned weights loaded")

        model.to(device=self._device, dtype=torch.bfloat16)
        model.eval()

        # Create official Generator
        logger.info("Creating official Generator...")
        self._generator = Generator(model)

        # Load voice context from training data (like test script)
        logger.info("Loading voice context...")
        self._load_voice_context()

        # Warmup
        logger.info("Warming up...")
        for i in range(5):
            start_w = time.time()
            _ = self._generator.generate(
                text="hello how are you",
                speaker=0,
                context=self._voice_context[:2] if self._voice_context else [],
                max_audio_length_ms=3000,
                temperature=0.8,
                topk=50
            )
            logger.info(f"  Warmup {i+1}/5: {(time.time()-start_w)*1000:.0f}ms")

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"OFFICIAL CSM READY in {elapsed:.1f}s")
        logger.info(f"  Voice context: {len(self._voice_context)} segments")
        logger.info("=" * 60)

        self._initialized = True

    def _load_voice_context(self) -> None:
        """Load voice context EXACTLY like test script."""
        import json
        import torchaudio

        TRAINING_DATA = '/home/ec2-user/SageMaker/project_maya/training/data/csm_ready_ex04'
        train_json = os.path.join(TRAINING_DATA, 'train.json')

        if not os.path.exists(train_json):
            logger.warning("Training data not found")
            return

        with open(train_json) as f:
            train_samples = json.load(f)

        # Get default style samples (same as test script)
        default_samples = [s for s in train_samples if s.get("style") == "default"][:5]
        if not default_samples:
            default_samples = train_samples[:5]

        total_duration = 0
        for sample in default_samples:
            if total_duration >= 10:
                break

            audio_path = os.path.join(TRAINING_DATA, sample["path"])
            if not os.path.exists(audio_path):
                continue

            audio, sr = torchaudio.load(audio_path)
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)
            audio = audio.mean(dim=0) if audio.dim() > 1 and audio.size(0) > 1 else audio.squeeze(0)

            text = sample["text"]
            if text.startswith("["):
                text = text.split("]", 1)[-1].strip()

            self._voice_context.append(Segment(
                speaker=0,
                text=text,
                audio=audio.cpu()
            ))

            total_duration += sample.get("duration", len(audio) / 24000)
            logger.info(f"  Context: '{text[:40]}...'")

    def generate_stream(
        self,
        text: str,
        use_context: bool = True
    ) -> PyGenerator[torch.Tensor, None, None]:
        """Generate audio and stream in chunks."""
        if not self._initialized:
            self.initialize()

        if not text or not text.strip():
            return

        # Minimal preprocessing (test script uses text as-is, lowercase)
        text = text.lower().strip()

        start_time = time.time()

        # Generate using official Generator (EXACTLY like test script)
        context = self._voice_context if use_context else []

        audio = self._generator.generate(
            text=text,
            speaker=0,
            context=context,
            max_audio_length_ms=8000,
            temperature=0.8,  # Same as test script
            topk=50,          # Same as test script
        )

        gen_time = (time.time() - start_time) * 1000
        audio_ms = len(audio) / self.SAMPLE_RATE * 1000
        logger.info(f">>> FIRST CHUNK at {gen_time:.0f}ms ({audio_ms:.0f}ms audio) <<<")

        # Normalize
        audio = audio - audio.mean()
        peak = audio.abs().max()
        if peak > 1e-6:
            audio = audio * (0.7 / peak)

        # Stream in 500ms chunks
        chunk_size = self.SAMPLE_RATE // 2
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) > 0:
                yield chunk.to(self._device)

        logger.info(f"TTS complete: {audio_ms:.0f}ms audio in {(time.time()-start_time)*1000:.0f}ms")

    def generate(self, text: str, use_context: bool = True) -> torch.Tensor:
        """Generate complete audio."""
        chunks = list(self.generate_stream(text, use_context))
        if not chunks:
            return torch.zeros(int(self.SAMPLE_RATE * 0.5), device=self._device)
        return torch.cat(chunks)

    def add_context(self, text: str, audio: torch.Tensor, is_user: bool = False) -> None:
        """Context is fixed to training data for consistent quality."""
        pass

    def clear_context(self) -> None:
        """Voice context is preserved."""
        pass

    @property
    def is_initialized(self) -> bool:
        return self._initialized
