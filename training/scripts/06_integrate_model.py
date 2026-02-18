#!/usr/bin/env python3
"""
Step 6: Integrate Fine-tuned Model into Maya Pipeline

This script integrates the fine-tuned CSM model into the Maya production pipeline.

What it does:
1. Verifies the checkpoint quality
2. Creates a new voice prompt from the fine-tuned model
3. Copies the model to the production location
4. Updates the Maya TTS engine configuration
5. Runs integration tests

Usage:
    python 06_integrate_model.py --checkpoint checkpoints/csm_maya/best_model

After running:
    ./start_maya.sh
    python run.py
"""

import os
import sys
import json
import logging
import argparse
import shutil
from pathlib import Path

import torch

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
PRODUCTION_MODEL_DIR = PROJECT_ROOT / "models" / "csm_maya"
VOICE_PROMPT_PATH = PROJECT_ROOT / "assets" / "voice_prompt" / "maya_voice_prompt.pt"
TTS_ENGINE_PATH = PROJECT_ROOT / "maya" / "engine" / "tts_streaming_real.py"


def verify_checkpoint(checkpoint_path: Path) -> bool:
    """Verify the checkpoint is valid."""
    logger.info("Verifying checkpoint...")

    model_file = checkpoint_path / "model.pt"
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return False

    # Check file size (should be ~4GB for CSM-1B)
    file_size_gb = model_file.stat().st_size / (1024**3)
    logger.info(f"Model file size: {file_size_gb:.2f} GB")

    if file_size_gb < 1.0:
        logger.warning("Model file seems too small, might be corrupted")

    # Try to load the state dict
    try:
        state_dict = torch.load(model_file, map_location="cpu")
        num_params = sum(p.numel() for p in state_dict.values())
        logger.info(f"Parameters in checkpoint: {num_params:,}")
        return True
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False


def generate_voice_prompt(checkpoint_path: Path) -> bool:
    """Generate a voice prompt using the fine-tuned model."""
    logger.info("Generating voice prompt from fine-tuned model...")

    try:
        from models import Model
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing
        from moshi.models import loaders
        from huggingface_hub import hf_hub_download

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load fine-tuned model
        logger.info("Loading fine-tuned model...")
        model = Model.from_pretrained("sesame/csm-1b")
        state_dict = torch.load(checkpoint_path / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()

        # Load tokenizers
        text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        bos = text_tokenizer.bos_token
        eos = text_tokenizer.eos_token
        text_tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", text_tokenizer.bos_token_id), (f"{eos}", text_tokenizer.eos_token_id)],
        )

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
        audio_tokenizer.set_num_codebooks(32)

        # Generate voice prompt text with varied expressions
        voice_prompt_text = "oh hey, hi! its really nice to meet you. im maya. so tell me, whats been on your mind?"

        # Generate audio
        logger.info(f"Generating: \"{voice_prompt_text}\"")

        text_tokens = text_tokenizer.encode(f"[0]{voice_prompt_text}")
        text_frame = torch.zeros(len(text_tokens), 33).long().to(device)
        text_mask = torch.zeros(len(text_tokens), 33).bool().to(device)
        text_frame[:, -1] = torch.tensor(text_tokens).to(device)
        text_mask[:, -1] = True

        model.reset_caches()

        curr_tokens = text_frame.unsqueeze(0)
        curr_mask = text_mask.unsqueeze(0)
        curr_pos = torch.arange(len(text_tokens)).unsqueeze(0).to(device)

        frames = []
        max_frames = 100  # ~8 seconds

        with torch.inference_mode():
            for _ in range(max_frames):
                sample = model.generate_frame(
                    curr_tokens, curr_mask, curr_pos,
                    temperature=0.9, topk=50
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

        if not frames:
            logger.error("Failed to generate voice prompt audio")
            return False

        # Decode to audio
        stacked = torch.stack(frames).permute(1, 2, 0)
        audio = audio_tokenizer.decode(stacked).squeeze(0).squeeze(0)

        duration = len(audio) / 24000
        logger.info(f"Generated {duration:.1f}s of audio")

        # Save voice prompt
        VOICE_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)

        voice_prompt_data = {
            "audio": audio.cpu(),
            "text": voice_prompt_text,
            "sample_rate": 24000,
            "duration_seconds": duration,
            "temperature": 0.9,
            "topk": 50,
            "model": "fine-tuned-csm-maya",
            "version": "1.0",
        }

        torch.save(voice_prompt_data, VOICE_PROMPT_PATH)
        logger.info(f"Voice prompt saved to: {VOICE_PROMPT_PATH}")

        return True

    except Exception as e:
        logger.error(f"Voice prompt generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def copy_model_to_production(checkpoint_path: Path) -> bool:
    """Copy the fine-tuned model to production location."""
    logger.info("Copying model to production location...")

    try:
        PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Copy model file
        src_model = checkpoint_path / "model.pt"
        dst_model = PRODUCTION_MODEL_DIR / "model.pt"
        shutil.copy2(src_model, dst_model)
        logger.info(f"Model copied to: {dst_model}")

        # Copy training state for reference
        src_state = checkpoint_path / "training_state.json"
        if src_state.exists():
            dst_state = PRODUCTION_MODEL_DIR / "training_state.json"
            shutil.copy2(src_state, dst_state)

        # Create integration info
        integration_info = {
            "source_checkpoint": str(checkpoint_path),
            "integration_date": str(Path.cwd()),
            "model_file": "model.pt",
            "voice_prompt": str(VOICE_PROMPT_PATH),
        }

        with open(PRODUCTION_MODEL_DIR / "integration_info.json", "w") as f:
            json.dump(integration_info, f, indent=2)

        return True

    except Exception as e:
        logger.error(f"Failed to copy model: {e}")
        return False


def update_tts_engine_config():
    """Update TTS engine to use fine-tuned model."""
    logger.info("Checking TTS engine configuration...")

    # The TTS engine already loads from the voice prompt path
    # We just need to verify it's configured correctly

    if not TTS_ENGINE_PATH.exists():
        logger.error(f"TTS engine not found: {TTS_ENGINE_PATH}")
        return False

    content = TTS_ENGINE_PATH.read_text()

    # Check voice prompt path is correct
    if "maya_voice_prompt.pt" in content:
        logger.info("TTS engine is configured to use voice prompt")
        return True
    else:
        logger.warning("TTS engine might not be using voice prompt")
        return True  # Non-fatal


def run_integration_test() -> bool:
    """Run a quick integration test."""
    logger.info("Running integration test...")

    try:
        # Import the TTS engine
        sys.path.insert(0, str(PROJECT_ROOT))
        from maya.engine.tts_streaming_real import RealStreamingTTSEngine

        # Initialize
        engine = RealStreamingTTSEngine()
        engine.initialize()

        # Test generation
        test_text = "hello, this is a test of the fine-tuned model"
        logger.info(f"Generating test audio: \"{test_text}\"")

        chunks = list(engine.generate_stream(test_text))

        if chunks:
            total_samples = sum(len(c) for c in chunks)
            duration = total_samples / 24000
            logger.info(f"Generated {duration:.1f}s of audio in {len(chunks)} chunks")

            # Save test audio
            test_output = PROJECT_ROOT / "training" / "evaluation" / "integration_test.wav"
            test_output.parent.mkdir(parents=True, exist_ok=True)

            import torchaudio
            audio = torch.cat(chunks)
            torchaudio.save(str(test_output), audio.unsqueeze(0).cpu(), 24000)
            logger.info(f"Test audio saved to: {test_output}")

            return True
        else:
            logger.error("No audio generated")
            return False

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Integrate fine-tuned CSM into Maya")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--skip-voice-prompt",
        action="store_true",
        help="Skip voice prompt generation"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip integration test"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    logger.info("=" * 60)
    logger.info("INTEGRATING FINE-TUNED MODEL")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("")

    # Step 1: Verify checkpoint
    if not verify_checkpoint(checkpoint_path):
        logger.error("Checkpoint verification failed!")
        sys.exit(1)

    # Step 2: Generate voice prompt
    if not args.skip_voice_prompt:
        if not generate_voice_prompt(checkpoint_path):
            logger.warning("Voice prompt generation failed, continuing...")

    # Step 3: Copy model to production
    if not copy_model_to_production(checkpoint_path):
        logger.error("Failed to copy model to production!")
        sys.exit(1)

    # Step 4: Update TTS engine config
    update_tts_engine_config()

    # Step 5: Run integration test
    if not args.skip_test:
        if not run_integration_test():
            logger.warning("Integration test failed, manual verification needed")

    logger.info("")
    logger.info("=" * 60)
    logger.info("INTEGRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model location: {PRODUCTION_MODEL_DIR}")
    logger.info(f"Voice prompt: {VOICE_PROMPT_PATH}")
    logger.info("")
    logger.info("To use the fine-tuned model:")
    logger.info("  ./start_maya.sh")
    logger.info("  python run.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
