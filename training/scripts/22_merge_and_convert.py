#!/usr/bin/env python3
"""
Orpheus 3B: LoRA Merge + GGUF Conversion
==========================================

Post-training pipeline:
1. Merge LoRA adapter weights into base model
2. Convert merged model to GGUF format
3. Quantize to Q4_K with F16 output tensors (zero-failure quantization)
4. Benchmark against base model GGUF

Key insight (dahara1 research):
    Q4_K with --output-tensor-type f16 --token-embedding-type f16
    gives 0/1000 failures vs Q4_K_M's 16/1000 failures for audio tokens.

Usage:
    python 22_merge_and_convert.py --checkpoint training/checkpoints/orpheus_lora/final
    python 22_merge_and_convert.py --checkpoint training/checkpoints/orpheus_lora/checkpoint-300
"""

import os
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/.cache/huggingface"

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/ec2-user/SageMaker/project_maya")
LLAMA_CPP = Path("/home/ec2-user/SageMaker/llama.cpp")


def merge_lora(checkpoint_path: str, output_dir: str):
    """Merge LoRA adapter into base model."""
    logger.info(f"Loading base model (BF16)...")
    t0 = time.time()

    base_model = AutoModelForCausalLM.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
    )
    logger.info(f"Base model loaded in {time.time()-t0:.1f}s")

    logger.info(f"Loading LoRA adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "canopylabs/orpheus-3b-0.1-ft",
        cache_dir="/home/ec2-user/SageMaker/.cache/huggingface",
    )
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Merged model saved ({sum(p.numel() for p in model.parameters()):,} params)")
    del model, base_model
    torch.cuda.empty_cache()

    return output_dir


def convert_to_gguf(model_dir: str, output_path: str, quant_type: str = "q4_k"):
    """Convert HuggingFace model to GGUF format.

    Uses llama.cpp's convert_hf_to_gguf.py for F16 conversion,
    then quantize with Q4_K and F16 output/embedding tensors.
    """
    convert_script = LLAMA_CPP / "convert_hf_to_gguf.py"
    quantize_bin = LLAMA_CPP / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        logger.error(f"convert_hf_to_gguf.py not found: {convert_script}")
        return None
    if not quantize_bin.exists():
        logger.error(f"llama-quantize not found: {quantize_bin}")
        return None

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to F16 GGUF
    f16_path = str(output_dir / "orpheus-maya-f16.gguf")
    logger.info(f"Converting to F16 GGUF...")

    cmd = [
        sys.executable, str(convert_script),
        model_dir,
        "--outfile", f16_path,
        "--outtype", "f16",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(f"F16 conversion failed:\n{result.stderr}")
        return None
    logger.info(f"F16 GGUF created: {f16_path}")

    # Step 2: Quantize to Q4_K with F16 output tensors
    # This gives 0/1000 audio token failures (dahara1 research)
    logger.info(f"Quantizing to {quant_type.upper()} with F16 output tensors...")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_CPP / "build" / "ggml" / "src") + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(quantize_bin),
        f16_path,
        output_path,
        quant_type.upper(),
        "--output-tensor-type", "f16",
        "--token-embedding-type", "f16",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if result.returncode != 0:
        logger.error(f"Quantization failed:\n{result.stderr}")
        # Fall back to standard Q4_K_M
        logger.info("Falling back to standard Q4_K_M...")
        cmd_fallback = [
            str(quantize_bin),
            f16_path,
            output_path,
            "Q4_K_M",
        ]
        result = subprocess.run(cmd_fallback, capture_output=True, text=True, env=env, timeout=600)
        if result.returncode != 0:
            logger.error(f"Fallback quantization also failed:\n{result.stderr}")
            return None

    logger.info(f"Quantized GGUF created: {output_path}")

    # Clean up F16 intermediate
    f16_size = os.path.getsize(f16_path) / 1e9
    quant_size = os.path.getsize(output_path) / 1e9
    logger.info(f"  F16 size: {f16_size:.2f}GB → {quant_type}: {quant_size:.2f}GB")

    return output_path


def benchmark_gguf(gguf_path: str, gpu: int = 2, port: int = 5008):
    """Quick benchmark of the GGUF model via llama-server."""
    import requests

    llama_server = LLAMA_CPP / "build" / "bin" / "llama-server"
    if not llama_server.exists():
        logger.warning("llama-server not found, skipping benchmark")
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_CPP / "build" / "ggml" / "src") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    logger.info(f"Starting llama-server for benchmark (GPU {gpu}, port {port})...")

    # Start server
    proc = subprocess.Popen(
        [
            str(llama_server),
            "-m", gguf_path,
            "--port", str(port),
            "-ngl", "999",
            "-c", "8192",
            "--flash-attn",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        logger.warning("Server did not start in 30s, skipping benchmark")
        proc.kill()
        return

    # Quick test
    test_texts = [
        "yeah im doing pretty good, hows everything with you",
        "oh thats so cool, tell me more about that",
        "honestly i think thats a great idea",
    ]

    results = []
    for text in test_texts:
        prompt = f"<|begin_of_text|><custom_token_3>maya: {text}<custom_token_4><custom_token_5>"
        payload = {
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.6,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["<custom_token_2>"],
            "stream": False,
        }

        t0 = time.time()
        try:
            resp = requests.post(f"http://127.0.0.1:{port}/v1/completions", json=payload, timeout=120)
            gen_time = time.time() - t0
            data = resp.json()
            text_out = data["choices"][0]["text"]
            # Count audio tokens
            import re
            tokens = re.findall(r'<custom_token_\d+>', text_out)
            results.append({
                "text": text,
                "gen_time": gen_time,
                "n_tokens": len(tokens),
            })
            logger.info(f"  '{text[:40]}...' → {len(tokens)} tokens in {gen_time:.2f}s")
        except Exception as e:
            logger.warning(f"  Generation failed: {e}")

    # Stop server
    proc.kill()
    proc.wait()

    if results:
        avg_time = sum(r["gen_time"] for r in results) / len(results)
        avg_tokens = sum(r["n_tokens"] for r in results) / len(results)
        logger.info(f"  Avg: {avg_time:.2f}s, {avg_tokens:.0f} tokens, {avg_tokens/avg_time:.1f} tok/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to GGUF")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint directory")
    parser.add_argument("--quant", default="q4_k", help="Quantization type (default: q4_k)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--benchmark", action="store_true", help="Run quick benchmark")
    parser.add_argument("--benchmark-gpu", type=int, default=2, help="GPU for benchmark")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge (use existing merged)")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "training" / "models" / "orpheus_maya"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_dir = output_dir / "merged_hf"
    gguf_path = output_dir / f"orpheus-maya-{args.quant}.gguf"

    logger.info("=" * 80)
    logger.info("  ORPHEUS LoRA MERGE + GGUF CONVERSION")
    logger.info("=" * 80)
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"  Quant: {args.quant}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 80)

    # Step 1: Merge
    if not args.skip_merge:
        logger.info("\n[Step 1/3] Merging LoRA adapter into base model...")
        t0 = time.time()
        merge_lora(str(checkpoint), str(merged_dir))
        logger.info(f"Merge complete in {time.time()-t0:.1f}s")
    else:
        logger.info("\n[Step 1/3] Skipping merge (using existing)")

    # Step 2: Convert to GGUF
    logger.info(f"\n[Step 2/3] Converting to GGUF ({args.quant})...")
    t0 = time.time()
    result = convert_to_gguf(str(merged_dir), str(gguf_path), args.quant)
    if result:
        gguf_size = os.path.getsize(gguf_path) / 1e9
        logger.info(f"GGUF conversion complete in {time.time()-t0:.1f}s ({gguf_size:.2f}GB)")
    else:
        logger.error("GGUF conversion failed!")
        return

    # Step 3: Benchmark
    if args.benchmark:
        logger.info(f"\n[Step 3/3] Benchmarking GGUF on GPU {args.benchmark_gpu}...")
        benchmark_gguf(str(gguf_path), gpu=args.benchmark_gpu)
    else:
        logger.info("\n[Step 3/3] Skipping benchmark (use --benchmark to enable)")

    logger.info("\n" + "=" * 80)
    logger.info("  CONVERSION COMPLETE")
    logger.info(f"  GGUF: {gguf_path}")
    logger.info(f"  Size: {gguf_size:.2f}GB")
    logger.info("=" * 80)
    logger.info(f"\nTo deploy, copy GGUF to production and update config:")
    logger.info(f"  cp {gguf_path} /path/to/production/")
    logger.info(f"  Update maya/config.py TTS gguf_quant")


if __name__ == "__main__":
    main()
