#!/usr/bin/env python3
"""Test CUDA graph compatibility with CSM generate_frame.

CUDA graphs require static tensor shapes. After the first prompt processing,
generate_frame() uses fixed shapes:
- tokens: (1, 1, 33)
- tokens_mask: (1, 1, 33)
- input_pos: (1, 1)

This script tests:
1. Whether generate_frame works with reduce-overhead compilation
2. Whether manual CUDA graph capture works
3. Latency comparison: max-autotune vs reduce-overhead vs CUDA graph
"""
import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def load_model():
    """Load CSM-1B model with fine-tuned weights."""
    from models import Model
    from maya.config import TTS

    print("Loading CSM-1B model...")
    model = Model.from_pretrained("sesame/csm-1b")

    if os.path.exists(TTS.custom_model_path):
        print(f"Loading fine-tuned weights...")
        sd = torch.load(TTS.custom_model_path, map_location="cuda")
        valid_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in sd.items() if k in valid_keys}
        model.load_state_dict(filtered, strict=True)
        print(f"  Loaded {len(filtered)} keys")
        del sd, filtered

    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    model.setup_caches(1)
    return model


def create_test_inputs(model, device="cuda"):
    """Create test inputs matching single-frame generation shape."""
    # After first prompt, subsequent frames use these fixed shapes
    tokens = torch.zeros(1, 1, 33, dtype=torch.long, device=device)
    tokens_mask = torch.ones(1, 1, 33, dtype=torch.bool, device=device)
    input_pos = torch.tensor([[100]], dtype=torch.long, device=device)  # Any valid position
    return tokens, tokens_mask, input_pos


@torch.inference_mode()
def test_baseline(model, n_iters=50):
    """Baseline: no compilation."""
    print("\n--- Baseline (no compile) ---")
    tokens, mask, pos = create_test_inputs(model)

    # Warmup
    for _ in range(5):
        model.generate_frame(tokens, mask, pos, 0.9, 50)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate_frame(tokens, mask, pos, 0.9, 50)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"  Avg: {avg:.2f}ms | P50: {p50:.2f}ms | P95: {p95:.2f}ms")
    return avg


@torch.inference_mode()
def test_max_autotune(n_iters=50):
    """Test max-autotune compilation (current approach)."""
    print("\n--- max-autotune (current approach) ---")
    model = load_model()

    print("  Compiling backbone (max-autotune)...")
    model.backbone = torch.compile(model.backbone, mode='max-autotune', fullgraph=False)
    print("  Compiling decoder (max-autotune)...")
    model.decoder = torch.compile(model.decoder, mode='max-autotune', fullgraph=False)

    tokens, mask, pos = create_test_inputs(model)

    # Warmup (compile happens here)
    print("  Warming up...")
    for i in range(10):
        start = time.time()
        model.generate_frame(tokens, mask, pos, 0.9, 50)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        if i < 3:
            print(f"    Warmup {i+1}: {elapsed:.0f}ms")

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate_frame(tokens, mask, pos, 0.9, 50)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"  Avg: {avg:.2f}ms | P50: {p50:.2f}ms | P95: {p95:.2f}ms")

    del model
    torch.cuda.empty_cache()
    return avg


@torch.inference_mode()
def test_reduce_overhead(n_iters=50):
    """Test reduce-overhead compilation (CUDA graphs)."""
    print("\n--- reduce-overhead (CUDA graphs) ---")
    model = load_model()

    # Compile generate_frame directly with reduce-overhead
    print("  Compiling generate_frame (reduce-overhead)...")
    try:
        model.generate_frame = torch.compile(
            model.generate_frame,
            mode='reduce-overhead',
            fullgraph=False,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        del model
        torch.cuda.empty_cache()
        return None

    tokens, mask, pos = create_test_inputs(model)

    # Warmup (compile + CUDA graph capture)
    print("  Warming up (CUDA graph capture)...")
    for i in range(15):
        try:
            start = time.time()
            model.generate_frame(tokens, mask, pos, 0.9, 50)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            if i < 5:
                print(f"    Warmup {i+1}: {elapsed:.0f}ms")
        except Exception as e:
            print(f"    Warmup {i+1} FAILED: {e}")
            del model
            torch.cuda.empty_cache()
            return None

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate_frame(tokens, mask, pos, 0.9, 50)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"  Avg: {avg:.2f}ms | P50: {p50:.2f}ms | P95: {p95:.2f}ms")

    del model
    torch.cuda.empty_cache()
    return avg


@torch.inference_mode()
def test_dual_compile(n_iters=50):
    """Test: max-autotune on modules + reduce-overhead on generate_frame."""
    print("\n--- DUAL: max-autotune modules + reduce-overhead generate_frame ---")
    model = load_model()

    print("  Compiling backbone (max-autotune)...")
    model.backbone = torch.compile(model.backbone, mode='max-autotune', fullgraph=False)
    print("  Compiling decoder (max-autotune)...")
    model.decoder = torch.compile(model.decoder, mode='max-autotune', fullgraph=False)

    # Also compile generate_frame wrapper for CUDA graphs
    print("  Compiling generate_frame (reduce-overhead)...")
    try:
        model.generate_frame = torch.compile(
            model.generate_frame,
            mode='reduce-overhead',
            fullgraph=False,
        )
    except Exception as e:
        print(f"  generate_frame compile FAILED: {e}")
        del model
        torch.cuda.empty_cache()
        return None

    tokens, mask, pos = create_test_inputs(model)

    # Warmup
    print("  Warming up (may be slow - dual compilation)...")
    for i in range(15):
        try:
            start = time.time()
            model.generate_frame(tokens, mask, pos, 0.9, 50)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            if i < 5:
                print(f"    Warmup {i+1}: {elapsed:.0f}ms")
        except Exception as e:
            print(f"    Warmup {i+1} FAILED: {e}")
            del model
            torch.cuda.empty_cache()
            return None

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate_frame(tokens, mask, pos, 0.9, 50)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"  Avg: {avg:.2f}ms | P50: {p50:.2f}ms | P95: {p95:.2f}ms")

    del model
    torch.cuda.empty_cache()
    return avg


def main():
    print("=" * 60)
    print("CSM CUDA GRAPH COMPATIBILITY TEST")
    print("=" * 60)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    results = {}

    # Test 1: max-autotune (our current approach)
    results['max-autotune'] = test_max_autotune()

    # Test 2: reduce-overhead (CUDA graphs via generate_frame)
    results['reduce-overhead'] = test_reduce_overhead()

    # Test 3: dual compilation
    results['dual'] = test_dual_compile()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, avg in results.items():
        if avg is not None:
            print(f"  {name:25s}: {avg:.2f}ms per frame")
        else:
            print(f"  {name:25s}: FAILED")

    baseline = results.get('max-autotune')
    if baseline:
        print(f"\n  Baseline (max-autotune): {baseline:.2f}ms")
        for name, avg in results.items():
            if avg is not None and name != 'max-autotune':
                diff = baseline - avg
                pct = diff / baseline * 100
                print(f"  {name}: {avg:.2f}ms ({'+' if diff > 0 else ''}{diff:.2f}ms, {pct:+.1f}%)")


if __name__ == "__main__":
    main()
