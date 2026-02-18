#!/usr/bin/env python3
"""
Production Validation Test - Verify Sesame AI Level Performance

This test validates the full production pipeline with all optimizations:
1. GC disabled during inference
2. Multi-GPU architecture (TTS on GPU 1, vLLM on GPU 0)
3. Fine-tuned CSM model
4. Sesame voice prompt

Target: P50 < 200ms (Sesame AI level)
"""

import sys
import os
import time
import gc
from pathlib import Path

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
os.chdir('/home/ec2-user/SageMaker/project_maya')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np


def check_gpu_configuration():
    """Verify GPU configuration is correct."""
    print("=" * 70)
    print("GPU CONFIGURATION CHECK")
    print("=" * 70)

    num_gpus = torch.cuda.device_count()
    print(f"\n  Available GPUs: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {props.name}")
        print(f"         Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    if num_gpus >= 2:
        print("\n  [OK] Multi-GPU configuration available")
        return True
    else:
        print("\n  [WARN] Only 1 GPU - components will share GPU 0")
        return False


def check_docker_services():
    """Check if Docker services are running."""
    print("\n" + "=" * 70)
    print("DOCKER SERVICES CHECK")
    print("=" * 70)

    import subprocess

    services_ok = True

    # Check vLLM
    try:
        result = subprocess.run(
            ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:8001/health'],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip() == '200':
            print("\n  [OK] vLLM server running on port 8001")
        else:
            print("\n  [FAIL] vLLM server not responding")
            services_ok = False
    except Exception as e:
        print(f"\n  [FAIL] vLLM check failed: {e}")
        services_ok = False

    # Check Faster Whisper
    try:
        result = subprocess.run(
            ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:8002/health'],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip() == '200':
            print("  [OK] Faster Whisper server running on port 8002")
        else:
            print("  [FAIL] Faster Whisper server not responding")
            services_ok = False
    except Exception as e:
        print(f"  [FAIL] Faster Whisper check failed: {e}")
        services_ok = False

    return services_ok


def run_latency_test(num_turns=20):
    """Run the latency test with GC optimization."""
    print("\n" + "=" * 70)
    print("LATENCY TEST")
    print("=" * 70)

    # Load components
    print("\n  Loading vLLM engine...")
    from maya.engine.llm_vllm import VLLMEngine
    llm = VLLMEngine()
    llm.initialize()

    print("  Loading TTS engine (GPU 1)...")
    from maya.engine.tts_streaming_real import RealStreamingTTSEngine
    tts = RealStreamingTTSEngine()
    tts.initialize()

    # Verify TTS is on correct GPU
    print(f"  TTS device: {tts._device}")

    # Warmup
    print("\n  Warming up (5 iterations)...")
    for i in range(5):
        resp = llm.generate("hello")
        for chunk in tts.generate_stream(resp, use_context=False):
            pass
    print("  Warmup complete")

    # Force GC before test
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Disable GC for test (same as production)
    print("\n  [GC DISABLED for inference]")
    gc.disable()

    # Test
    print("\n  Running latency test...")
    print("-" * 60)

    test_inputs = [
        "Hello Maya",
        "How are you today",
        "Whats your favorite thing",
        "Tell me something interesting",
        "That sounds cool",
        "What else do you know",
        "I love learning new things",
        "You are really smart",
        "What is your favorite topic",
        "That sounds fascinating",
        "Can you explain more",
        "I see what you mean",
        "That makes sense",
        "What do you think about that",
        "Interesting perspective",
        "I agree with you",
        "Thanks for explaining",
        "You are great at this",
        "One more question",
        "Goodbye Maya",
    ]

    results = []

    for i, user_input in enumerate(test_inputs[:num_turns]):
        # LLM
        llm_start = time.time()
        response = llm.generate(user_input)
        llm_time = (time.time() - llm_start) * 1000

        # TTS (first chunk only)
        tts_start = time.time()
        for chunk in tts.generate_stream(response, use_context=False):
            break
        tts_time = (time.time() - tts_start) * 1000

        total = llm_time + tts_time
        status = "OK" if total < 250 else "WARN" if total < 300 else "SLOW"
        print(f"    [{status:4s}] Turn {i+1:2d}: {total:5.0f}ms (LLM:{llm_time:4.0f} + TTS:{tts_time:4.0f})")

        results.append({
            "turn": i + 1,
            "llm": llm_time,
            "tts": tts_time,
            "total": total,
        })

    # Re-enable GC
    gc.enable()
    gc.collect()

    return results


def analyze_results(results):
    """Analyze and report results."""
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    llm_times = [r["llm"] for r in results]
    tts_times = [r["tts"] for r in results]
    total_times = [r["total"] for r in results]

    # Statistics
    p50 = np.median(total_times)
    p95 = np.percentile(total_times, 95)
    avg = np.mean(total_times)
    best = min(total_times)
    worst = max(total_times)

    under_200 = sum(1 for t in total_times if t < 200)
    under_250 = sum(1 for t in total_times if t < 250)
    under_300 = sum(1 for t in total_times if t < 300)

    print(f"\n  Component Latencies:")
    print(f"    LLM:   avg={np.mean(llm_times):5.0f}ms  p50={np.median(llm_times):5.0f}ms  p95={np.percentile(llm_times, 95):5.0f}ms")
    print(f"    TTS:   avg={np.mean(tts_times):5.0f}ms  p50={np.median(tts_times):5.0f}ms  p95={np.percentile(tts_times, 95):5.0f}ms")

    print(f"\n  Total Latency:")
    print(f"    P50:  {p50:5.0f}ms")
    print(f"    P95:  {p95:5.0f}ms")
    print(f"    Avg:  {avg:5.0f}ms")
    print(f"    Best: {best:5.0f}ms")
    print(f"    Worst:{worst:5.0f}ms")

    print(f"\n  Distribution:")
    print(f"    Under 200ms: {under_200}/{len(results)} ({under_200/len(results)*100:.0f}%)")
    print(f"    Under 250ms: {under_250}/{len(results)} ({under_250/len(results)*100:.0f}%)")
    print(f"    Under 300ms: {under_300}/{len(results)} ({under_300/len(results)*100:.0f}%)")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    sesame_target = 200
    gap = p50 - sesame_target

    print(f"\n  Sesame AI Target: {sesame_target}ms")
    print(f"  Our P50:          {p50:.0f}ms")
    print(f"  Gap:              {gap:+.0f}ms")

    if p50 <= sesame_target:
        print(f"\n  STATUS: SESAME AI LEVEL ACHIEVED")
        print(f"  P50 is {abs(gap):.0f}ms UNDER target")
        return True
    elif p50 <= 220:
        print(f"\n  STATUS: NEAR SESAME LEVEL")
        print(f"  P50 is {gap:.0f}ms above target (within acceptable margin)")
        return True
    elif p50 <= 250:
        print(f"\n  STATUS: CLOSE")
        print(f"  Minor optimization needed")
        return False
    else:
        print(f"\n  STATUS: NEEDS OPTIMIZATION")
        print(f"  Significant gap from target")
        return False


def main():
    print("=" * 70)
    print("PRODUCTION VALIDATION TEST")
    print("Verifying Sesame AI Level Performance")
    print("=" * 70)

    # Pre-flight checks
    multi_gpu = check_gpu_configuration()
    services_ok = check_docker_services()

    if not services_ok:
        print("\n[FAIL] Docker services not running. Start with:")
        print("  ./start_maya.sh start")
        return

    # Run test
    results = run_latency_test(num_turns=20)

    # Analyze
    success = analyze_results(results)

    print("\n" + "=" * 70)
    if success:
        print("PRODUCTION VALIDATION: PASSED")
    else:
        print("PRODUCTION VALIDATION: FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
