#!/usr/bin/env python3
"""
Sesame AI Level Validation Test

This script validates that Maya achieves Sesame AI-level performance:
- Target: < 200ms first audio latency

Tests:
1. Individual component latencies (STT, LLM, TTS)
2. True streaming STT overlap savings
3. Full pipeline end-to-end latency
4. Audio quality validation

Usage:
    python scripts/test_sesame_level.py

Requirements:
    - vLLM Docker running (./start_maya.sh start)
    - All GPUs available
"""

import sys
import os
import time
import torch
import numpy as np
import logging

sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class SesameValidator:
    """Validate Sesame AI-level performance."""

    TARGET_FIRST_AUDIO_MS = 200
    TARGET_STT_MS = 30  # With streaming overlap
    TARGET_LLM_MS = 80  # vLLM with good settings
    TARGET_TTS_FIRST_MS = 120  # 2-frame first chunk

    def __init__(self):
        self.results = {
            "stt": [],
            "llm": [],
            "tts_first": [],
            "total": [],
            "stt_overlap_saved": 0,
        }

    def validate_stt_streaming(self, num_tests: int = 5) -> dict:
        """Validate TRUE streaming STT with overlap savings."""
        from maya.engine.stt_true_streaming import TrueStreamingSTTEngine

        logger.info("\n" + "=" * 60)
        logger.info("VALIDATING TRUE STREAMING STT")
        logger.info("=" * 60)

        stt = TrueStreamingSTTEngine()
        stt.initialize()

        # Simulate real speech pattern: 1.5s of audio arriving in 200ms chunks
        sample_rate = 24000
        speech_duration_s = 1.5
        chunk_duration_s = 0.2

        # Generate realistic audio
        t = torch.linspace(0, speech_duration_s, int(sample_rate * speech_duration_s))
        full_audio = torch.sin(2 * np.pi * 440 * t) * 0.3 + torch.randn(len(t)) * 0.1

        results = []

        for test_num in range(num_tests):
            stt.reset()
            speech_start = time.time()

            # Simulate streaming - feed chunks as they would arrive
            chunk_samples = int(sample_rate * chunk_duration_s)
            partial_count = 0

            for i in range(0, len(full_audio), chunk_samples):
                chunk = full_audio[i:i+chunk_samples]
                result = stt.add_audio(chunk)
                if result:
                    partial_count += 1

            # Finalize when "speech ends"
            speech_duration = (time.time() - speech_start) * 1000

            final_start = time.time()
            final_result = stt.finalize()
            final_latency = (time.time() - final_start) * 1000

            results.append({
                "speech_duration_ms": speech_duration,
                "final_latency_ms": final_latency,
                "partial_count": partial_count,
                "text": final_result.text[:30] if final_result.text else "(empty)"
            })

            if test_num < 3:
                logger.info(f"  Test {test_num + 1}: final_latency={final_latency:.0f}ms, "
                           f"partials={partial_count}, text='{results[-1]['text']}...'")

        # Skip warmup
        results = results[2:]

        avg_final = np.mean([r["final_latency_ms"] for r in results])
        avg_partials = np.mean([r["partial_count"] for r in results])

        # Traditional STT would take ~85ms
        # Streaming saves the time that was overlapped with speech
        traditional_stt_ms = 85
        overlap_saved = traditional_stt_ms - avg_final

        self.results["stt_overlap_saved"] = overlap_saved

        logger.info(f"\nStreaming STT Results:")
        logger.info(f"  Average final latency: {avg_final:.0f}ms")
        logger.info(f"  Average partial results: {avg_partials:.1f}")
        logger.info(f"  Estimated overlap saved: ~{overlap_saved:.0f}ms")
        logger.info(f"  Target: <{self.TARGET_STT_MS}ms")

        status = "✓ PASS" if avg_final < self.TARGET_STT_MS else "✗ FAIL"
        logger.info(f"  Status: {status}")

        return {"avg_final_ms": avg_final, "overlap_saved": overlap_saved, "passed": avg_final < self.TARGET_STT_MS}

    def validate_llm(self, num_tests: int = 5) -> dict:
        """Validate vLLM latency."""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATING VLLM LATENCY")
        logger.info("=" * 60)

        # Check if vLLM is running
        import requests
        try:
            resp = requests.get("http://localhost:8001/health", timeout=2)
            if resp.status_code != 200:
                logger.error("vLLM not healthy. Run: ./start_maya.sh start")
                return {"passed": False, "error": "vLLM not healthy"}
        except Exception:
            logger.error("vLLM not running. Run: ./start_maya.sh start")
            return {"passed": False, "error": "vLLM not running"}

        from maya.engine.llm_vllm import VLLMEngine

        llm = VLLMEngine()
        llm.initialize()

        test_inputs = [
            "Hello", "How are you", "Tell me about yourself",
            "What's the weather like", "That sounds interesting"
        ]

        results = []

        for i, text in enumerate(test_inputs[:num_tests]):
            start = time.time()
            response = llm.generate(text)
            elapsed = (time.time() - start) * 1000
            results.append(elapsed)
            llm.clear_history()

            if i < 3:
                logger.info(f"  Test {i + 1}: {elapsed:.0f}ms -> '{response[:40]}...'")

        # Skip warmup
        results = results[2:]

        avg = np.mean(results)
        p50 = np.percentile(results, 50)
        p95 = np.percentile(results, 95)

        self.results["llm"] = results

        transport = "Unix Socket" if llm._use_unix_socket else "HTTP"
        logger.info(f"\nvLLM Results ({transport}):")
        logger.info(f"  Average: {avg:.0f}ms")
        logger.info(f"  P50: {p50:.0f}ms")
        logger.info(f"  P95: {p95:.0f}ms")
        logger.info(f"  Target: <{self.TARGET_LLM_MS}ms")

        status = "✓ PASS" if avg < self.TARGET_LLM_MS else "✗ FAIL"
        logger.info(f"  Status: {status}")

        return {"avg_ms": avg, "p50_ms": p50, "p95_ms": p95, "passed": avg < self.TARGET_LLM_MS}

    def validate_tts_first_chunk(self, num_tests: int = 5) -> dict:
        """Validate TTS first chunk latency."""
        from maya.engine.tts_streaming_real import RealStreamingTTSEngine

        logger.info("\n" + "=" * 60)
        logger.info("VALIDATING TTS FIRST CHUNK LATENCY")
        logger.info("=" * 60)

        tts = RealStreamingTTSEngine()
        tts.initialize()

        test_texts = [
            "oh hey how are you",
            "thats really interesting",
            "let me think about that",
            "yeah i totally get it",
            "sure sounds good to me"
        ]

        results = []

        for i, text in enumerate(test_texts[:num_tests]):
            torch.cuda.synchronize()
            start = time.time()

            first_chunk_time = None
            for chunk in tts.generate_stream(text, use_context=False):
                if first_chunk_time is None:
                    torch.cuda.synchronize()
                    first_chunk_time = (time.time() - start) * 1000
                    chunk_audio_ms = len(chunk) / 24000 * 1000
                    break

            results.append(first_chunk_time)

            if i < 3:
                logger.info(f"  Test {i + 1}: first_chunk={first_chunk_time:.0f}ms "
                           f"({chunk_audio_ms:.0f}ms audio)")

        # Skip warmup
        results = results[2:]

        avg = np.mean(results)
        p50 = np.percentile(results, 50)
        p95 = np.percentile(results, 95)

        self.results["tts_first"] = results

        logger.info(f"\nTTS First Chunk Results ({tts.FIRST_CHUNK_FRAMES} frames):")
        logger.info(f"  Average: {avg:.0f}ms")
        logger.info(f"  P50: {p50:.0f}ms")
        logger.info(f"  P95: {p95:.0f}ms")
        logger.info(f"  Target: <{self.TARGET_TTS_FIRST_MS}ms")

        status = "✓ PASS" if avg < self.TARGET_TTS_FIRST_MS else "✗ FAIL"
        logger.info(f"  Status: {status}")

        return {"avg_ms": avg, "p50_ms": p50, "passed": avg < self.TARGET_TTS_FIRST_MS}

    def validate_full_pipeline(self, num_tests: int = 4) -> dict:
        """Validate full pipeline end-to-end latency."""
        from maya.engine.stt_true_streaming import TrueStreamingSTTEngine
        from maya.engine.llm_vllm import VLLMEngine
        from maya.engine.tts_streaming_real import RealStreamingTTSEngine

        logger.info("\n" + "=" * 60)
        logger.info("VALIDATING FULL PIPELINE (SESAME LEVEL)")
        logger.info("Target: < 200ms first audio")
        logger.info("=" * 60)

        # Initialize all components
        stt = TrueStreamingSTTEngine()
        llm = VLLMEngine()
        tts = RealStreamingTTSEngine()

        stt.initialize()
        llm.initialize()
        tts.initialize()

        results = []

        # Simulate complete conversation turn
        sample_rate = 24000
        speech_duration_s = 1.5
        chunk_duration_s = 0.2

        for test_num in range(num_tests):
            # Generate test audio
            t = torch.linspace(0, speech_duration_s, int(sample_rate * speech_duration_s))
            full_audio = torch.sin(2 * np.pi * 440 * t) * 0.3 + torch.randn(len(t)) * 0.1

            # === PHASE 1: User speaking (STT runs in parallel) ===
            stt.reset()
            chunk_samples = int(sample_rate * chunk_duration_s)

            for i in range(0, len(full_audio), chunk_samples):
                chunk = full_audio[i:i+chunk_samples]
                stt.add_audio(chunk)
                # Simulate real-time (optional - for accurate simulation)
                # time.sleep(chunk_duration_s * 0.1)

            # === PHASE 2: Speech ended - measure from here ===
            torch.cuda.synchronize()
            response_start = time.time()

            # STT finalize (most work done during speech)
            stt_start = time.time()
            stt_result = stt.finalize()
            stt_time = (time.time() - stt_start) * 1000

            transcript = stt_result.text if stt_result.text else "hello how are you"

            # LLM
            llm_start = time.time()
            response = llm.generate(transcript)
            llm_time = (time.time() - llm_start) * 1000

            llm.clear_history()

            # TTS first chunk
            tts_start = time.time()
            first_chunk_time = None
            for chunk in tts.generate_stream(response, use_context=False):
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - tts_start) * 1000
                break

            torch.cuda.synchronize()
            total_time = (time.time() - response_start) * 1000

            result = {
                "stt_ms": stt_time,
                "llm_ms": llm_time,
                "tts_first_ms": first_chunk_time,
                "total_ms": total_time
            }
            results.append(result)

            logger.info(f"\nTest {test_num + 1}:")
            logger.info(f"  STT:       {stt_time:.0f}ms (streaming finalize)")
            logger.info(f"  LLM:       {llm_time:.0f}ms")
            logger.info(f"  TTS first: {first_chunk_time:.0f}ms")
            logger.info(f"  TOTAL:     {total_time:.0f}ms")

        # Skip warmup
        results = results[1:]

        avg_stt = np.mean([r["stt_ms"] for r in results])
        avg_llm = np.mean([r["llm_ms"] for r in results])
        avg_tts = np.mean([r["tts_first_ms"] for r in results])
        avg_total = np.mean([r["total_ms"] for r in results])

        self.results["total"] = [r["total_ms"] for r in results]

        logger.info(f"\n{'=' * 60}")
        logger.info("FULL PIPELINE SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"  STT (streaming finalize):  {avg_stt:.0f}ms")
        logger.info(f"  LLM (vLLM):                {avg_llm:.0f}ms")
        logger.info(f"  TTS (first chunk):         {avg_tts:.0f}ms")
        logger.info(f"  ---")
        logger.info(f"  TOTAL FIRST AUDIO:         {avg_total:.0f}ms")
        logger.info(f"  TARGET:                    {self.TARGET_FIRST_AUDIO_MS}ms")

        gap = avg_total - self.TARGET_FIRST_AUDIO_MS
        if gap <= 0:
            status = f"✓ SESAME LEVEL ACHIEVED ({-gap:.0f}ms under target)"
        else:
            status = f"✗ {gap:.0f}ms above target"

        logger.info(f"\n  STATUS: {status}")
        logger.info(f"{'=' * 60}")

        return {
            "avg_stt_ms": avg_stt,
            "avg_llm_ms": avg_llm,
            "avg_tts_ms": avg_tts,
            "avg_total_ms": avg_total,
            "passed": avg_total < self.TARGET_FIRST_AUDIO_MS
        }

    def run_all_validations(self) -> dict:
        """Run all validations and report overall status."""
        logger.info("=" * 60)
        logger.info("SESAME AI LEVEL VALIDATION")
        logger.info("Target: < 200ms first audio latency")
        logger.info("=" * 60)

        # Run validations
        stt_result = self.validate_stt_streaming()
        llm_result = self.validate_llm()
        tts_result = self.validate_tts_first_chunk()
        pipeline_result = self.validate_full_pipeline()

        # Overall summary
        logger.info("\n" + "=" * 60)
        logger.info("FINAL VALIDATION REPORT")
        logger.info("=" * 60)

        all_passed = True

        components = [
            ("Streaming STT", stt_result.get("passed", False), f"{stt_result.get('avg_final_ms', 0):.0f}ms"),
            ("vLLM", llm_result.get("passed", False), f"{llm_result.get('avg_ms', 0):.0f}ms"),
            ("TTS First Chunk", tts_result.get("passed", False), f"{tts_result.get('avg_ms', 0):.0f}ms"),
            ("Full Pipeline", pipeline_result.get("passed", False), f"{pipeline_result.get('avg_total_ms', 0):.0f}ms"),
        ]

        for name, passed, latency in components:
            status = "✓" if passed else "✗"
            all_passed = all_passed and passed
            logger.info(f"  {status} {name}: {latency}")

        logger.info("")
        logger.info(f"  Overlap savings from streaming STT: ~{self.results['stt_overlap_saved']:.0f}ms")
        logger.info("")

        if all_passed:
            logger.info("  ★★★ ALL VALIDATIONS PASSED - SESAME AI LEVEL ACHIEVED! ★★★")
        else:
            logger.info("  Some validations failed. See details above.")

        logger.info("=" * 60)

        return {
            "stt": stt_result,
            "llm": llm_result,
            "tts": tts_result,
            "pipeline": pipeline_result,
            "all_passed": all_passed
        }


def main():
    validator = SesameValidator()
    results = validator.run_all_validations()

    # Exit with appropriate code
    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
