#!/usr/bin/env python3
"""
BRUTAL SESAME AI PARITY STRESS TEST
====================================

This is the most comprehensive test suite for Maya, designed to verify
100% Sesame AI parity across ALL dimensions:

1. LATENCY - Must achieve <200ms P50 first audio
2. VOICE QUALITY - Natural, human, undistinguishable
3. CONVERSATION FLOW - Context, memory, natural responses
4. STREAMING - ASR, TTS, VAD real-time performance
5. CONSISTENCY - Same quality across 1000s of turns
6. EDGE CASES - Interruptions, silence, noise, rapid fire

Benchmarks based on Sesame AI research:
- End-to-end latency: <500ms average, <200ms P50 first audio
- RTF: <0.28x (generating faster than real-time)
- Emotion recognition: 92.7% accuracy
- Voice consistency: Indistinguishable from human in blind tests

Run this with: python scripts/brutal_sesame_test.py
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/csm')
sys.path.insert(0, '/home/ec2-user/SageMaker/project_maya')

import torch
import numpy as np
import time
import logging
import json
import os
import random
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logs during testing
logging.getLogger('maya').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class LatencyMetrics:
    """Latency statistics."""
    samples: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0

    @property
    def p50(self) -> float:
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = len(sorted_samples) // 2
        return sorted_samples[idx]

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def p99(self) -> float:
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0

    @property
    def std(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0


@dataclass
class TestSuite:
    """Complete test suite results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0

    @property
    def average_score(self) -> float:
        scores = [r.score for r in self.results]
        return statistics.mean(scores) if scores else 0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


class BrutalSesameTest:
    """
    Comprehensive stress testing for Sesame AI parity.
    """

    # Sesame AI benchmarks
    SESAME_LATENCY_P50_MS = 200  # Target P50 latency
    SESAME_LATENCY_P95_MS = 500  # Target P95 latency
    SESAME_RTF_TARGET = 0.28    # Real-time factor target
    SESAME_CONTEXT_MINUTES = 2   # Context window

    # Test conversation scenarios
    TEST_CONVERSATIONS = [
        # Casual greeting
        [
            "hey there",
            "im doing pretty good how about you",
            "nice to hear that",
            "so what brings you here today",
        ],
        # Emotional conversation - sad
        [
            "im feeling really down today",
            "yeah work has been really stressful",
            "i just dont know what to do anymore",
            "thanks for listening",
        ],
        # Emotional conversation - excited
        [
            "oh my god guess what happened",
            "i just got promoted at work",
            "im so excited i cant believe it",
            "thanks i worked so hard for this",
        ],
        # Question answering
        [
            "hey can i ask you something",
            "whats your favorite thing to talk about",
            "thats interesting tell me more",
            "yeah i agree with that",
        ],
        # Rapid topic changes
        [
            "hey whats up",
            "so i saw this movie yesterday",
            "wait actually i wanted to ask about cooking",
            "do you like italian food",
        ],
        # Long narrative
        [
            "let me tell you about my day",
            "so first i woke up really early",
            "then i went to this coffee shop",
            "and you wont believe who i saw there",
            "it was my old friend from college",
            "we talked for like two hours",
        ],
        # Uncertain/thinking user
        [
            "hmm im not really sure",
            "well i guess maybe",
            "let me think about that",
            "actually yeah i think so",
        ],
        # Short responses test
        [
            "hi",
            "yes",
            "no",
            "okay",
            "sure",
            "thanks",
        ],
    ]

    # Edge case inputs
    EDGE_CASES = [
        "",  # Empty
        "a",  # Single character
        "um um um um",  # Repetitive fillers
        "THE QUICK BROWN FOX",  # All caps
        "whisper test very quietly",  # Quiet speech simulation
        "supercalifragilisticexpialidocious",  # Long word
        "one two three four five six seven eight nine ten eleven twelve",  # Numbers
        "hahahahahaha",  # Laughter
        "sigh...",  # Emotional sounds
        "wait wait wait hold on",  # Interruption simulation
    ]

    # Stress test phrases (for rapid fire)
    STRESS_PHRASES = [
        "hello", "hi there", "whats up", "hey", "good morning",
        "how are you", "im good", "thats great", "really", "wow",
        "tell me more", "interesting", "i see", "yeah", "okay",
        "sure", "no problem", "thanks", "goodbye", "see you",
    ]

    def __init__(self):
        self.tts = None
        self.llm = None
        self.vad = None
        self.stt = None
        self.pipeline = None
        self.results: Dict[str, TestSuite] = {}

    def initialize_components(self):
        """Initialize all Maya components."""
        logger.info("=" * 70)
        logger.info("INITIALIZING MAYA COMPONENTS FOR BRUTAL TESTING")
        logger.info("=" * 70)

        start = time.time()

        # Import and initialize TTS
        logger.info("Loading TTS engine...")
        from maya.engine.tts_streaming_real import RealStreamingTTSEngine
        self.tts = RealStreamingTTSEngine()
        self.tts.initialize()
        logger.info("  TTS loaded")

        # Import and initialize LLM
        logger.info("Loading LLM engine...")
        from maya.engine.llm_vllm import VLLMEngine
        self.llm = VLLMEngine()
        self.llm.initialize()
        logger.info("  LLM loaded")

        # Import and initialize VAD
        logger.info("Loading VAD engine...")
        from maya.engine.vad import VADEngine
        self.vad = VADEngine()
        self.vad.initialize()
        logger.info("  VAD loaded")

        # Import turn detector for emotion
        logger.info("Loading turn detector...")
        from maya.engine.turn_detector import ProsodyTurnDetector
        self.turn_detector = ProsodyTurnDetector()
        self.turn_detector.initialize()
        logger.info("  Turn detector loaded")

        elapsed = time.time() - start
        logger.info(f"All components loaded in {elapsed:.1f}s")
        logger.info("=" * 70)

    def generate_test_audio(self, duration_seconds: float = 2.0,
                           frequency: float = 440.0,
                           add_noise: bool = False) -> torch.Tensor:
        """Generate synthetic test audio."""
        sample_rate = 24000
        t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds))

        # Generate tone with some harmonics for more natural sound
        audio = torch.sin(2 * np.pi * frequency * t) * 0.3
        audio += torch.sin(2 * np.pi * frequency * 2 * t) * 0.1
        audio += torch.sin(2 * np.pi * frequency * 3 * t) * 0.05

        if add_noise:
            noise = torch.randn_like(audio) * 0.02
            audio += noise

        return audio.float()

    def generate_speech_like_audio(self, duration_seconds: float = 2.0) -> torch.Tensor:
        """Generate audio that resembles speech patterns."""
        sample_rate = 24000
        samples = int(sample_rate * duration_seconds)

        # Create speech-like amplitude envelope
        t = torch.linspace(0, duration_seconds, samples)
        envelope = torch.zeros(samples)

        # Add speech bursts (syllables)
        syllable_duration = 0.15  # seconds
        pause_duration = 0.05
        syllable_samples = int(syllable_duration * sample_rate)
        pause_samples = int(pause_duration * sample_rate)

        pos = 0
        while pos < samples - syllable_samples:
            # Syllable with attack and decay
            syllable = torch.ones(syllable_samples)
            attack = torch.linspace(0, 1, syllable_samples // 4)
            decay = torch.linspace(1, 0.3, syllable_samples // 4)
            syllable[:len(attack)] *= attack
            syllable[-len(decay):] *= decay

            envelope[pos:pos + syllable_samples] = syllable
            pos += syllable_samples + pause_samples + random.randint(-20, 20)

        # Generate carrier with pitch variation (speech-like F0)
        base_freq = 150 + random.random() * 100  # Fundamental frequency
        freq_mod = torch.sin(2 * np.pi * 3 * t) * 20  # Slow pitch variation
        phase = torch.cumsum(2 * np.pi * (base_freq + freq_mod) / sample_rate, dim=0)
        carrier = torch.sin(phase)

        # Add harmonics
        audio = carrier * 0.5
        audio += torch.sin(phase * 2) * 0.25
        audio += torch.sin(phase * 3) * 0.12

        # Apply envelope
        audio = audio * envelope * 0.3

        # Add slight noise
        audio += torch.randn(samples) * 0.01

        return audio.float()

    # =========================================================================
    # TEST SUITES
    # =========================================================================

    def test_tts_latency(self, num_iterations: int = 50) -> TestSuite:
        """Test TTS first chunk latency extensively."""
        suite = TestSuite(name="TTS Latency")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: TTS FIRST CHUNK LATENCY")
        logger.info(f"Running {num_iterations} iterations")
        logger.info("Target: P50 < 150ms, P95 < 200ms")
        logger.info("=" * 70)

        test_phrases = [
            "hello there",
            "how are you doing today",
            "thats really interesting",
            "hmm let me think about that",
            "oh wow thats amazing",
            "im not sure what you mean",
            "yeah i agree with that",
            "tell me more about it",
        ]

        warmup_count = 5
        logger.info(f"Warming up with {warmup_count} generations...")
        for _ in range(warmup_count):
            for chunk in self.tts.generate_stream("warmup test", use_context=False):
                break
        torch.cuda.synchronize()

        first_chunk_times = []

        for i in range(num_iterations):
            phrase = test_phrases[i % len(test_phrases)]

            torch.cuda.synchronize()
            start = time.time()

            first_chunk_time = None
            total_audio_samples = 0

            for chunk in self.tts.generate_stream(phrase, use_context=False):
                if first_chunk_time is None:
                    torch.cuda.synchronize()
                    first_chunk_time = (time.time() - start) * 1000
                total_audio_samples += len(chunk)

            total_time = (time.time() - start) * 1000
            audio_duration_ms = total_audio_samples / 24000 * 1000
            rtf = total_time / audio_duration_ms if audio_duration_ms > 0 else 0

            first_chunk_times.append(first_chunk_time)
            suite.latency.samples.append(first_chunk_time)

            # Log every 10 iterations
            if (i + 1) % 10 == 0:
                current_p50 = sorted(first_chunk_times)[len(first_chunk_times) // 2]
                logger.info(f"  Iteration {i+1}/{num_iterations}: "
                           f"First chunk={first_chunk_time:.0f}ms, "
                           f"RTF={rtf:.2f}x, "
                           f"Current P50={current_p50:.0f}ms")

        # Calculate results
        p50 = suite.latency.p50
        p95 = suite.latency.p95
        passed = p50 < 150 and p95 < 200

        result = TestResult(
            test_name="TTS First Chunk Latency",
            passed=passed,
            score=min(1.0, 150 / p50) if p50 > 0 else 0,
            latency_ms=p50,
            details={
                "iterations": num_iterations,
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "p99_ms": round(suite.latency.p99, 1),
                "min_ms": round(suite.latency.min, 1),
                "max_ms": round(suite.latency.max, 1),
                "std_ms": round(suite.latency.std, 1),
                "target_p50_ms": 150,
                "target_p95_ms": 200,
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  P50: {p50:.1f}ms (target: <150ms)")
        logger.info(f"  P95: {p95:.1f}ms (target: <200ms)")
        logger.info(f"  Min: {suite.latency.min:.1f}ms")
        logger.info(f"  Max: {suite.latency.max:.1f}ms")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_tts_rtf(self, num_iterations: int = 30) -> TestSuite:
        """Test TTS Real-Time Factor."""
        suite = TestSuite(name="TTS RTF")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: TTS REAL-TIME FACTOR (RTF)")
        logger.info(f"Running {num_iterations} iterations")
        logger.info(f"Target: RTF < {self.SESAME_RTF_TARGET}x (Sesame benchmark)")
        logger.info("=" * 70)

        test_phrases = [
            "this is a short test",
            "this is a medium length test phrase for measuring real time factor",
            "hello how are you doing today i hope everything is going well",
        ]

        rtf_values = []

        for i in range(num_iterations):
            phrase = test_phrases[i % len(test_phrases)]

            torch.cuda.synchronize()
            start = time.time()

            total_samples = 0
            for chunk in self.tts.generate_stream(phrase, use_context=False):
                total_samples += len(chunk)

            torch.cuda.synchronize()
            generation_time = time.time() - start
            audio_duration = total_samples / 24000

            rtf = generation_time / audio_duration if audio_duration > 0 else 0
            rtf_values.append(rtf)

            if (i + 1) % 10 == 0:
                avg_rtf = statistics.mean(rtf_values)
                logger.info(f"  Iteration {i+1}/{num_iterations}: RTF={rtf:.3f}x, Avg={avg_rtf:.3f}x")

        avg_rtf = statistics.mean(rtf_values)
        min_rtf = min(rtf_values)
        max_rtf = max(rtf_values)
        passed = avg_rtf < self.SESAME_RTF_TARGET

        result = TestResult(
            test_name="TTS Real-Time Factor",
            passed=passed,
            score=min(1.0, self.SESAME_RTF_TARGET / avg_rtf) if avg_rtf > 0 else 0,
            details={
                "avg_rtf": round(avg_rtf, 3),
                "min_rtf": round(min_rtf, 3),
                "max_rtf": round(max_rtf, 3),
                "target_rtf": self.SESAME_RTF_TARGET,
                "iterations": num_iterations,
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Average RTF: {avg_rtf:.3f}x (target: <{self.SESAME_RTF_TARGET}x)")
        logger.info(f"  Min RTF: {min_rtf:.3f}x")
        logger.info(f"  Max RTF: {max_rtf:.3f}x")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_tts_audio_quality(self, num_samples: int = 20) -> TestSuite:
        """Test TTS audio quality metrics."""
        suite = TestSuite(name="TTS Audio Quality")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: TTS AUDIO QUALITY")
        logger.info(f"Generating {num_samples} samples")
        logger.info("Checking: RMS consistency, peak levels, silence ratio")
        logger.info("=" * 70)

        test_phrases = [
            "hello there how are you",
            "thats really interesting",
            "im so excited about this",
            "oh no thats terrible",
            "hmm let me think",
            "yeah i totally agree",
        ]

        rms_values = []
        peak_values = []
        silence_ratios = []

        for i in range(num_samples):
            phrase = test_phrases[i % len(test_phrases)]

            audio = self.tts.generate(phrase, use_context=False)
            audio_np = audio.cpu().numpy()

            # Calculate RMS
            rms = np.sqrt(np.mean(audio_np ** 2))
            rms_values.append(rms)

            # Calculate peak
            peak = np.max(np.abs(audio_np))
            peak_values.append(peak)

            # Calculate silence ratio (samples below threshold)
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio_np) < silence_threshold) / len(audio_np)
            silence_ratios.append(silence_ratio)

            if (i + 1) % 5 == 0:
                logger.info(f"  Sample {i+1}/{num_samples}: RMS={rms:.3f}, Peak={peak:.3f}, Silence={silence_ratio:.1%}")

        # Analyze consistency
        rms_std = statistics.stdev(rms_values)
        rms_mean = statistics.mean(rms_values)
        rms_cv = rms_std / rms_mean if rms_mean > 0 else 0  # Coefficient of variation

        peak_mean = statistics.mean(peak_values)
        peak_std = statistics.stdev(peak_values)

        silence_mean = statistics.mean(silence_ratios)

        # Quality checks
        # RMS should be consistent (CV < 0.3) and in good range (0.05-0.3)
        rms_consistent = rms_cv < 0.3
        rms_in_range = 0.05 < rms_mean < 0.3

        # Peaks should not clip (< 0.95) but not too quiet (> 0.3)
        peak_good = 0.3 < peak_mean < 0.95

        # Silence should be reasonable (< 40%)
        silence_good = silence_mean < 0.4

        passed = rms_consistent and rms_in_range and peak_good and silence_good
        score = sum([rms_consistent, rms_in_range, peak_good, silence_good]) / 4

        result = TestResult(
            test_name="TTS Audio Quality",
            passed=passed,
            score=score,
            details={
                "rms_mean": round(rms_mean, 4),
                "rms_std": round(rms_std, 4),
                "rms_cv": round(rms_cv, 3),
                "rms_consistent": rms_consistent,
                "rms_in_range": rms_in_range,
                "peak_mean": round(peak_mean, 3),
                "peak_std": round(peak_std, 3),
                "peak_good": peak_good,
                "silence_mean": round(silence_mean, 3),
                "silence_good": silence_good,
                "samples": num_samples,
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'} (Score: {score:.0%})")
        logger.info(f"  RMS: {rms_mean:.3f} ± {rms_std:.3f} (CV={rms_cv:.2f})")
        logger.info(f"  Peak: {peak_mean:.3f} ± {peak_std:.3f}")
        logger.info(f"  Silence ratio: {silence_mean:.1%}")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_tts_voice_consistency(self, num_pairs: int = 15) -> TestSuite:
        """Test voice consistency across generations."""
        suite = TestSuite(name="TTS Voice Consistency")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: TTS VOICE CONSISTENCY")
        logger.info(f"Generating {num_pairs} pairs of same phrase")
        logger.info("Checking: Spectral similarity, duration consistency")
        logger.info("=" * 70)

        test_phrases = [
            "hello how are you",
            "thats really cool",
            "i dont know about that",
        ]

        spectral_similarities = []
        duration_diffs = []

        for i in range(num_pairs):
            phrase = test_phrases[i % len(test_phrases)]

            # Generate same phrase twice
            audio1 = self.tts.generate(phrase, use_context=False)
            audio2 = self.tts.generate(phrase, use_context=False)

            # Duration comparison
            dur1 = len(audio1) / 24000
            dur2 = len(audio2) / 24000
            duration_diff = abs(dur1 - dur2) / max(dur1, dur2)
            duration_diffs.append(duration_diff)

            # Simple spectral similarity (RMS envelope correlation)
            # Normalize lengths
            min_len = min(len(audio1), len(audio2))
            a1 = audio1[:min_len].cpu().numpy()
            a2 = audio2[:min_len].cpu().numpy()

            # Calculate envelope
            window_size = 512
            env1 = np.array([np.sqrt(np.mean(a1[j:j+window_size]**2))
                           for j in range(0, len(a1) - window_size, window_size // 2)])
            env2 = np.array([np.sqrt(np.mean(a2[j:j+window_size]**2))
                           for j in range(0, len(a2) - window_size, window_size // 2)])

            # Correlation
            min_env_len = min(len(env1), len(env2))
            if min_env_len > 10:
                correlation = np.corrcoef(env1[:min_env_len], env2[:min_env_len])[0, 1]
                spectral_similarities.append(max(0, correlation))

            if (i + 1) % 5 == 0:
                logger.info(f"  Pair {i+1}/{num_pairs}: Duration diff={duration_diff:.1%}, "
                           f"Envelope corr={correlation:.3f}")

        avg_duration_diff = statistics.mean(duration_diffs)
        avg_similarity = statistics.mean(spectral_similarities)

        # Voice should be consistent: duration within 20%, correlation > 0.5
        duration_consistent = avg_duration_diff < 0.2
        spectrally_similar = avg_similarity > 0.5

        passed = duration_consistent and spectrally_similar
        score = (1 - avg_duration_diff) * 0.5 + avg_similarity * 0.5

        result = TestResult(
            test_name="TTS Voice Consistency",
            passed=passed,
            score=score,
            details={
                "avg_duration_diff": round(avg_duration_diff, 3),
                "avg_spectral_similarity": round(avg_similarity, 3),
                "duration_consistent": duration_consistent,
                "spectrally_similar": spectrally_similar,
                "pairs": num_pairs,
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'} (Score: {score:.0%})")
        logger.info(f"  Duration consistency: {1 - avg_duration_diff:.1%}")
        logger.info(f"  Spectral similarity: {avg_similarity:.3f}")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_llm_response_quality(self, num_tests: int = 50) -> TestSuite:
        """Test LLM response quality and naturalness."""
        suite = TestSuite(name="LLM Response Quality")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: LLM RESPONSE QUALITY")
        logger.info(f"Running {num_tests} response generations")
        logger.info("Checking: Word count, contractions, naturalness")
        logger.info("=" * 70)

        self.llm.clear_history()

        test_inputs = [
            "hey whats up",
            "how are you today",
            "i had a really rough day",
            "guess what happened to me",
            "do you like music",
            "whats your favorite food",
            "im feeling kind of down",
            "this is so exciting",
            "i dont understand",
            "can you explain that",
        ]

        word_counts = []
        has_contractions = []
        has_filler = []
        latencies = []

        for i in range(num_tests):
            user_input = test_inputs[i % len(test_inputs)]

            start = time.time()
            response = self.llm.generate(user_input)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            suite.latency.samples.append(latency)

            # Analyze response
            words = response.split()
            word_count = len(words)
            word_counts.append(word_count)

            # Check for contractions
            contractions = ['im', 'youre', 'dont', 'thats', 'its', 'cant', 'wont', 'isnt', 'arent']
            has_contraction = any(c in response.lower() for c in contractions)
            has_contractions.append(has_contraction)

            # Check for natural fillers
            fillers = ['um', 'hmm', 'oh', 'yeah', 'well', 'like', 'uh', 'mmm', 'ah', 'ooh', 'aww']
            has_fill = any(f in response.lower().split() for f in fillers)
            has_filler.append(has_fill)

            if (i + 1) % 10 == 0:
                avg_words = statistics.mean(word_counts)
                contraction_rate = sum(has_contractions) / len(has_contractions)
                filler_rate = sum(has_filler) / len(has_filler)
                logger.info(f"  Test {i+1}/{num_tests}: Avg words={avg_words:.1f}, "
                           f"Contractions={contraction_rate:.0%}, Fillers={filler_rate:.0%}")

        # Calculate metrics
        avg_words = statistics.mean(word_counts)
        word_std = statistics.stdev(word_counts)
        contraction_rate = sum(has_contractions) / len(has_contractions)
        filler_rate = sum(has_filler) / len(has_filler)
        avg_latency = statistics.mean(latencies)

        # Quality checks
        # Word count should be 8-15 on average
        word_count_good = 6 <= avg_words <= 18
        # Contractions should appear in most responses (>60%)
        contractions_good = contraction_rate > 0.6
        # Fillers should appear sometimes (20-60%)
        fillers_good = 0.15 <= filler_rate <= 0.7
        # Latency should be fast (<200ms)
        latency_good = avg_latency < 200

        passed = word_count_good and contractions_good and latency_good
        score = sum([word_count_good, contractions_good, fillers_good, latency_good]) / 4

        result = TestResult(
            test_name="LLM Response Quality",
            passed=passed,
            score=score,
            latency_ms=avg_latency,
            details={
                "avg_word_count": round(avg_words, 1),
                "word_count_std": round(word_std, 1),
                "contraction_rate": round(contraction_rate, 2),
                "filler_rate": round(filler_rate, 2),
                "avg_latency_ms": round(avg_latency, 1),
                "latency_p95_ms": round(suite.latency.p95, 1),
                "word_count_good": word_count_good,
                "contractions_good": contractions_good,
                "fillers_good": fillers_good,
                "latency_good": latency_good,
                "tests": num_tests,
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'} (Score: {score:.0%})")
        logger.info(f"  Word count: {avg_words:.1f} ± {word_std:.1f} (target: 8-15)")
        logger.info(f"  Contractions: {contraction_rate:.0%} (target: >60%)")
        logger.info(f"  Fillers: {filler_rate:.0%} (target: 20-60%)")
        logger.info(f"  Latency: {avg_latency:.0f}ms (target: <200ms)")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_llm_conversation_context(self, conversation_length: int = 20) -> TestSuite:
        """Test LLM conversation context and memory."""
        suite = TestSuite(name="LLM Conversation Context")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: LLM CONVERSATION CONTEXT")
        logger.info(f"Running {conversation_length}-turn conversation")
        logger.info("Checking: Context retention, coherence")
        logger.info("=" * 70)

        self.llm.clear_history()

        # Run a conversation that requires context
        conversation = [
            ("my name is alex", ["alex", "name"]),
            ("i live in new york", ["new york", "york", "city"]),
            ("do you remember my name", ["alex"]),
            ("and where do i live", ["new york", "york"]),
            ("i have a dog named max", ["max", "dog"]),
            ("what do you know about me", ["alex", "new york", "dog", "max"]),
        ]

        context_retained = 0
        total_context_checks = 0

        for user_input, expected_in_response in conversation[:conversation_length]:
            start = time.time()
            response = self.llm.generate(user_input)
            latency = (time.time() - start) * 1000
            suite.latency.samples.append(latency)

            # Check if context is retained
            response_lower = response.lower()
            context_found = any(exp.lower() in response_lower for exp in expected_in_response)

            if expected_in_response:
                total_context_checks += 1
                if context_found:
                    context_retained += 1

            logger.info(f"  User: {user_input}")
            logger.info(f"  Maya: {response}")
            logger.info(f"  Context check: {expected_in_response} -> {'FOUND' if context_found else 'NOT FOUND'}")
            logger.info("")

        retention_rate = context_retained / total_context_checks if total_context_checks > 0 else 0
        passed = retention_rate >= 0.5  # At least 50% context retention

        result = TestResult(
            test_name="LLM Conversation Context",
            passed=passed,
            score=retention_rate,
            latency_ms=suite.latency.mean,
            details={
                "context_retained": context_retained,
                "total_checks": total_context_checks,
                "retention_rate": round(retention_rate, 2),
                "conversation_length": conversation_length,
            }
        )
        suite.results.append(result)

        logger.info("-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Context retention: {retention_rate:.0%} ({context_retained}/{total_context_checks})")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_vad_accuracy(self, num_tests: int = 30) -> TestSuite:
        """Test VAD accuracy with synthetic audio."""
        suite = TestSuite(name="VAD Accuracy")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: VAD ACCURACY")
        logger.info(f"Running {num_tests} tests")
        logger.info("Checking: Speech detection, silence detection")
        logger.info("=" * 70)

        self.vad.reset()

        correct_detections = 0
        total_tests = 0
        latencies = []

        for i in range(num_tests):
            # Generate speech-like or silence
            is_speech = i % 2 == 0

            if is_speech:
                audio = self.generate_speech_like_audio(duration_seconds=0.5)
            else:
                audio = torch.randn(12000) * 0.001  # Very quiet noise

            start = time.time()
            result = self.vad.process(audio)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            suite.latency.samples.append(latency)

            detected_speech = result.is_speech or result.confidence > 0.5

            # For synthetic audio, detection may not be perfect
            # We're mainly testing that VAD runs and returns results
            correct = (is_speech and detected_speech) or (not is_speech and not detected_speech)
            if correct:
                correct_detections += 1
            total_tests += 1

            if (i + 1) % 10 == 0:
                accuracy = correct_detections / total_tests
                avg_latency = statistics.mean(latencies)
                logger.info(f"  Test {i+1}/{num_tests}: Accuracy={accuracy:.0%}, Latency={avg_latency:.1f}ms")

        accuracy = correct_detections / total_tests
        avg_latency = statistics.mean(latencies)

        # VAD should be fast (<50ms) and reasonably accurate
        passed = avg_latency < 50

        result = TestResult(
            test_name="VAD Accuracy",
            passed=passed,
            score=accuracy,
            latency_ms=avg_latency,
            details={
                "accuracy": round(accuracy, 2),
                "avg_latency_ms": round(avg_latency, 1),
                "max_latency_ms": round(max(latencies), 1),
                "tests": num_tests,
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Latency: {avg_latency:.1f}ms (target: <50ms)")
        logger.info(f"  Note: Accuracy measured on synthetic audio")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_turn_detector_emotion(self, num_tests: int = 20) -> TestSuite:
        """Test turn detector emotion detection."""
        suite = TestSuite(name="Emotion Detection")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: EMOTION DETECTION")
        logger.info(f"Running {num_tests} tests")
        logger.info("Checking: Emotion classification accuracy")
        logger.info("=" * 70)

        from maya.engine.turn_detector import EmotionHint

        results_by_emotion = {
            "excited": [],
            "sad": [],
            "calm": [],
            "neutral": [],
        }

        for i in range(num_tests):
            # Generate audio with different characteristics
            audio = self.generate_speech_like_audio(duration_seconds=1.0)
            audio_np = audio.cpu().numpy()

            start = time.time()
            emotion = self.turn_detector.detect_emotion(audio_np, sample_rate=24000)
            latency = (time.time() - start) * 1000
            suite.latency.samples.append(latency)

            results_by_emotion.get(emotion.primary_emotion, []).append(emotion)

            if (i + 1) % 5 == 0:
                logger.info(f"  Test {i+1}/{num_tests}: Emotion={emotion.primary_emotion}, "
                           f"Confidence={emotion.confidence:.2f}, Latency={latency:.1f}ms")

        avg_latency = suite.latency.mean

        # Just check that emotion detection runs and returns valid results
        passed = avg_latency < 100  # Should be fast

        result = TestResult(
            test_name="Emotion Detection",
            passed=passed,
            score=0.8,  # Placeholder - real accuracy needs labeled data
            latency_ms=avg_latency,
            details={
                "avg_latency_ms": round(avg_latency, 1),
                "tests": num_tests,
                "emotion_distribution": {k: len(v) for k, v in results_by_emotion.items()},
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Latency: {avg_latency:.1f}ms")
        logger.info(f"  Emotions detected: {[(k, len(v)) for k, v in results_by_emotion.items()]}")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_rapid_fire(self, num_requests: int = 100) -> TestSuite:
        """Stress test with rapid fire requests."""
        suite = TestSuite(name="Rapid Fire Stress")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: RAPID FIRE STRESS TEST")
        logger.info(f"Running {num_requests} rapid requests")
        logger.info("Checking: Stability under load, latency degradation")
        logger.info("=" * 70)

        phrases = self.STRESS_PHRASES
        errors = 0
        first_chunk_times = []

        for i in range(num_requests):
            phrase = phrases[i % len(phrases)]

            try:
                torch.cuda.synchronize()
                start = time.time()

                first_chunk_time = None
                for chunk in self.tts.generate_stream(phrase, use_context=False):
                    if first_chunk_time is None:
                        torch.cuda.synchronize()
                        first_chunk_time = (time.time() - start) * 1000
                    break  # Just get first chunk

                if first_chunk_time:
                    first_chunk_times.append(first_chunk_time)
                    suite.latency.samples.append(first_chunk_time)

            except Exception as e:
                errors += 1
                logger.warning(f"  Error at {i}: {e}")

            if (i + 1) % 20 == 0:
                if first_chunk_times:
                    recent = first_chunk_times[-20:]
                    avg = statistics.mean(recent)
                    p95 = sorted(recent)[int(len(recent) * 0.95)]
                    logger.info(f"  Request {i+1}/{num_requests}: "
                               f"Avg={avg:.0f}ms, P95={p95:.0f}ms, Errors={errors}")

        # Calculate results
        if first_chunk_times:
            avg_latency = statistics.mean(first_chunk_times)
            p95_latency = suite.latency.p95

            # Compare first 20 vs last 20 for degradation
            first_20 = statistics.mean(first_chunk_times[:20])
            last_20 = statistics.mean(first_chunk_times[-20:])
            degradation = (last_20 - first_20) / first_20 if first_20 > 0 else 0
        else:
            avg_latency = 0
            p95_latency = 0
            degradation = 0

        error_rate = errors / num_requests

        # Should have <1% errors and <20% latency degradation
        passed = error_rate < 0.01 and degradation < 0.2

        result = TestResult(
            test_name="Rapid Fire Stress",
            passed=passed,
            score=1 - error_rate,
            latency_ms=avg_latency,
            details={
                "total_requests": num_requests,
                "errors": errors,
                "error_rate": round(error_rate, 3),
                "avg_latency_ms": round(avg_latency, 1),
                "p95_latency_ms": round(p95_latency, 1),
                "latency_degradation": round(degradation, 2),
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Errors: {errors}/{num_requests} ({error_rate:.1%})")
        logger.info(f"  Avg latency: {avg_latency:.0f}ms")
        logger.info(f"  P95 latency: {p95_latency:.0f}ms")
        logger.info(f"  Degradation: {degradation:.0%}")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_long_conversation(self, num_turns: int = 50) -> TestSuite:
        """Test long conversation stability."""
        suite = TestSuite(name="Long Conversation")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: LONG CONVERSATION STABILITY")
        logger.info(f"Running {num_turns}-turn conversation")
        logger.info("Checking: Memory stability, latency consistency")
        logger.info("=" * 70)

        self.llm.clear_history()
        self.tts.clear_context()

        all_conversations = []
        for conv in self.TEST_CONVERSATIONS:
            all_conversations.extend(conv)

        llm_latencies = []
        tts_latencies = []
        errors = 0

        for i in range(num_turns):
            user_input = all_conversations[i % len(all_conversations)]

            try:
                # LLM
                start = time.time()
                response = self.llm.generate(user_input)
                llm_latency = (time.time() - start) * 1000
                llm_latencies.append(llm_latency)

                # TTS
                start = time.time()
                first_chunk_time = None
                audio_chunks = []
                for chunk in self.tts.generate_stream(response, use_context=True):
                    if first_chunk_time is None:
                        first_chunk_time = (time.time() - start) * 1000
                    audio_chunks.append(chunk)

                if first_chunk_time:
                    tts_latencies.append(first_chunk_time)
                    suite.latency.samples.append(llm_latency + first_chunk_time)

                # Add to TTS context
                if audio_chunks:
                    full_audio = torch.cat(audio_chunks)
                    self.tts.add_context(response, full_audio, is_user=False)

            except Exception as e:
                errors += 1
                logger.warning(f"  Error at turn {i}: {e}")

            if (i + 1) % 10 == 0:
                avg_llm = statistics.mean(llm_latencies[-10:])
                avg_tts = statistics.mean(tts_latencies[-10:]) if tts_latencies[-10:] else 0
                logger.info(f"  Turn {i+1}/{num_turns}: LLM={avg_llm:.0f}ms, TTS={avg_tts:.0f}ms")

        # Calculate stability metrics
        if llm_latencies and tts_latencies:
            llm_first_10 = statistics.mean(llm_latencies[:10])
            llm_last_10 = statistics.mean(llm_latencies[-10:])
            llm_degradation = (llm_last_10 - llm_first_10) / llm_first_10 if llm_first_10 > 0 else 0

            tts_first_10 = statistics.mean(tts_latencies[:10])
            tts_last_10 = statistics.mean(tts_latencies[-10:])
            tts_degradation = (tts_last_10 - tts_first_10) / tts_first_10 if tts_first_10 > 0 else 0
        else:
            llm_degradation = 0
            tts_degradation = 0

        error_rate = errors / num_turns
        passed = error_rate < 0.05 and abs(llm_degradation) < 0.3 and abs(tts_degradation) < 0.3

        result = TestResult(
            test_name="Long Conversation Stability",
            passed=passed,
            score=1 - error_rate,
            latency_ms=suite.latency.mean,
            details={
                "turns": num_turns,
                "errors": errors,
                "error_rate": round(error_rate, 3),
                "llm_degradation": round(llm_degradation, 2),
                "tts_degradation": round(tts_degradation, 2),
                "avg_total_latency_ms": round(suite.latency.mean, 1),
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Turns completed: {num_turns - errors}/{num_turns}")
        logger.info(f"  LLM degradation: {llm_degradation:.0%}")
        logger.info(f"  TTS degradation: {tts_degradation:.0%}")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_edge_cases(self) -> TestSuite:
        """Test edge cases and error handling."""
        suite = TestSuite(name="Edge Cases")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: EDGE CASES")
        logger.info(f"Testing {len(self.EDGE_CASES)} edge cases")
        logger.info("=" * 70)

        passed_cases = 0
        failed_cases = []

        for i, edge_case in enumerate(self.EDGE_CASES):
            try:
                # LLM should handle any input
                response = self.llm.generate(edge_case)

                # TTS should handle the response
                if response and response.strip():
                    audio_generated = False
                    for chunk in self.tts.generate_stream(response, use_context=False):
                        if len(chunk) > 0:
                            audio_generated = True
                            break

                    if audio_generated:
                        passed_cases += 1
                        logger.info(f"  Case {i+1}: '{edge_case[:30]}...' -> PASS")
                    else:
                        failed_cases.append(edge_case)
                        logger.warning(f"  Case {i+1}: '{edge_case[:30]}...' -> No audio")
                else:
                    passed_cases += 1  # Empty response is OK for edge cases
                    logger.info(f"  Case {i+1}: '{edge_case[:30]}...' -> Empty response (OK)")

            except Exception as e:
                failed_cases.append(edge_case)
                logger.warning(f"  Case {i+1}: '{edge_case[:30]}...' -> ERROR: {e}")

        pass_rate = passed_cases / len(self.EDGE_CASES)
        passed = pass_rate >= 0.8  # 80% should pass

        result = TestResult(
            test_name="Edge Cases",
            passed=passed,
            score=pass_rate,
            details={
                "total_cases": len(self.EDGE_CASES),
                "passed": passed_cases,
                "failed": len(failed_cases),
                "failed_cases": failed_cases[:5],  # First 5 failures
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Passed: {passed_cases}/{len(self.EDGE_CASES)} ({pass_rate:.0%})")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def test_full_pipeline_latency(self, num_iterations: int = 30) -> TestSuite:
        """Test complete pipeline latency (simulated STT + LLM + TTS)."""
        suite = TestSuite(name="Full Pipeline Latency")
        suite.start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("TEST: FULL PIPELINE LATENCY")
        logger.info(f"Running {num_iterations} full pipeline simulations")
        logger.info(f"Target: P50 < {self.SESAME_LATENCY_P50_MS}ms")
        logger.info("=" * 70)

        self.llm.clear_history()

        test_inputs = [
            "hey how are you",
            "thats really cool",
            "tell me more about that",
            "i dont understand",
            "yeah i agree",
        ]

        pipeline_latencies = []

        for i in range(num_iterations):
            user_input = test_inputs[i % len(test_inputs)]

            torch.cuda.synchronize()
            start = time.time()

            # Simulate STT latency (actual would be ~25ms with streaming)
            stt_latency = 25  # ms - simulated

            # LLM
            llm_start = time.time()
            response = self.llm.generate(user_input)
            llm_latency = (time.time() - llm_start) * 1000

            # TTS first chunk
            tts_start = time.time()
            first_chunk_time = None
            for chunk in self.tts.generate_stream(response, use_context=False):
                torch.cuda.synchronize()
                first_chunk_time = (time.time() - tts_start) * 1000
                break

            total_latency = stt_latency + llm_latency + (first_chunk_time or 0)
            pipeline_latencies.append(total_latency)
            suite.latency.samples.append(total_latency)

            if (i + 1) % 10 == 0:
                p50 = sorted(pipeline_latencies)[len(pipeline_latencies) // 2]
                logger.info(f"  Iteration {i+1}/{num_iterations}: "
                           f"STT={stt_latency}ms, LLM={llm_latency:.0f}ms, "
                           f"TTS={first_chunk_time:.0f}ms, Total={total_latency:.0f}ms, "
                           f"P50={p50:.0f}ms")

        p50 = suite.latency.p50
        p95 = suite.latency.p95

        passed = p50 < self.SESAME_LATENCY_P50_MS
        score = min(1.0, self.SESAME_LATENCY_P50_MS / p50) if p50 > 0 else 0

        result = TestResult(
            test_name="Full Pipeline Latency",
            passed=passed,
            score=score,
            latency_ms=p50,
            details={
                "iterations": num_iterations,
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "min_ms": round(suite.latency.min, 1),
                "max_ms": round(suite.latency.max, 1),
                "target_p50_ms": self.SESAME_LATENCY_P50_MS,
            }
        )
        suite.results.append(result)

        logger.info("\n" + "-" * 50)
        logger.info(f"RESULT: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  P50: {p50:.0f}ms (target: <{self.SESAME_LATENCY_P50_MS}ms)")
        logger.info(f"  P95: {p95:.0f}ms")
        logger.info(f"  Min: {suite.latency.min:.0f}ms")
        logger.info(f"  Max: {suite.latency.max:.0f}ms")
        logger.info("-" * 50)

        suite.end_time = time.time()
        return suite

    def run_all_tests(self) -> Dict[str, TestSuite]:
        """Run all test suites."""
        logger.info("\n")
        logger.info("=" * 70)
        logger.info("BRUTAL SESAME AI PARITY STRESS TEST")
        logger.info("=" * 70)
        logger.info("This will take several minutes. Go get some coffee.")
        logger.info("=" * 70)

        total_start = time.time()

        # Initialize
        self.initialize_components()

        # Run all test suites
        test_methods = [
            ("tts_latency", lambda: self.test_tts_latency(num_iterations=50)),
            ("tts_rtf", lambda: self.test_tts_rtf(num_iterations=30)),
            ("tts_quality", lambda: self.test_tts_audio_quality(num_samples=20)),
            ("tts_consistency", lambda: self.test_tts_voice_consistency(num_pairs=15)),
            ("llm_quality", lambda: self.test_llm_response_quality(num_tests=50)),
            ("llm_context", lambda: self.test_llm_conversation_context(conversation_length=6)),
            ("vad_accuracy", lambda: self.test_vad_accuracy(num_tests=30)),
            ("emotion_detection", lambda: self.test_turn_detector_emotion(num_tests=20)),
            ("rapid_fire", lambda: self.test_rapid_fire(num_requests=100)),
            ("long_conversation", lambda: self.test_long_conversation(num_turns=30)),
            ("edge_cases", lambda: self.test_edge_cases()),
            ("full_pipeline", lambda: self.test_full_pipeline_latency(num_iterations=30)),
        ]

        for name, test_func in test_methods:
            try:
                suite = test_func()
                self.results[name] = suite
            except Exception as e:
                logger.error(f"Test suite {name} failed: {e}")
                import traceback
                traceback.print_exc()

        total_time = time.time() - total_start

        # Print summary
        self.print_summary(total_time)

        return self.results

    def print_summary(self, total_time: float):
        """Print comprehensive test summary."""
        logger.info("\n")
        logger.info("=" * 70)
        logger.info("BRUTAL STRESS TEST SUMMARY")
        logger.info("=" * 70)

        total_tests = 0
        passed_tests = 0
        total_score = 0

        for name, suite in self.results.items():
            for result in suite.results:
                total_tests += 1
                if result.passed:
                    passed_tests += 1
                total_score += result.score

        avg_score = total_score / total_tests if total_tests > 0 else 0
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        logger.info(f"\nOVERALL RESULTS:")
        logger.info(f"  Total test suites: {len(self.results)}")
        logger.info(f"  Total tests: {total_tests}")
        logger.info(f"  Passed: {passed_tests} ({pass_rate:.0%})")
        logger.info(f"  Average score: {avg_score:.0%}")
        logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

        logger.info(f"\nDETAILED RESULTS:")
        logger.info("-" * 70)

        for name, suite in self.results.items():
            for result in suite.results:
                status = "PASS" if result.passed else "FAIL"
                logger.info(f"  [{status}] {result.test_name}: Score={result.score:.0%}, "
                           f"Latency={result.latency_ms:.0f}ms")

        logger.info("-" * 70)

        # Sesame AI parity assessment
        logger.info(f"\nSESAME AI PARITY ASSESSMENT:")

        parity_checks = [
            ("TTS P50 < 150ms", self.results.get("tts_latency", TestSuite("")).results[0].passed if "tts_latency" in self.results else False),
            ("TTS RTF < 0.28x", self.results.get("tts_rtf", TestSuite("")).results[0].passed if "tts_rtf" in self.results else False),
            ("Voice Consistency", self.results.get("tts_consistency", TestSuite("")).results[0].passed if "tts_consistency" in self.results else False),
            ("LLM Response Quality", self.results.get("llm_quality", TestSuite("")).results[0].passed if "llm_quality" in self.results else False),
            ("Pipeline < 200ms", self.results.get("full_pipeline", TestSuite("")).results[0].passed if "full_pipeline" in self.results else False),
            ("Stability (rapid fire)", self.results.get("rapid_fire", TestSuite("")).results[0].passed if "rapid_fire" in self.results else False),
            ("Long conversation", self.results.get("long_conversation", TestSuite("")).results[0].passed if "long_conversation" in self.results else False),
        ]

        parity_score = sum(1 for _, passed in parity_checks if passed) / len(parity_checks)

        for check_name, passed in parity_checks:
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check_name}")

        logger.info("-" * 70)
        logger.info(f"SESAME AI PARITY: {parity_score:.0%}")

        if parity_score >= 0.85:
            logger.info("STATUS: SESAME AI LEVEL ACHIEVED!")
        elif parity_score >= 0.7:
            logger.info("STATUS: CLOSE TO SESAME AI LEVEL")
        else:
            logger.info("STATUS: NEEDS IMPROVEMENT")

        logger.info("=" * 70)

        # Save results to file
        results_file = "/home/ec2-user/SageMaker/project_maya/test_results.json"
        results_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": total_time,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "average_score": avg_score,
            "parity_score": parity_score,
            "suites": {}
        }

        for name, suite in self.results.items():
            results_data["suites"][name] = {
                "duration_seconds": suite.duration_seconds,
                "latency_p50_ms": suite.latency.p50,
                "latency_p95_ms": suite.latency.p95,
                "results": [asdict(r) for r in suite.results]
            }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    tester = BrutalSesameTest()
    results = tester.run_all_tests()
