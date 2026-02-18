"""
Smart Turn Detection Engine - Prosodic Feature Analysis

Implements intelligent turn detection based on prosodic cues:
1. Pitch contour analysis (rising = incomplete, falling = complete)
2. Energy/intensity decay at utterance end
3. Speaking rate changes (slowdown before turn end)
4. Pause patterns

This replaces simple silence-based VAD with intelligent turn detection.

Research basis:
- 80ms of pitch before utterance end can classify rising/falling with 93% accuracy
- Pre-boundary lengthening is a strong cue for turn completion
- Energy typically drops at natural turn boundaries

Key difference from silence VAD:
- VAD: User pauses for 600ms → assume turn complete
- Prosodic: Analyzes audio features → predicts if utterance is complete

This prevents:
- Interrupting users who are thinking mid-sentence
- Waiting too long when user is clearly done
- Awkward "um... are you there?" moments
"""

import numpy as np
import logging
import time
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy import signal  # For proper anti-aliased resampling

logger = logging.getLogger(__name__)


@dataclass
class ProsodyFeatures:
    """Extracted prosodic features from audio."""
    pitch_trend: float  # -1 (falling) to +1 (rising)
    energy_trend: float  # -1 (falling) to +1 (rising)
    final_energy: float  # Normalized energy of final segment
    speech_rate_change: float  # Negative = slowing down
    has_final_pause: bool  # Pause in final 200ms
    has_pitch_reset: bool  # Pitch reset indicates continuation (Sesame-level feature)
    pitch_variance: float  # High variance = still thinking/speaking


@dataclass
class EmotionHint:
    """Basic emotion detection from prosody.

    SESAME-LEVEL: Detect user emotion for adaptive responses.
    - excited: high pitch, high energy, fast speech
    - calm: moderate pitch, moderate energy, normal speed
    - sad: low pitch, low energy, slow speech
    - uncertain: rising pitch at end, variable energy
    """
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # -1 (low energy) to +1 (high energy)
    primary_emotion: str  # "excited", "calm", "sad", "uncertain", "neutral"
    confidence: float  # 0.0 to 1.0


class ProsodyTurnDetector:
    """
    Turn detection using prosodic feature analysis.

    Based on research showing that:
    - Falling pitch contour strongly indicates turn completion
    - Pre-boundary lengthening (slower speech) indicates turn end
    - Energy decay indicates natural boundary
    - Brief final pause confirms completion

    Usage:
        detector = ProsodyTurnDetector()
        is_complete, confidence = detector.is_turn_complete(audio_buffer)
    """

    # Audio parameters
    SAMPLE_RATE = 16000
    MIN_AUDIO_MS = 300  # Minimum audio to analyze
    ANALYSIS_WINDOW_MS = 500  # Analyze last 500ms for prosody

    # Feature weights for classification
    PITCH_WEIGHT = 0.35
    ENERGY_WEIGHT = 0.25
    RATE_WEIGHT = 0.20
    PAUSE_WEIGHT = 0.20

    # Thresholds
    COMPLETE_THRESHOLD = 0.55

    def __init__(self):
        self._initialized = False
        self._total_inferences = 0
        self._total_time = 0.0

    def initialize(self) -> None:
        """Initialize the detector."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("LOADING PROSODY-BASED TURN DETECTOR")
        logger.info("=" * 60)
        logger.info("Features: pitch contour, energy, speech rate, pauses")
        logger.info(f"Threshold: {self.COMPLETE_THRESHOLD}")
        logger.info("=" * 60)

        self._initialized = True

    def _extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch (F0) contour using autocorrelation method.

        This is a proper pitch estimation that finds the fundamental frequency
        by detecting periodicity in the signal. Much more accurate than ZCR
        for turn detection (falling pitch = statement complete).
        """
        try:
            frame_length = 1024
            hop_length = 256
            min_f0 = 50   # Hz - minimum pitch
            max_f0 = 400  # Hz - maximum pitch

            if len(audio) < frame_length:
                return np.array([])

            # Calculate pitch per frame using autocorrelation
            num_frames = 1 + (len(audio) - frame_length) // hop_length
            pitches = np.zeros(num_frames)

            min_lag = int(self.SAMPLE_RATE / max_f0)
            max_lag = int(self.SAMPLE_RATE / min_f0)

            for i in range(num_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]

                # Normalize frame
                frame = frame - np.mean(frame)
                if np.max(np.abs(frame)) < 1e-6:
                    continue

                # Autocorrelation
                corr = np.correlate(frame, frame, mode='full')
                corr = corr[len(corr)//2:]  # Take positive lags only

                # Find peak in valid lag range
                if max_lag < len(corr):
                    search_region = corr[min_lag:max_lag]
                    if len(search_region) > 0:
                        peak_idx = np.argmax(search_region) + min_lag
                        # Validate peak (should be significant)
                        if corr[peak_idx] > 0.3 * corr[0]:
                            pitches[i] = self.SAMPLE_RATE / peak_idx

            return pitches

        except Exception as e:
            logger.debug(f"Pitch extraction failed: {e}")
            return np.array([])

    def _extract_energy(self, audio: np.ndarray, hop_length: int = 256) -> np.ndarray:
        """Extract energy contour from audio."""
        frame_length = 1024

        if len(audio) < frame_length:
            return np.array([np.sqrt(np.mean(audio ** 2))])

        # Calculate RMS energy per frame
        num_frames = 1 + (len(audio) - frame_length) // hop_length
        energy = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        return energy

    def _compute_trend(self, values: np.ndarray, window_ratio: float = 0.3) -> float:
        """
        Compute trend (rising/falling) of a feature over final portion.

        Returns:
            -1.0 to +1.0 where negative = falling, positive = rising
        """
        if len(values) < 4:
            return 0.0

        # Focus on final portion
        window_size = max(2, int(len(values) * window_ratio))
        final_values = values[-window_size:]

        # Skip zeros/NaN
        valid = final_values[final_values > 0]
        if len(valid) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(valid))
        slope, _ = np.polyfit(x, valid, 1)

        # Normalize slope to [-1, 1]
        mean_val = np.mean(valid)
        if mean_val > 0:
            normalized_slope = slope / (mean_val * 0.1)  # Scale factor
            return float(np.clip(normalized_slope, -1.0, 1.0))

        return 0.0

    def _detect_final_pause(self, audio: np.ndarray, threshold: float = 0.02) -> bool:
        """Check if there's a pause in the final 200ms."""
        pause_samples = int(self.SAMPLE_RATE * 0.2)  # 200ms
        if len(audio) < pause_samples:
            return False

        final_audio = audio[-pause_samples:]
        energy = np.sqrt(np.mean(final_audio ** 2))

        return energy < threshold

    def _detect_pitch_reset(self, pitch: np.ndarray) -> Tuple[bool, float]:
        """
        Detect pitch reset - a strong indicator of continued speech.

        Pitch reset occurs when:
        1. Pitch drops significantly then rises again (continuation)
        2. High pitch variance in final segment (still thinking)

        This is a KEY Sesame-level feature that prevents interrupting users
        who are pausing to think mid-sentence.

        Returns:
            (has_reset, variance) - whether pitch reset detected and pitch variance
        """
        if len(pitch) < 6:
            return False, 0.0

        # Filter out unvoiced frames (pitch = 0)
        voiced_pitch = pitch[pitch > 0]
        if len(voiced_pitch) < 4:
            return False, 0.0

        # Calculate pitch variance (high = still active speech)
        variance = float(np.var(voiced_pitch) / (np.mean(voiced_pitch) + 1e-6))

        # Look for pitch reset pattern: drop then rise
        # Analyze last 40% of pitch contour
        analysis_start = int(len(voiced_pitch) * 0.6)
        final_pitch = voiced_pitch[analysis_start:]

        if len(final_pitch) < 3:
            return False, variance

        # Find if there's a local minimum followed by rise
        min_idx = np.argmin(final_pitch)

        # Reset detected if:
        # 1. Minimum is not at the very end (there's a rise after)
        # 2. The rise is significant (> 10% of mean pitch)
        if min_idx < len(final_pitch) - 2:
            rise_after_min = final_pitch[min_idx + 1:].max() - final_pitch[min_idx]
            mean_pitch = np.mean(final_pitch)
            if rise_after_min > mean_pitch * 0.1:
                return True, variance

        # Also check for high variance (still thinking/modulating)
        # Threshold: variance > 0.05 suggests active speech
        if variance > 0.05:
            return True, variance

        return False, variance

    def _extract_features(self, audio: np.ndarray) -> ProsodyFeatures:
        """Extract all prosodic features from audio."""
        # Pitch contour and trend
        pitch = self._extract_pitch(audio)
        pitch_trend = self._compute_trend(pitch) if len(pitch) > 0 else 0.0

        # Energy contour and trend
        energy = self._extract_energy(audio)
        energy_trend = self._compute_trend(energy) if len(energy) > 0 else 0.0

        # Final energy (normalized)
        final_energy = float(energy[-1] / np.max(energy)) if len(energy) > 0 and np.max(energy) > 0 else 0.5

        # Speech rate change (approximated by energy variations)
        # More variation = faster speech, less = slowing down
        if len(energy) > 4:
            first_half_var = np.var(energy[:len(energy)//2])
            second_half_var = np.var(energy[len(energy)//2:])
            if first_half_var > 0:
                speech_rate_change = (second_half_var - first_half_var) / first_half_var
                speech_rate_change = float(np.clip(speech_rate_change, -1.0, 1.0))
            else:
                speech_rate_change = 0.0
        else:
            speech_rate_change = 0.0

        # Final pause detection
        has_final_pause = self._detect_final_pause(audio)

        # Pitch reset detection (Sesame-level thinking pause detection)
        has_pitch_reset, pitch_variance = self._detect_pitch_reset(pitch)

        return ProsodyFeatures(
            pitch_trend=pitch_trend,
            energy_trend=energy_trend,
            final_energy=final_energy,
            speech_rate_change=speech_rate_change,
            has_final_pause=has_final_pause,
            has_pitch_reset=has_pitch_reset,
            pitch_variance=pitch_variance
        )

    # Weight for pitch reset (negative weight - reduces completion confidence)
    PITCH_RESET_WEIGHT = 0.25

    def _classify(self, features: ProsodyFeatures) -> Tuple[bool, float]:
        """
        Classify turn as complete or incomplete based on features.

        Complete turn indicators:
        - Falling pitch (pitch_trend < 0)
        - Falling energy (energy_trend < 0)
        - Slowing speech (speech_rate_change < 0)
        - Final pause present

        Incomplete turn indicators (Sesame-level):
        - Pitch reset detected (user thinking, will continue)
        - High pitch variance (still modulating voice)

        Returns:
            (is_complete, confidence)
        """
        # Score each feature (higher = more likely complete)
        scores = []

        # Pitch: falling = complete
        pitch_score = (1.0 - features.pitch_trend) / 2.0  # Map [-1,1] to [1,0]
        scores.append(pitch_score * self.PITCH_WEIGHT)

        # Energy: falling = complete
        energy_score = (1.0 - features.energy_trend) / 2.0
        scores.append(energy_score * self.ENERGY_WEIGHT)

        # Speech rate: slowing = complete
        rate_score = (1.0 - features.speech_rate_change) / 2.0
        scores.append(rate_score * self.RATE_WEIGHT)

        # Final pause: present = complete
        pause_score = 1.0 if features.has_final_pause else 0.3
        scores.append(pause_score * self.PAUSE_WEIGHT)

        # Combined score (before pitch reset adjustment)
        base_confidence = sum(scores)

        # CRITICAL Sesame-level feature: Pitch reset detection
        # If pitch reset detected, user is likely still thinking - DON'T interrupt
        reset_penalty = 0.0
        if features.has_pitch_reset:
            # Strong penalty for pitch reset
            reset_penalty = self.PITCH_RESET_WEIGHT
            logger.debug(f"Pitch reset detected: penalty={reset_penalty:.2f}, variance={features.pitch_variance:.3f}")

        # Also penalize high pitch variance (still actively speaking)
        if features.pitch_variance > 0.03:
            variance_penalty = min(0.15, features.pitch_variance * 2)
            reset_penalty = max(reset_penalty, variance_penalty)
            logger.debug(f"High pitch variance: penalty={variance_penalty:.2f}")

        # Final confidence
        confidence = max(0.0, base_confidence - reset_penalty)
        is_complete = confidence >= self.COMPLETE_THRESHOLD

        return is_complete, confidence

    def is_turn_complete(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[bool, float]:
        """
        Determine if the user's turn is complete.

        Args:
            audio: Audio buffer (numpy array)
            sample_rate: Sample rate of input audio

        Returns:
            Tuple of (is_complete, confidence)
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.squeeze()

        # Resample if needed - use scipy.signal.resample for proper anti-aliasing
        # This prevents aliasing artifacts that degrade pitch detection accuracy
        if sample_rate != self.SAMPLE_RATE:
            target_length = int(len(audio) * self.SAMPLE_RATE / sample_rate)
            # scipy.signal.resample uses FFT-based resampling with anti-aliasing
            audio = signal.resample(audio, target_length).astype(np.float32)

        # Check minimum length
        min_samples = int(self.SAMPLE_RATE * self.MIN_AUDIO_MS / 1000)
        if len(audio) < min_samples:
            return False, 0.0

        # Focus on analysis window (last 500ms)
        analysis_samples = int(self.SAMPLE_RATE * self.ANALYSIS_WINDOW_MS / 1000)
        if len(audio) > analysis_samples:
            audio = audio[-analysis_samples:]

        # Extract features and classify
        try:
            features = self._extract_features(audio)
            is_complete, confidence = self._classify(features)
        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            # Default to incomplete on error
            return False, 0.3

        # Track performance
        elapsed = time.time() - start
        self._total_inferences += 1
        self._total_time += elapsed

        logger.debug(
            f"Turn: {'COMPLETE' if is_complete else 'INCOMPLETE'} "
            f"(conf={confidence:.2f}, pitch={features.pitch_trend:.2f}, "
            f"energy={features.energy_trend:.2f}) [{elapsed*1000:.0f}ms]"
        )

        return is_complete, confidence

    def get_adaptive_silence_timeout(
        self,
        audio: np.ndarray,
        base_timeout_ms: int = 600,
        extended_timeout_ms: int = 2000,
        sample_rate: int = 16000
    ) -> int:
        """
        Get adaptive silence timeout based on turn completeness.

        If prosody suggests incomplete turn, extend the silence timeout.
        """
        is_complete, confidence = self.is_turn_complete(audio, sample_rate)

        if is_complete and confidence > 0.7:
            # High confidence complete - use shorter timeout
            return base_timeout_ms
        elif is_complete:
            # Medium confidence - use base timeout
            return base_timeout_ms
        else:
            # Incomplete - scale timeout based on incompleteness
            incomplete_confidence = 1.0 - confidence
            timeout = base_timeout_ms + int(
                (extended_timeout_ms - base_timeout_ms) * incomplete_confidence
            )
            return min(timeout, extended_timeout_ms)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def average_latency_ms(self) -> float:
        if self._total_inferences == 0:
            return 0.0
        return (self._total_time / self._total_inferences) * 1000

    def get_stats(self) -> dict:
        return {
            "total_inferences": self._total_inferences,
            "average_latency_ms": self.average_latency_ms,
        }

    def detect_emotion(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> EmotionHint:
        """
        SESAME-LEVEL: Detect basic emotion from prosodic features.

        Uses pitch and energy patterns to infer emotional state:
        - High pitch + high energy + fast = excited
        - Low pitch + low energy + slow = sad
        - Rising pitch at end = uncertain/questioning
        - Moderate everything = calm/neutral

        Args:
            audio: Audio buffer (numpy array)
            sample_rate: Sample rate of input audio

        Returns:
            EmotionHint with valence, arousal, and primary emotion
        """
        if not self._initialized:
            self.initialize()

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.squeeze()

        # Resample if needed
        if sample_rate != self.SAMPLE_RATE:
            target_length = int(len(audio) * self.SAMPLE_RATE / sample_rate)
            audio = signal.resample(audio, target_length).astype(np.float32)

        # Minimum length check
        if len(audio) < 4000:  # 250ms minimum
            return EmotionHint(
                valence=0.0,
                arousal=0.0,
                primary_emotion="neutral",
                confidence=0.0
            )

        try:
            # Extract features
            features = self._extract_features(audio)

            # Calculate arousal from energy
            # High final energy and rising energy = high arousal
            arousal = (features.final_energy - 0.5) * 2  # Map to [-1, 1]
            arousal += features.energy_trend * 0.3
            arousal = float(np.clip(arousal, -1.0, 1.0))

            # Calculate valence from pitch
            # Rising pitch often indicates positive/excited (but also uncertain)
            # Falling pitch with low energy = negative
            # High variance often indicates animated/positive speech
            valence = features.pitch_trend * 0.4
            valence += features.pitch_variance * 2  # High variance = animated = positive
            valence -= 0.3 if features.final_energy < 0.3 else 0  # Low energy penalty
            valence = float(np.clip(valence, -1.0, 1.0))

            # Determine primary emotion
            if features.pitch_trend > 0.3 and arousal > 0.3:
                if features.has_final_pause:
                    primary_emotion = "uncertain"
                    confidence = 0.6
                else:
                    primary_emotion = "excited"
                    confidence = 0.7
            elif features.pitch_trend < -0.3 and arousal < -0.2:
                primary_emotion = "sad"
                confidence = 0.6
            elif arousal > 0.4:
                primary_emotion = "excited"
                confidence = 0.5
            elif arousal < -0.3 and valence < -0.2:
                primary_emotion = "sad"
                confidence = 0.5
            elif abs(valence) < 0.3 and abs(arousal) < 0.3:
                primary_emotion = "calm"
                confidence = 0.7
            else:
                primary_emotion = "neutral"
                confidence = 0.5

            return EmotionHint(
                valence=valence,
                arousal=arousal,
                primary_emotion=primary_emotion,
                confidence=confidence
            )

        except Exception as e:
            logger.debug(f"Emotion detection failed: {e}")
            return EmotionHint(
                valence=0.0,
                arousal=0.0,
                primary_emotion="neutral",
                confidence=0.0
            )


# Alias for backward compatibility
SmartTurnDetector = ProsodyTurnDetector
