"""
Unit tests for the Response Router.

Tests routing logic for Quick Path vs Enhanced Path decisions.
"""

from __future__ import annotations

import pytest

from maya.core.response_router import (
    ResponsePath,
    ResponseRouter,
    RouteDecision,
    RouteReason,
    RouterConfig,
    create_response_router,
)
from maya.constants import QUICK_ACKNOWLEDGMENTS


class TestResponseRouter:
    """Tests for ResponseRouter class."""

    @pytest.fixture
    def router(self) -> ResponseRouter:
        """Create a router with default config."""
        return ResponseRouter()

    @pytest.fixture
    def custom_router(self) -> ResponseRouter:
        """Create a router with custom config."""
        config = RouterConfig(
            max_quick_words=3,
            max_quick_chars=20,
            enable_emotion_routing=True,
            enable_question_detection=True,
        )
        return ResponseRouter(config)

    # ==========================================================================
    # Quick Acknowledgment Tests
    # ==========================================================================

    @pytest.mark.parametrize("text", list(QUICK_ACKNOWLEDGMENTS))
    def test_quick_acknowledgments_route_quick(self, router: ResponseRouter, text: str) -> None:
        """Test that all quick acknowledgments route to quick path."""
        decision = router.route(text)

        assert decision.path == ResponsePath.QUICK
        assert decision.reason == RouteReason.QUICK_ACKNOWLEDGMENT
        assert decision.confidence == 1.0

    @pytest.mark.parametrize("text", [
        "Uh-huh",  # Capital
        "YEAH",    # All caps
        "  okay  ", # Whitespace
        "OK",      # Uppercase variant
    ])
    def test_quick_acknowledgments_case_insensitive(self, router: ResponseRouter, text: str) -> None:
        """Test that quick acknowledgments are case-insensitive."""
        decision = router.route(text)

        assert decision.path == ResponsePath.QUICK
        assert decision.reason == RouteReason.QUICK_ACKNOWLEDGMENT

    # ==========================================================================
    # Short Response Tests
    # ==========================================================================

    @pytest.mark.parametrize("text,expected_path", [
        ("hi", ResponsePath.QUICK),
        ("hello", ResponsePath.QUICK),
        ("thanks", ResponsePath.QUICK),
        ("bye", ResponsePath.QUICK),
        ("no", ResponsePath.QUICK),
    ])
    def test_short_responses_route_quick(
        self,
        router: ResponseRouter,
        text: str,
        expected_path: ResponsePath,
    ) -> None:
        """Test that short responses route to quick path."""
        decision = router.route(text)

        assert decision.path == expected_path
        assert decision.reason == RouteReason.SHORT_RESPONSE

    def test_word_count_threshold(self, router: ResponseRouter) -> None:
        """Test that responses exceeding word count go to enhanced path."""
        # Under threshold
        short = router.route("one two three four")
        assert short.path == ResponsePath.QUICK

        # Over threshold (default is 5 words)
        long = router.route("one two three four five six seven")
        assert long.path == ResponsePath.ENHANCED
        assert long.reason == RouteReason.WORD_COUNT_EXCEEDED

    def test_char_count_threshold(self, router: ResponseRouter) -> None:
        """Test that responses exceeding char count go to enhanced path."""
        # Under threshold
        short = router.route("hi there")
        assert short.path == ResponsePath.QUICK

        # Over threshold (default is 30 chars)
        long = router.route("a" * 35)
        assert long.path == ResponsePath.ENHANCED
        assert long.reason == RouteReason.CHAR_COUNT_EXCEEDED

    # ==========================================================================
    # Enhanced Path Tests
    # ==========================================================================

    @pytest.mark.parametrize("text", [
        "Let me explain how this works in detail.",
        "That's a great question, and here's my answer.",
        "I think we should consider multiple factors here.",
    ])
    def test_long_responses_route_enhanced(self, router: ResponseRouter, text: str) -> None:
        """Test that long responses route to enhanced path."""
        decision = router.route(text)

        assert decision.path == ResponsePath.ENHANCED

    @pytest.mark.parametrize("text", [
        "What do you mean?",
        "How does that work?",
        "Can you explain?",
        "Why is that?",
        "Is that right?",
    ])
    def test_questions_route_enhanced(self, router: ResponseRouter, text: str) -> None:
        """Test that questions route to enhanced path."""
        decision = router.route(text)

        assert decision.path == ResponsePath.ENHANCED
        assert decision.reason == RouteReason.CONTAINS_QUESTION

    @pytest.mark.parametrize("text", [
        "let me think about that",
        "hmm, that's interesting",
        "let's see here",
        "good question, let me think",
    ])
    def test_thinking_markers_route_enhanced(self, router: ResponseRouter, text: str) -> None:
        """Test that thinking markers route to enhanced path."""
        decision = router.route(text)

        assert decision.path == ResponsePath.ENHANCED
        assert decision.reason == RouteReason.THINKING_MARKER

    @pytest.mark.parametrize("text", [
        "I love that idea",
        "That's amazing work",
        "I hate when that happens",
        "This is wonderful news",
    ])
    def test_emotional_content_routes_enhanced(self, router: ResponseRouter, text: str) -> None:
        """Test that emotional content routes to enhanced path."""
        decision = router.route(text)

        assert decision.path == ResponsePath.ENHANCED
        assert decision.reason == RouteReason.EMOTIONAL_CONTENT

    @pytest.mark.parametrize("text", [
        "a, b; c, d",  # Short but complex
        "x - y; z",    # Multiple clause markers
    ])
    def test_complex_sentences_route_enhanced(self, router: ResponseRouter, text: str) -> None:
        """Test that complex sentences with multiple clauses route to enhanced."""
        decision = router.route(text)

        assert decision.path == ResponsePath.ENHANCED
        assert decision.reason == RouteReason.COMPLEX_SENTENCE

    # ==========================================================================
    # Force Path Tests
    # ==========================================================================

    def test_force_quick_path(self, router: ResponseRouter) -> None:
        """Test forcing quick path overrides all rules."""
        # This would normally go to enhanced
        text = "Let me explain this in great detail for you."

        decision = router.route(text, force_path=ResponsePath.QUICK)

        assert decision.path == ResponsePath.QUICK
        assert decision.reason == RouteReason.FORCE_QUICK
        assert decision.confidence == 1.0

    def test_force_enhanced_path(self, router: ResponseRouter) -> None:
        """Test forcing enhanced path overrides all rules."""
        # This would normally go to quick
        text = "okay"

        decision = router.route(text, force_path=ResponsePath.ENHANCED)

        assert decision.path == ResponsePath.ENHANCED
        assert decision.reason == RouteReason.FORCE_ENHANCED
        assert decision.confidence == 1.0

    # ==========================================================================
    # Interruption Tests
    # ==========================================================================

    def test_interruption_routes_quick(self, router: ResponseRouter) -> None:
        """Test that interruptions always route to quick path."""
        # Even long text should go quick if it's an interruption
        text = "Oh wait, I need to interrupt you here"

        decision = router.route(text, is_interruption=True)

        assert decision.path == ResponsePath.QUICK
        assert decision.reason == RouteReason.INTERRUPTION
        assert decision.confidence == 0.95

    # ==========================================================================
    # Decision Properties Tests
    # ==========================================================================

    def test_decision_is_quick(self, router: ResponseRouter) -> None:
        """Test RouteDecision.is_quick property."""
        decision = router.route("yeah")

        assert decision.is_quick is True
        assert decision.is_enhanced is False

    def test_decision_is_enhanced(self, router: ResponseRouter) -> None:
        """Test RouteDecision.is_enhanced property."""
        decision = router.route("Let me explain this to you in detail.")

        assert decision.is_enhanced is True
        assert decision.is_quick is False

    def test_decision_contains_metadata(self, router: ResponseRouter) -> None:
        """Test that decisions contain expected metadata."""
        decision = router.route("hello there")

        assert "original_text" in decision.metadata
        assert "normalized_text" in decision.metadata
        assert decision.word_count == 2
        assert decision.char_count == 11

    # ==========================================================================
    # Batch Routing Tests
    # ==========================================================================

    def test_route_batch(self, router: ResponseRouter) -> None:
        """Test batch routing of multiple texts."""
        texts = ["yeah", "What do you mean?", "okay", "hmm"]

        decisions = router.route_batch(texts)

        assert len(decisions) == 4
        assert decisions[0].path == ResponsePath.QUICK  # quick acknowledgment
        assert decisions[1].path == ResponsePath.ENHANCED  # contains question
        assert decisions[2].path == ResponsePath.QUICK  # quick acknowledgment
        assert decisions[3].path == ResponsePath.ENHANCED  # thinking marker

    # ==========================================================================
    # Statistics Tests
    # ==========================================================================

    def test_stats_tracking(self, router: ResponseRouter) -> None:
        """Test that routing stats are tracked correctly."""
        # Route some texts
        router.route("yeah")
        router.route("okay")
        router.route("Let me explain this in detail")

        stats = router.get_stats()

        assert stats["total_routed"] == 3
        assert stats["quick_count"] == 2
        assert stats["enhanced_count"] == 1
        assert 0.6 < stats["quick_ratio"] < 0.7

    def test_stats_reset(self, router: ResponseRouter) -> None:
        """Test that stats can be reset."""
        router.route("yeah")
        router.route("okay")

        router.reset_stats()
        stats = router.get_stats()

        assert stats["total_routed"] == 0
        assert stats["quick_count"] == 0
        assert stats["enhanced_count"] == 0

    # ==========================================================================
    # Config Tests
    # ==========================================================================

    def test_custom_config(self, custom_router: ResponseRouter) -> None:
        """Test router with custom configuration."""
        # With max_quick_words=3, "one two three four" should go enhanced
        decision = custom_router.route("one two three four")

        assert decision.path == ResponsePath.ENHANCED
        assert decision.reason == RouteReason.WORD_COUNT_EXCEEDED

    def test_disable_question_detection(self) -> None:
        """Test disabling question detection."""
        config = RouterConfig(enable_question_detection=False)
        router = ResponseRouter(config)

        # Short question should now route quick (not detected as question)
        decision = router.route("why?")

        # Should route based on length, not question detection
        assert decision.reason != RouteReason.CONTAINS_QUESTION

    def test_disable_emotion_routing(self) -> None:
        """Test disabling emotion-based routing."""
        config = RouterConfig(enable_emotion_routing=False)
        router = ResponseRouter(config)

        # Short emotional word should now route quick
        decision = router.route("love")

        assert decision.path == ResponsePath.QUICK
        assert decision.reason != RouteReason.EMOTIONAL_CONTENT

    def test_custom_default_path(self) -> None:
        """Test custom default path."""
        config = RouterConfig(default_path=ResponsePath.QUICK)
        router = ResponseRouter(config)

        # Ambiguous case should use configured default
        # (Note: most cases have explicit rules, this tests fallback)
        decision = router.route("hmm")  # This has a thinking marker

        # Thinking marker takes precedence
        assert decision.reason == RouteReason.THINKING_MARKER

    # ==========================================================================
    # Factory Function Tests
    # ==========================================================================

    def test_create_response_router(self) -> None:
        """Test the factory function."""
        router = create_response_router()

        assert isinstance(router, ResponseRouter)
        assert router.config is not None

    def test_create_response_router_with_config(self) -> None:
        """Test factory function with custom config."""
        config = RouterConfig(max_quick_words=10)
        router = create_response_router(config)

        assert router.config.max_quick_words == 10


class TestRouteDecision:
    """Tests for RouteDecision dataclass."""

    def test_default_values(self) -> None:
        """Test RouteDecision default values."""
        decision = RouteDecision(
            path=ResponsePath.QUICK,
            reason=RouteReason.SHORT_RESPONSE,
        )

        assert decision.confidence == 1.0
        assert decision.text == ""
        assert decision.word_count == 0
        assert decision.char_count == 0
        assert decision.metadata == {}

    def test_custom_values(self) -> None:
        """Test RouteDecision with custom values."""
        decision = RouteDecision(
            path=ResponsePath.ENHANCED,
            reason=RouteReason.EMOTIONAL_CONTENT,
            confidence=0.85,
            text="hello",
            word_count=1,
            char_count=5,
            metadata={"key": "value"},
        )

        assert decision.path == ResponsePath.ENHANCED
        assert decision.reason == RouteReason.EMOTIONAL_CONTENT
        assert decision.confidence == 0.85
        assert decision.text == "hello"
        assert decision.word_count == 1
        assert decision.char_count == 5
        assert decision.metadata == {"key": "value"}


class TestRouterConfig:
    """Tests for RouterConfig dataclass."""

    def test_default_config(self) -> None:
        """Test RouterConfig default values."""
        config = RouterConfig()

        assert config.max_quick_words == 5
        assert config.max_quick_chars == 30
        assert config.enable_emotion_routing is True
        assert config.enable_question_detection is True
        assert config.enable_thinking_detection is True
        assert config.default_path == ResponsePath.ENHANCED

    def test_custom_config(self) -> None:
        """Test RouterConfig with custom values."""
        config = RouterConfig(
            max_quick_words=10,
            max_quick_chars=50,
            enable_emotion_routing=False,
        )

        assert config.max_quick_words == 10
        assert config.max_quick_chars == 50
        assert config.enable_emotion_routing is False
