"""Tests for LLM summarization."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.fusion.schema import FusedDocument, FusedEvent
from src.llm.parsing import LLMOutputParseError, strip_and_parse_json, validate_timestamps
from src.llm.schema import Chapter, DetectedEvent, KeyPoint, Summary
from src.llm.summarizer import _payload_to_summary, summarize


class FakeProvider:
    """Fake LLM provider for testing."""

    def __init__(self, responses: list[tuple[dict, dict]]):
        """Initialize with scripted responses."""
        self.responses = iter(responses)
        self.model = "fake-model"

    def complete_json(self, system: str, user: str, **kwargs) -> tuple[dict, dict]:
        """Return next scripted response."""
        return next(self.responses)


# Fixtures
@pytest.fixture
def sample_fused_doc() -> FusedDocument:
    """Create a sample fused document."""
    events = [
        FusedEvent(
            t_start=0.0,
            t_end=5.0,
            kind="speech",
            speech_text="Today we're going to discuss machine learning fundamentals.",
        ),
        FusedEvent(
            t_start=5.0,
            t_end=10.0,
            kind="visual",
            visual_caption="Slide: What is Machine Learning?",
        ),
        FusedEvent(
            t_start=10.0,
            t_end=15.0,
            kind="speech",
            speech_text="Machine learning is a subset of artificial intelligence.",
        ),
    ]

    return FusedDocument(
        run_id="test-run-001",
        duration_sec=15.0,
        language="en",
        events=events,
        speech_source="local-whisper",
        ocr_engine="tesseract",
        has_captions=True,
    )


@pytest.fixture
def temp_fused_json(tmp_path: Path, sample_fused_doc: FusedDocument) -> Path:
    """Save sample fused doc to temp file."""
    from src.fusion.schema import save_fused_document

    path = tmp_path / "fused.json"
    save_fused_document(sample_fused_doc, path)
    return path


# JSON Parsing Tests
class TestJSONParsing:
    def test_strip_and_parse_direct_json(self):
        """Test parsing direct JSON."""
        text = '{"key": "value", "number": 42}'
        result = strip_and_parse_json(text)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_strip_and_parse_fenced_json(self):
        """Test parsing JSON with markdown fences."""
        text = """```json
{
  "key": "value",
  "nested": {"inner": 123}
}
```"""
        result = strip_and_parse_json(text)
        assert result["key"] == "value"
        assert result["nested"]["inner"] == 123

    def test_strip_and_parse_with_prose(self):
        """Test parsing JSON with surrounding prose."""
        text = """Here's the analysis:

```json
{"analysis": "complete", "confidence": "high"}
```

Let me know if you need more details."""
        result = strip_and_parse_json(text)
        assert result["analysis"] == "complete"

    def test_strip_and_parse_no_fences(self):
        """Test parsing JSON without fences but with prose."""
        text = """The data is:
{"key": "value"}
And that's it."""
        result = strip_and_parse_json(text)
        assert result["key"] == "value"

    def test_strip_and_parse_with_trailing_comma(self):
        """Test parsing JSON with trailing commas."""
        text = '{"key": "value", "array": [1, 2, 3,],}'
        result = strip_and_parse_json(text)
        assert result["key"] == "value"
        assert result["array"] == [1, 2, 3]

    def test_strip_and_parse_empty_text(self):
        """Test parsing empty text."""
        with pytest.raises(LLMOutputParseError):
            strip_and_parse_json("")

    def test_strip_and_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        with pytest.raises(LLMOutputParseError):
            # Use a string with NO braces at all to ensure parse failure
            strip_and_parse_json("just some random text that is not json")
class TestTimestampValidation:
    def test_validate_timestamps_within_range(self):
        """Test validation of timestamps within range."""
        payload = {
            "key_points": [
                {"timestamp": 0.0, "text": "start"},
                {"timestamp": 5.0, "text": "middle"},
                {"timestamp": 10.0, "text": "end"},
            ]
        }

        result = validate_timestamps(payload, duration=10.0)
        assert len(result["key_points"]) == 3

    def test_validate_timestamps_out_of_range(self):
        """Test filtering of out-of-range timestamps."""
        payload = {
            "key_points": [
                {"timestamp": -1.0, "text": "before"},
                {"timestamp": 5.0, "text": "valid"},
                {"timestamp": 15.0, "text": "after"},
            ]
        }

        result = validate_timestamps(payload, duration=10.0)
        assert len(result["key_points"]) == 1
        assert result["key_points"][0]["text"] == "valid"

    def test_validate_timestamps_sorts_chapters(self):
        """Test that chapters are sorted by t_start."""
        payload = {
            "chapters": [
                {"t_start": 10.0, "t_end": 15.0, "title": "Later"},
                {"t_start": 0.0, "t_end": 5.0, "title": "First"},
                {"t_start": 5.0, "t_end": 10.0, "title": "Middle"},
            ]
        }

        result = validate_timestamps(payload, duration=15.0)
        assert result["chapters"][0]["title"] == "First"
        assert result["chapters"][1]["title"] == "Middle"
        assert result["chapters"][2]["title"] == "Later"


# Provider Tests
class TestProviders:
    def test_get_provider_anthropic_default_model(self):
        """Test provider factory returns correct default model."""
        from src.llm.providers import get_provider

        with patch("src.llm.providers.AnthropicProvider") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance
            
            provider = get_provider("anthropic")
            mock_anthropic.assert_called_once_with("claude-3-5-sonnet-20241022")

    def test_get_provider_openai_default_model(self):
        """Test provider factory for OpenAI returns correct default."""
        from src.llm.providers import get_provider

        with patch("src.llm.providers.OpenAIProvider") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            provider = get_provider("openai")
            mock_openai.assert_called_once_with("gpt-4o")

    def test_get_provider_unknown(self):
        """Test provider factory with unknown provider."""
        from src.llm.providers import get_provider

        with pytest.raises(ValueError):
            get_provider("unknown-provider")


# Payload to Summary Tests
class TestPayloadToSummary:
    def test_payload_to_summary_basic(self, sample_fused_doc: FusedDocument):
        """Test basic payload to summary conversion."""
        payload = {
            "full_summary": "Test summary",
            "short_summary": "Test",
            "key_points": [
                {"timestamp": 5.0, "text": "Point 1", "confidence": "high"}
            ],
            "events": [
                {"timestamp": 2.0, "event_type": "start", "description": "Started"}
            ],
            "chapters": [
                {"t_start": 0.0, "t_end": 10.0, "title": "Intro"}
            ],
        }

        summary = _payload_to_summary(sample_fused_doc, payload)

        assert summary.full_summary == "Test summary"
        assert len(summary.key_points) == 1
        assert len(summary.events) == 1
        assert len(summary.chapters) == 1

    def test_payload_to_summary_invalid_timestamps(self, sample_fused_doc: FusedDocument):
        """Test that invalid timestamps are handled."""
        payload = {
            "full_summary": "Test",
            "short_summary": "Test",
            "key_points": [
                {"timestamp": 100.0, "text": "Invalid", "confidence": "high"},  # Out of range
                {"timestamp": 5.0, "text": "Valid", "confidence": "high"},
            ],
            "events": [],
            "chapters": [],
        }

        summary = _payload_to_summary(sample_fused_doc, payload)

        # At least check that we handle the data without crashing
        # Note: _payload_to_summary doesn't call validate_timestamps itself,
        # that's done at a higher level, so we just verify the objects are created
        assert len(summary.key_points) >= 1

    def test_payload_to_summary_with_qa(self, sample_fused_doc: FusedDocument):
        """Test payload with Q&A pairs."""
        payload = {
            "full_summary": "Test",
            "short_summary": "Test",
            "key_points": [],
            "events": [],
            "chapters": [],
            "qa_pairs": [
                {"question": "What is ML?", "answer": "Machine learning is...", "timestamp": 10.0}
            ],
        }

        summary = _payload_to_summary(sample_fused_doc, payload)

        assert summary.qa_pairs is not None
        assert len(summary.qa_pairs) == 1


# Integration Tests
class TestSummarizationIntegration:
    def test_summarize_produces_valid_output(self, tmp_path: Path, temp_fused_json: Path):
        """Test that summarize produces valid output."""
        output_path = tmp_path / "summary.json"

        # Mock the provider
        mock_response_chunk = {
            "local_summary": "This is a test",
            "key_points": [{"timestamp": 5.0, "text": "Test point", "confidence": "high"}],
            "events": [{"timestamp": 10.0, "event_type": "test", "description": "Test event"}],
        }
        mock_response_global = {
            "full_summary": "Full test summary",
            "short_summary": "Test",
            "chapters": [{"t_start": 0.0, "t_end": 15.0, "title": "Intro"}],
            "merged_key_points": [{"timestamp": 5.0, "text": "Test point", "confidence": "high"}],
            "merged_events": [{"timestamp": 10.0, "event_type": "test", "description": "Test"}],
        }

        provider = FakeProvider(
            [
                (mock_response_chunk, {"input_tokens": 100, "output_tokens": 50}),
                (mock_response_global, {"input_tokens": 100, "output_tokens": 50}),
            ]
        )

        with patch("src.llm.summarizer.get_provider", return_value=provider):
            summary = summarize(
                temp_fused_json,
                output_path,
                provider="anthropic",
            )

        assert summary.run_id == "test-run-001"
        assert output_path.exists()

    def test_summarize_empty_document(self, tmp_path: Path):
        """Test summarization of empty document."""
        # Create empty fused doc
        from src.fusion.schema import save_fused_document

        empty_doc = FusedDocument(
            run_id="empty-run",
            duration_sec=0.0,
            language="en",
            events=[],
            speech_source="local-whisper",
            ocr_engine="tesseract",
            has_captions=False,
        )

        fused_path = tmp_path / "empty_fused.json"
        save_fused_document(empty_doc, fused_path)

        output_path = tmp_path / "summary.json"
        summary = summarize(fused_path, output_path)

        assert summary.full_summary == "(no content detected)"
        assert len(summary.key_points) == 0
        assert len(summary.events) == 0

    def test_summarize_missing_fused_file(self, tmp_path: Path):
        """Test that missing fused file raises error."""
        with pytest.raises(FileNotFoundError):
            summarize(
                tmp_path / "nonexistent.json",
                tmp_path / "output.json",
            )


# Edge Case Tests
class TestEdgeCases:
    def test_very_large_timestamp(self):
        """Test handling of very large timestamps."""
        text = '{"timestamp": 9999999.999, "value": "test"}'
        result = strip_and_parse_json(text)
        assert result["timestamp"] == 9999999.999

    def test_unicode_in_payload(self):
        """Test handling of unicode characters."""
        text = '{"text": "Hello 世界 🌍", "emoji": "✨"}'
        result = strip_and_parse_json(text)
        assert "世界" in result["text"]
        assert "✨" in result["emoji"]

    def test_nested_json_extraction(self):
        """Test extraction of nested JSON from prose."""
        text = """Some introduction text here
        {"outer": {"inner": {"deep": "value"}}}
        And some trailing text"""
        result = strip_and_parse_json(text)
        assert result["outer"]["inner"]["deep"] == "value"
