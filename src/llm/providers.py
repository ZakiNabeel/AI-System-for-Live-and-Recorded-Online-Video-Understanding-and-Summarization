"""Base provider interface and implementations."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.parsing import LLMOutputParseError, strip_and_parse_json

LOGGER = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """
        Call LLM and parse JSON response.

        Args:
            system: System message
            user: User message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            (parsed_json, usage_dict)

        Raises:
            LLMOutputParseError: If output cannot be parsed
        """
        ...


class AnthropicProvider:
    """Anthropic Claude provider."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize with model name."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required")

        self.client = Anthropic()
        self.model = model
        LOGGER.info(f"Initialized AnthropicProvider with {model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Call Claude and parse JSON response."""
        LOGGER.debug(f"Calling {self.model} (max_tokens={max_tokens})")

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        text = resp.content[0].text.strip()
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }

        try:
            payload = strip_and_parse_json(text)
        except LLMOutputParseError as e:
            LOGGER.error(f"Failed to parse response: {e.text[:200]}")
            raise

        return payload, usage


class OpenAIProvider:
    """OpenAI GPT provider."""

    def __init__(self, model: str = "gpt-4o"):
        """Initialize with model name."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required")

        self.client = OpenAI()
        self.model = model
        LOGGER.info(f"Initialized OpenAIProvider with {model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Call GPT and parse JSON response."""
        LOGGER.debug(f"Calling {self.model} (max_tokens={max_tokens})")

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )

        text = resp.choices[0].message.content.strip()
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }

        try:
            payload = strip_and_parse_json(text)
        except LLMOutputParseError as e:
            LOGGER.error(f"Failed to parse response: {e.text[:200]}")
            raise

        return payload, usage


class OllamaProvider:
    """Ollama local LLM provider."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """Initialize with model name and base URL."""
        self.model = model
        self.base_url = base_url
        LOGGER.info(f"Initialized OllamaProvider with {model} at {base_url}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Call Ollama and parse JSON response."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required")

        LOGGER.debug(f"Calling {self.model} via Ollama (max_tokens={max_tokens})")

        prompt = f"{system}\n\n{user}"

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    "stream": False,
                },
                timeout=300,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            LOGGER.error(f"Ollama request failed: {e}")
            raise

        text = resp.json().get("response", "").strip()
        usage = {"input_tokens": 0, "output_tokens": 0}  # Ollama doesn't report usage

        try:
            payload = strip_and_parse_json(text)
        except LLMOutputParseError as e:
            LOGGER.error(f"Failed to parse response: {e.text[:200]}")
            raise

        return payload, usage


class GeminiProvider:
    """Google Gemini provider (google-genai SDK)."""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai package required: pip install google-genai"
            )

        import os
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.model = model
        LOGGER.info(f"Initialized GeminiProvider with {model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Call Gemini and parse a JSON response."""
        LOGGER.debug(f"Calling {self.model} (max_tokens={max_tokens})")

        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        text = (getattr(response, "text", "") or "").strip()

        meta = getattr(response, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        }

        try:
            payload = strip_and_parse_json(text)
        except LLMOutputParseError as e:
            LOGGER.error(f"Failed to parse Gemini response: {e.text[:200]}")
            raise

        return payload, usage


def get_provider(
    provider_name: str,
    model: str | None = None,
) -> LLMProvider:
    """
    Factory function to get a provider instance.

    Args:
        provider_name: "anthropic", "openai", "ollama", or "gemini"
        model: Override default model

    Returns:
        Provider instance

    Raises:
        ValueError: Unknown provider
    """
    defaults = {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4o",
        "ollama": "llama2",
        "gemini": "gemini-2.5-flash-lite",
    }

    if provider_name == "anthropic":
        return AnthropicProvider(model or defaults["anthropic"])
    elif provider_name == "openai":
        return OpenAIProvider(model or defaults["openai"])
    elif provider_name == "ollama":
        return OllamaProvider(model or defaults["ollama"])
    elif provider_name == "gemini":
        return GeminiProvider(model or defaults["gemini"])
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
