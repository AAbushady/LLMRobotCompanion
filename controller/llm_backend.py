"""Swappable LLM backend: abstract interface + Claude and OpenAI-compatible implementations."""

import abc
import json
import logging
import time

import requests

from . import config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Non-retryable LLM error."""
    pass


class LLMBackend(abc.ABC):
    """Abstract LLM backend interface."""

    @abc.abstractmethod
    def complete(self, messages, system_prompt=None, max_tokens=None):
        """Send messages to the LLM and return the text response.

        Args:
            messages: list of {"role": str, "content": str} dicts.
            system_prompt: optional system prompt string.
            max_tokens: optional override for max response tokens.

        Returns:
            str: The LLM's text response.

        Raises:
            LLMError: On non-retryable failures.
        """

    def _retry_loop(self, fn):
        """Call fn() with retries on transient HTTP errors.

        Retries on status 429, 500, 502, 503, 529.
        Raises LLMError on non-retryable errors or after exhausting retries.
        """
        retryable = {429, 500, 502, 503, 529}
        last_error = None

        for attempt in range(1 + config.LLM_RETRIES):
            try:
                response = fn()
                if response.status_code == 200:
                    return response

                if response.status_code in retryable:
                    last_error = "HTTP {}: {}".format(
                        response.status_code, response.text[:200]
                    )
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s",
                        attempt + 1, 1 + config.LLM_RETRIES, last_error
                    )
                    if attempt < config.LLM_RETRIES:
                        delay = min(2 ** attempt, 10)
                        time.sleep(delay)
                    continue

                # Non-retryable error
                raise LLMError("HTTP {}: {}".format(
                    response.status_code, response.text[:500]
                ))

            except requests.RequestException as exc:
                last_error = str(exc)
                logger.warning(
                    "LLM request exception (attempt %d/%d): %s",
                    attempt + 1, 1 + config.LLM_RETRIES, last_error
                )
                if attempt < config.LLM_RETRIES:
                    delay = min(2 ** attempt, 10)
                    time.sleep(delay)
                continue

        raise LLMError("Exhausted retries. Last error: {}".format(last_error))


class ClaudeBackend(LLMBackend):
    """Claude Messages API backend via raw HTTP."""

    def __init__(self):
        if not config.CLAUDE_API_KEY:
            raise LLMError(
                "ANTHROPIC_API_KEY environment variable is not set"
            )
        self._api_key = config.CLAUDE_API_KEY
        self._model = config.CLAUDE_MODEL
        self._url = config.CLAUDE_API_URL
        self._version = config.CLAUDE_API_VERSION

    def complete(self, messages, system_prompt=None, max_tokens=None):
        if max_tokens is None:
            max_tokens = config.LLM_MAX_TOKENS

        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": config.LLM_TEMPERATURE,
            "messages": messages,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._version,
            "content-type": "application/json",
        }

        def do_request():
            return requests.post(
                self._url,
                headers=headers,
                data=json.dumps(payload),
                timeout=config.LLM_TIMEOUT,
            )

        response = self._retry_loop(do_request)
        body = response.json()

        # Extract text from Claude Messages API response
        content = body.get("content", [])
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block["text"])

        if not parts:
            raise LLMError("No text content in Claude response: {}".format(
                json.dumps(body)[:300]
            ))

        return "\n".join(parts)


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible chat completions backend (OpenRouter, Aphrodite, vLLM, etc.)."""

    def __init__(self):
        if not config.OPENAI_API_URL:
            raise LLMError(
                "OPENAI_API_URL environment variable is not set"
            )
        self._url = config.OPENAI_API_URL
        self._api_key = config.OPENAI_API_KEY
        self._model = config.OPENAI_MODEL

    def complete(self, messages, system_prompt=None, max_tokens=None):
        if max_tokens is None:
            max_tokens = config.LLM_MAX_TOKENS

        # OpenAI format: system message prepended to messages list
        oai_messages = []
        if system_prompt:
            oai_messages.append({"role": "system", "content": system_prompt})
        oai_messages.extend(messages)

        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": config.LLM_TEMPERATURE,
            "messages": oai_messages,
        }

        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = "Bearer {}".format(self._api_key)

        def do_request():
            return requests.post(
                self._url,
                headers=headers,
                data=json.dumps(payload),
                timeout=config.LLM_TIMEOUT,
            )

        response = self._retry_loop(do_request)
        body = response.json()

        # Extract text from OpenAI-compatible response
        choices = body.get("choices", [])
        if not choices:
            raise LLMError(
                "No choices in OpenAI response: {}".format(
                    json.dumps(body)[:300]
                )
            )

        message = choices[0].get("message", {})
        text = message.get("content", "")
        if not text:
            raise LLMError("Empty content in OpenAI response")

        return text


def create_backend(name=None):
    """Factory: create an LLM backend by name.

    Args:
        name: "claude" or "openai". Defaults to config.LLM_BACKEND.

    Returns:
        LLMBackend instance.

    Raises:
        LLMError: If the backend name is unknown or configuration is missing.
    """
    if name is None:
        name = config.LLM_BACKEND

    name = name.lower().strip()

    if name == "claude":
        return ClaudeBackend()
    elif name == "openai":
        return OpenAIBackend()
    else:
        raise LLMError("Unknown LLM backend: '{}'".format(name))
