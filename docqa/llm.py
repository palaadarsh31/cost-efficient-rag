"""Minimal client for OpenAI-compatible /chat/completions endpoints.

Works with OpenAI, Gemini's OpenAI endpoint, Groq, Together, OpenRouter,
Ollama, vLLM and llama.cpp's server. Plain `requests` so there is no SDK
version to fight with.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import requests

from .chunking import count_tokens

log = logging.getLogger(__name__)

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    def __init__(self, message: str, *, completion: Completion | None = None):
        super().__init__(message)
        self.completion = completion


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str | None = None
    usage_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float | None = 0.0, max_tokens: int = 400, timeout: float = 120.0,
                 max_retries: int = 6, extra_body: dict | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_key = "max_tokens"
        self._disabled_parameters: set[str] = set()
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def for_generation(cls, settings) -> "ChatClient":
        return cls(settings.llm_base_url, settings.llm_model, settings.llm_api_key,
                   settings.llm_temperature, settings.llm_max_tokens, settings.llm_timeout,
                   extra_body=_extra_body("LLM_EXTRA_BODY"))

    @classmethod
    def for_judge(cls, settings) -> "ChatClient":
        temperature = None if settings.llm_temperature is None else 0.0
        return cls(settings.judge_base_url, settings.judge_model, settings.judge_api_key,
                   temperature, 800, settings.llm_timeout,
                   extra_body=_extra_body("JUDGE_EXTRA_BODY") or _extra_body("LLM_EXTRA_BODY"))

    def _payload(self, messages: list[dict], overrides: dict) -> dict:
        payload = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            payload[self.max_tokens_key] = self.max_tokens
        payload.update(self.extra_body)
        payload.update(overrides)
        for name in self._disabled_parameters:
            payload.pop(name, None)
        if self.max_tokens_key == "max_completion_tokens" and "max_tokens" in payload:
            requested_limit = payload.pop("max_tokens")
            if "max_completion_tokens" not in self.extra_body and "max_completion_tokens" not in overrides:
                payload["max_completion_tokens"] = requested_limit
        return payload

    def snapshot(self) -> dict:
        """Effective request settings without credentials or request content."""
        endpoint = urlsplit(self.url)
        hostname = endpoint.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        if endpoint.port:
            hostname += f":{endpoint.port}"
        payload = self._payload([], {})
        allowed = {"model", "temperature", "top_p", "max_tokens", "max_completion_tokens",
                   "seed", "presence_penalty", "frequency_penalty", "reasoning_effort", "stop"}
        return {"endpoint": urlunsplit((endpoint.scheme, hostname, endpoint.path, "", "")),
                "parameters": {key: value for key, value in payload.items() if key in allowed},
                "timeout_seconds": self.timeout, "max_retries": self.max_retries}

    def _adapt_to(self, resp) -> bool:
        """Some models reject standard params (e.g. newer Claude models refuse
        `temperature`, OpenAI reasoning models want `max_completion_tokens`).
        Drop or rename the offending param once and remember it."""
        if resp.status_code != 400:
            return False
        body = resp.text.lower()
        if "temperature" in body and "temperature" not in self._disabled_parameters:
            log.warning("%s rejected temperature; sending requests without it", self.model)
            self.temperature = None
            self._disabled_parameters.add("temperature")
            return True
        if "max_tokens" in body and "max_completion_tokens" in body and self.max_tokens_key == "max_tokens":
            log.warning("%s wants max_completion_tokens instead of max_tokens", self.model)
            self.max_tokens_key = "max_completion_tokens"
            return True
        return False

    def complete(self, messages: list[dict], **overrides) -> Completion:
        payload = self._payload(messages, overrides)
        started = time.perf_counter()
        resp = None
        adapted = 0
        attempt = 0
        while attempt <= self.max_retries:
            try:
                resp = self.session.post(self.url, json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"could not reach {self.url}: {exc}") from exc
                self._sleep(attempt, None)
                attempt += 1
                continue
            if adapted < 2 and self._adapt_to(resp):
                adapted += 1
                payload = self._payload(messages, overrides)
                continue
            if resp.status_code in RETRYABLE and attempt < self.max_retries:
                log.warning("LLM returned %s, retrying (attempt %d)", resp.status_code, attempt + 1)
                self._sleep(attempt, resp.headers.get("Retry-After"))
                attempt += 1
                continue
            break

        if resp is None or resp.status_code >= 400:
            body = resp.text[:500] if resp is not None else ""
            raise LLMError(f"{self.url} -> HTTP {getattr(resp, 'status_code', '?')}: {body}")

        try:
            data = resp.json()
        except (ValueError, TypeError) as exc:
            raise LLMError("LLM returned invalid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise LLMError("LLM response has no completion choices")
        choice = data["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise LLMError("LLM response has an invalid completion choice")
        content = choice["message"].get("content")
        if content is not None and not isinstance(content, str):
            raise LLMError("LLM response content is not text")
        text = (content or "").strip()
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            raise LLMError("LLM response has invalid token usage")
        estimated = usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None:
            prompt_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
        if completion_tokens is None:
            completion_tokens = count_tokens(text)

        if any(type(value) is not int or value < 0 for value in (prompt_tokens, completion_tokens)):
            raise LLMError("LLM response token usage must be nonnegative integers")
        completion = Completion(
            text=text,
            model=data.get("model") or self.model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            finish_reason=choice.get("finish_reason"),
            usage_estimated=estimated,
        )
        if not text:
            raise LLMError("LLM returned an empty completion; this is not an intentional abstention",
                           completion=completion)
        if completion.finish_reason in ("length", "max_tokens", "content_filter", "tool_calls", "error"):
            raise LLMError(f"LLM completion is incomplete (finish_reason={completion.finish_reason})",
                           completion=completion)
        return completion

    @staticmethod
    def _sleep(attempt: int, retry_after: str | None) -> None:
        delay = min(60.0, 2.0 ** attempt + random.uniform(0, 1))
        if retry_after:
            try:
                delay = min(90.0, max(delay, float(retry_after)))
            except ValueError:
                pass
        time.sleep(delay)


def _extra_body(var: str) -> dict:
    """Provider-specific request fields, e.g. LLM_EXTRA_BODY='{"reasoning_effort": "none"}'."""
    raw = os.getenv(var, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{var} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{var} must be a JSON object")
    return value
