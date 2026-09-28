"""Provider-agnostic LLM client plus validated structured (JSON) output."""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- providers
@dataclass
class AnthropicClient:
    api_key: str
    model: str = "claude-haiku-4-5-20251001"
    timeout: float = 60.0
    name: str = "anthropic"

    def complete(self, system: str, prompt: str) -> str:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 1024,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


@dataclass
class OpenAIClient:
    api_key: str
    model: str = "gpt-4o-mini"
    timeout: float = 60.0
    name: str = "openai"

    def complete(self, system: str, prompt: str) -> str:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


@dataclass
class GeminiClient:
    api_key: str
    model: str = "gemini-2.5-flash"
    timeout: float = 60.0
    name: str = "gemini"

    def complete(self, system: str, prompt: str) -> str:
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests: replies from a queue or a function of (system, prompt)."""

    replies: list[str] | Callable[[str, str], str] = field(default_factory=list)
    name: str = "fake"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if callable(self.replies):
            return self.replies(system, prompt)
        if not self.replies:
            raise LLMError("FakeLLM has no scripted replies left")
        return self.replies.pop(0)


PROVIDERS = {
    "anthropic": (AnthropicClient, "ANTHROPIC_API_KEY"),
    "openai": (OpenAIClient, "OPENAI_API_KEY"),
    "gemini": (GeminiClient, "GEMINI_API_KEY"),
}


def client_from_env() -> LLMClient | None:
    """Build a client from GOV_LLM_PROVIDER (+ matching *_API_KEY, optional GOV_LLM_MODEL). None = offline."""
    provider = os.environ.get("GOV_LLM_PROVIDER", "").strip().lower()
    if not provider or provider == "none":
        return None
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown GOV_LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    cls, key_var = PROVIDERS[provider]
    key = os.environ.get(key_var)
    if not key:
        raise LLMError(f"GOV_LLM_PROVIDER={provider} but {key_var} is not set")
    model = os.environ.get("GOV_LLM_MODEL")
    return cls(api_key=key, model=model) if model else cls(api_key=key)


# --------------------------------------------------------------------------- structured output
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(raw: str) -> object:
    """Parse JSON from a model reply, tolerating code fences and leading/trailing prose."""
    text = _FENCE.sub("", raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


@dataclass
class StructuredResult:
    value: BaseModel
    attempts: int
    errors: list[str]


def complete_structured(
    client: LLMClient,
    system: str,
    prompt: str,
    schema: type[T],
    max_repairs: int = 2,
    check: Callable[[T], None] | None = None,
) -> StructuredResult:
    """Ask for JSON matching `schema`; on parse/validation failure, send the error back and retry.

    `check` can apply extra semantic rules (raise ValueError to trigger a repair).
    """
    errors: list[str] = []
    schema_hint = json.dumps(schema.model_json_schema())
    request = f"{prompt}\n\nRespond with JSON only, matching this JSON schema:\n{schema_hint}"
    for attempt in range(1, max_repairs + 2):
        raw = client.complete(system, request)
        try:
            value = schema.model_validate(extract_json(raw))
            if check is not None:
                check(value)
            return StructuredResult(value, attempt, errors)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {str(exc)[:300]}")
            request = (
                f"{prompt}\n\nYour previous reply was invalid.\nReply: {raw[:1500]}\nError: {str(exc)[:500]}\n"
                f"Return corrected JSON only, matching this JSON schema:\n{schema_hint}"
            )
    raise LLMError("Structured output failed after repairs: " + " | ".join(errors))
