"""Thin HTTP client for an OpenAI-compatible chat-completions endpoint.

This targets the **OpenAI-compatible** schema (``POST {base_url}/chat/completions``
with ``{"model", "messages", "temperature", "max_tokens"}``) rather than
Ollama's native ``/api/generate``. The point of that choice is portability:
switching from a local Ollama to a hosted provider (OpenAI, Together, Groq, ...)
is a change to ``LLM_BASE_URL`` / ``LLM_MODEL`` / ``LLM_API_KEY`` in ``.env``,
not a rewrite of this module.

No ``openai`` SDK: the endpoint is a plain JSON POST and calling it directly
keeps the dependency surface small and the wire protocol visible. Retries reuse
the GROBID policy (``parsing.grobid.max_retries`` / ``retry_backoff_seconds``),
the same way :mod:`ragvlc.parsing.crossref` does -- retry on timeout,
connection error and 5xx; fail fast on 4xx.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from ragvlc.config import get_experiment, get_settings

# A chat message is just ``{"role": "system"|"user"|"assistant", "content": str}``;
# kept as a plain dict rather than a model to stay close to the wire format.
ChatMessage = dict[str, str]


class GenerationError(RuntimeError):
    """Raised when the LLM endpoint cannot be reached or refuses the request.

    Always names the model and the endpoint URL so a failure is traceable
    without re-running with extra logging.
    """


@dataclass(frozen=True)
class ChatCompletion:
    """The part of a chat-completions response we use, plus the raw body."""

    text: str  # choices[0].message.content
    model: str  # the model the server reports having run
    finish_reason: str | None  # "stop", "length", ...
    raw: dict


class LLMClient:
    """Holds the endpoint config so a caller (the API service, a batch script)
    constructs one and reuses it rather than rebuilding per request."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout: float,
        max_retries: int,
        backoff_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    @classmethod
    def from_config(cls) -> LLMClient:
        settings = get_settings()
        grobid_cfg = get_experiment().parsing.grobid  # "the GROBID retry policy"
        generation_cfg = get_experiment().generation
        return cls(
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_api_key,
            timeout=generation_cfg.timeout_seconds,
            max_retries=grobid_cfg.max_retries,
            backoff_seconds=grobid_cfg.retry_backoff_seconds,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletion:
        """POST one chat completion and return its text.

        Retries timeout / connection error / 5xx up to ``max_retries`` extra
        times with linear backoff. A 4xx (bad request, model not pulled, ...)
        fails immediately -- an identical request fails identically.
        """
        url = self.endpoint
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        max_attempts = 1 + self._max_retries

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
                response.raise_for_status()
                completion = _parse_completion(response.json(), fallback_model=self._model)
                if completion.finish_reason == "length" and not completion.text.strip():
                    # The whole token budget was consumed before any answer text
                    # was emitted -- common with a reasoning model whose chain of
                    # thought overruns `max_tokens`. Retrying identically won't
                    # help; raise so the caller can raise generation.max_tokens.
                    raise GenerationError(
                        f"model {self._model!r} at {url} hit the {max_tokens}-token limit "
                        f"before producing any answer text (raise generation.max_tokens)"
                    )
                return completion
            except httpx.TimeoutException as exc:
                last_error = exc  # transient -- retry
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code < 500:
                    raise GenerationError(
                        f"model {self._model!r} at {url} rejected the request with "
                        f"{exc.response.status_code}: {exc.response.text[:500]!r}"
                    ) from exc
                # 5xx -- transient -- retry
            except httpx.HTTPError as exc:
                last_error = exc  # connection refused/reset etc. -- retry
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                # Malformed / non-JSON body -- an identical call won't parse
                # any better, so fail rather than retry.
                raise GenerationError(
                    f"model {self._model!r} at {url} returned an unparseable response: {exc}"
                ) from exc

            if attempt < max_attempts:
                time.sleep(self._backoff_seconds * attempt)

        raise GenerationError(
            f"model {self._model!r} at {url} did not respond after {max_attempts} attempt(s): {last_error}"
        ) from last_error


def _parse_completion(body: dict, *, fallback_model: str) -> ChatCompletion:
    """Pull ``choices[0].message.content`` out of an OpenAI-shaped response.

    Raises ``KeyError`` / ``IndexError`` / ``TypeError`` on a body that isn't
    that shape; the caller turns those into a ``GenerationError``.
    """
    choice = body["choices"][0]
    text = choice["message"]["content"]
    if not isinstance(text, str):
        raise TypeError(f"choices[0].message.content is {type(text).__name__}, not str")
    return ChatCompletion(
        text=text,
        model=body.get("model") or fallback_model,
        finish_reason=choice.get("finish_reason"),
        raw=body,
    )
