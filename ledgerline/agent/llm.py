"""The language-model seam.

Third time this repo has needed the same shape, after embeddings and
cross-encoder reranking, so it is built the same way on purpose: a protocol, a
real provider that talks to the network, and a cache keyed by content hash that
CI reads instead. A miss is fatal and names the command that fixes it, because
a silent fallback to a different model means half a run scored under one system
and half under another.

What is different here, and why the graph cares: a missing embedding is a bug,
but a missing *model* is an operating condition. The API key may be absent, the
provider may be down, the org may be out of quota. So `ModelUnavailable` is a
first-class exception the graph catches and turns into `degraded` -- not into a
refusal, and not into a crash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from shared.logging import get_logger

log = get_logger(__name__)

#: The model this system is written against. Pricing below is per million
#: tokens and is used to record cost per run; both move together and both
#: belong in one place rather than scattered through call sites.
DEFAULT_MODEL = "claude-opus-5"
PRICING_USD_PER_MTOK = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class ModelUnavailable(RuntimeError):
    """No model could serve this call.

    Distinct from a bad response. This means the call never happened -- no key,
    no network, no quota -- and the correct response is to degrade the run
    rather than to claim the question was unanswerable.
    """


class CompletionCacheMiss(KeyError):
    """A prompt was not in the committed cache."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str = DEFAULT_MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    #: Populated when the provider declined on safety grounds. The graph treats
    #: this as a refusal with a reason, never as an answer.
    refusal_category: str | None = None

    @property
    def cost_usd(self) -> float:
        """Cost of this call. Cache reads bill at roughly a tenth of input."""
        rate_in, rate_out = PRICING_USD_PER_MTOK.get(
            self.model, PRICING_USD_PER_MTOK[DEFAULT_MODEL]
        )
        billable_in = self.input_tokens + self.cached_input_tokens * 0.1
        return (billable_in * rate_in + self.output_tokens * rate_out) / 1_000_000


@runtime_checkable
class LanguageModel(Protocol):
    model_name: str

    def complete(self, prompt: str, *, system: str | None = None) -> Completion: ...


def prompt_key(model: str, system: str | None, prompt: str) -> str:
    """Content hash of everything that determines the response.

    The model is part of the key. Without it, switching models would silently
    reuse the previous model's answers and the ablation would measure nothing.
    """
    joined = "\x00".join([model, system or "", " ".join(prompt.split())])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


class AnthropicModel:
    """Real completions via the Anthropic API.

    Thinking is left adaptive, which is the default on this model, and `effort`
    is exposed because the analysts have genuinely different budgets: reading
    one chunk to extract a claim is not the same job as reconciling a table
    against a transcript.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        effort: str = "medium",
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise ModelUnavailable(
                'the anthropic sdk is not installed. `pip install -e ".[ledgerline]"`.'
            ) from exc

        from shared.config import get_settings

        key = api_key or get_settings().anthropic_api_key
        if not key:
            raise ModelUnavailable(
                "no ANTHROPIC_API_KEY configured -- runs needing a model will degrade"
            )
        self._client = anthropic.Anthropic(api_key=key)
        self.model_name = model_name
        self._effort = effort
        self._max_tokens = max_tokens

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        import anthropic

        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=self._max_tokens,
                system=system or anthropic.NOT_GIVEN,
                output_config={"effort": self._effort},
                messages=[{"role": "user", "content": prompt}],
            )
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
            # Both are "try again later" rather than "this question is
            # unanswerable", so they degrade the run instead of refusing it.
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ModelUnavailable(f"provider {exc.status_code}") from exc
            raise

        usage = response.usage
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return Completion(
                text="",
                model=response.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                refusal_category=getattr(details, "category", None) or "unspecified",
            )

        return Completion(
            text="".join(b.text for b in response.content if b.type == "text"),
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )


@dataclass
class CachedModel:
    """Reads completions from a committed JSON file. What CI uses."""

    responses: dict[str, dict]
    model_name: str = DEFAULT_MODEL
    fallback: LanguageModel | None = None

    @classmethod
    def from_json(cls, path: str | Path, fallback: LanguageModel | None = None) -> CachedModel:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"completion cache missing: {resolved}. Run `ledgerline warm-cache`."
            )
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return cls(
            responses=payload.get("responses", {}),
            model_name=payload.get("model", DEFAULT_MODEL),
            fallback=fallback,
        )

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        key = prompt_key(self.model_name, system, prompt)
        record = self.responses.get(key)
        if record is None:
            if self.fallback is not None:
                return self.fallback.complete(prompt, system=system)
            raise CompletionCacheMiss(
                f"prompt not cached ({key}) -- run `ledgerline warm-cache`. "
                f"First 60 chars: {prompt[:60]!r}"
            )
        return Completion(
            text=record["text"],
            model=record.get("model", self.model_name),
            input_tokens=record.get("input_tokens", 0),
            output_tokens=record.get("output_tokens", 0),
        )


def save_completion_cache(
    path: str | Path,
    prompts: list[tuple[str, str | None]],
    model: LanguageModel,
) -> Path:
    """Run every (prompt, system) pair once and commit the responses."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    responses: dict[str, dict] = {}
    for prompt, system in prompts:
        completion = model.complete(prompt, system=system)
        responses[prompt_key(model.model_name, system, prompt)] = {
            "text": completion.text,
            "model": completion.model,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
        }

    resolved.write_text(
        json.dumps(
            {"model": model.model_name, "responses": responses}, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    log.info("llm.cache.saved", path=str(resolved), n=len(responses))
    return resolved
