from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Protocol, Any

from .types import InferenceContext, InferenceRequest, InferenceResult, OperatingMode
from .types import ProviderSelection

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None


class InferenceProvider(Protocol):
    provider_id: str

    async def infer(self, request: InferenceRequest, context: InferenceContext) -> InferenceResult:
        ...


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    max_tokens: int
    local_only: bool
    supports_seed: bool


MODEL_PROFILES: dict[str, ModelProfile] = {
    "mock_deterministic": ModelProfile(model_id="mock_deterministic", max_tokens=4096, local_only=True, supports_seed=True),
    "llama3.1:8b": ModelProfile(model_id="llama3.1:8b", max_tokens=8192, local_only=True, supports_seed=True),
    "llama3.2:3b": ModelProfile(model_id="llama3.2:3b", max_tokens=8192, local_only=True, supports_seed=True),
}


class MockDeterministicProvider:
    provider_id = "mock_deterministic"

    async def infer(self, request: InferenceRequest, context: InferenceContext) -> InferenceResult:
        t0 = time.perf_counter()
        seed = request.seed if request.seed is not None else 0
        prompt = "\n".join([f"{m['role']}:{m['content']}" for m in request.messages])
        digest = hashlib.sha256(f"{seed}:{prompt}:{request.model_id}".encode("utf-8")).hexdigest()
        hypothesis = f"h-{digest[:12]}"
        payload = {
            "intent": "answer_user_query",
            "hypotheses": [hypothesis],
            "response_draft": f"Deterministic draft for {request.model_id}: {hypothesis}",
            "confidence": 0.72,
            "memory_write_candidate": f"mem::{hypothesis}",
            "tool_action_candidate": None,
        }
        raw = json.dumps(payload)
        return InferenceResult(
            raw_output=raw,
            parsed_output=payload,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=max(1, len(raw.split())),
            latency_seconds=time.perf_counter() - t0,
            model_id=request.model_id,
            provider_id=self.provider_id,
            sampling={"temperature": request.temperature},
            seed=seed,
            parse_status="ok",
            safety_status="ok",
            retry_history=(),
            provenance={"digest": digest, "mode": context.mode.value},
        )


class OllamaProvider:
    provider_id = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")

    async def infer(self, request: InferenceRequest, context: InferenceContext) -> InferenceResult:
        if httpx is None:
            return InferenceResult(
                raw_output="",
                parsed_output=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=0.0,
                model_id=request.model_id,
                provider_id=self.provider_id,
                sampling={"temperature": request.temperature},
                seed=request.seed,
                parse_status="error",
                safety_status="unknown",
                retry_history=("httpx_unavailable",),
                provenance={"base_url": self.base_url},
                error_state="httpx_unavailable",
            )

        t0 = time.perf_counter()
        model_available = await self._model_available(request.model_id)
        if not model_available:
            return InferenceResult(
                raw_output="",
                parsed_output=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=time.perf_counter() - t0,
                model_id=request.model_id,
                provider_id=self.provider_id,
                sampling={"temperature": request.temperature},
                seed=request.seed,
                parse_status="error",
                safety_status="unknown",
                retry_history=("model_unavailable",),
                provenance={"base_url": self.base_url, "model_id": request.model_id},
                error_state="model_unavailable",
            )
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in request.messages])
        body = {
            "model": request.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                **({"seed": request.seed} if request.seed is not None else {}),
            },
        }
        retries: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=request.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=body)
                response.raise_for_status()
                payload = response.json()
            raw_text = str(payload.get("response", "")).strip()
            parsed = _best_effort_parse_json(raw_text)
            return InferenceResult(
                raw_output=raw_text,
                parsed_output=parsed,
                prompt_tokens=int(payload.get("prompt_eval_count", 0)),
                completion_tokens=int(payload.get("eval_count", 0)),
                latency_seconds=time.perf_counter() - t0,
                model_id=request.model_id,
                provider_id=self.provider_id,
                sampling={"temperature": request.temperature},
                seed=request.seed,
                parse_status="ok" if parsed is not None else "text_only",
                safety_status="ok",
                retry_history=tuple(retries),
                provenance={"base_url": self.base_url, "done": payload.get("done", False)},
            )
        except Exception as error:  # pragma: no cover - network/runtime dependent
            retries.append(type(error).__name__)
            return InferenceResult(
                raw_output="",
                parsed_output=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=time.perf_counter() - t0,
                model_id=request.model_id,
                provider_id=self.provider_id,
                sampling={"temperature": request.temperature},
                seed=request.seed,
                parse_status="error",
                safety_status="unknown",
                retry_history=tuple(retries),
                provenance={"base_url": self.base_url},
                error_state=type(error).__name__,
            )

    async def _model_available(self, model_id: str) -> bool:
        if httpx is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return True
        models = payload.get("models", [])
        return any(str(model.get("name", "")) == model_id for model in models)


def _best_effort_parse_json(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        return None
    return None


def run_inference_sync(provider: InferenceProvider, request: InferenceRequest, context: InferenceContext) -> InferenceResult:
    return asyncio.run(provider.infer(request, context))


def _is_local_provider(provider_id: str) -> bool:
    return provider_id.startswith("mock") or provider_id.startswith("ollama")


def choose_provider(
    *,
    mode: OperatingMode,
    preferred_provider: str,
    providers: dict[str, InferenceProvider],
    allow_remote_fallback: bool = False,
) -> ProviderSelection:
    if preferred_provider in providers:
        provider = providers[preferred_provider]
        if mode == OperatingMode.SOVEREIGN_LOCAL and not _is_local_provider(provider.provider_id):
            raise RuntimeError("Preferred provider is not allowed in sovereign_local mode")
        return ProviderSelection(
            provider_id=provider.provider_id,
            fallback_used=False,
            sovereignty_reduced=False,
            reason="preferred_provider_selected",
        )

    local_provider_ids = [provider_id for provider_id in providers if _is_local_provider(provider_id)]
    if local_provider_ids:
        selected_id = sorted(local_provider_ids)[0]
        return ProviderSelection(
            provider_id=selected_id,
            fallback_used=True,
            sovereignty_reduced=False,
            reason="local_fallback_selected",
        )

    if allow_remote_fallback or mode != OperatingMode.SOVEREIGN_LOCAL:
        if providers:
            selected_id = sorted(providers)[0]
            return ProviderSelection(
                provider_id=selected_id,
                fallback_used=True,
                sovereignty_reduced=not _is_local_provider(selected_id),
                reason="remote_or_alternate_fallback_selected",
            )

    raise RuntimeError("No inference provider available for the requested operating mode")
