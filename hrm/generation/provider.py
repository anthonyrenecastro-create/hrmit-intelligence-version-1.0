"""HRM-backed inference provider.

Replaces the mock with real generation driven by HRM state:
- Text: synthesized from memory retrieval + HRM cognition
- Vision: generated from HRM field dynamics
"""
from __future__ import annotations

import hashlib
import time

from hrm.integration.types import (
    InferenceContext,
    InferenceRequest,
    InferenceResult,
)


class HRMProvider:
    """Inference provider backed by the HRM's generative dynamics."""

    provider_id = "hrm_local"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._text_gen = None
        self._image_gen = None

    def _get_text_gen(self):
        if self._text_gen is None:
            from hrm.generation.hrm_text import HRMCognitionText
            from pathlib import Path
            import numpy as np
            # Resolve weights relative to repo root (not CWD)
            pkg_root = Path(__file__).resolve().parent.parent.parent
            w = pkg_root / "hrm_generated" / "hrm_cogtext.npz"
            if w.exists():
                # Auto-detect cog_dim from weights
                with np.load(w) as z:
                    cog_dim = int(z["W_gates"].shape[0]) if "W_gates" in z else 128
                self._text_gen = HRMCognitionText(cog_dim=cog_dim, seed=self.seed)
                self._text_gen.load(w.parent)
            else:
                self._text_gen = HRMCognitionText(seed=self.seed)
        return self._text_gen

    def _get_image_gen(self):
        if self._image_gen is None:
            from hrm.generation.vision_generator import HRMImageGenerator
            self._image_gen = HRMImageGenerator(seed=self.seed)
        return self._image_gen

    async def infer(self, request: InferenceRequest, context: InferenceContext) -> InferenceResult:
        t0 = time.perf_counter()
        seed = request.seed if request.seed is not None else self.seed
        prompt = "\n".join([f"{m['role']}:{m['content']}" for m in request.messages])
        # Extract the user query (last user message)
        user_text = ""
        for m in reversed(request.messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break

        # Generate via HRM text generator
        try:
            gen = self._get_text_gen()
            # Use a short generation; temperature from request
            temp = float(request.temperature) if request.temperature else 0.7
            max_tok = min(request.token_limit or 200, 500)
            draft = gen.generate(prompt=user_text[:200], max_tokens=max_tok, temperature=temp, seed=seed)
            if not draft.strip():
                draft = f"[HRM generated empty response for: {user_text[:60]}]"
            confidence = 0.65
            safety = "ok"
            parse_status = "ok"
        except Exception as e:
            draft = f"[HRM generation unavailable: {type(e).__name__}]"
            confidence = 0.0
            safety = "unknown"
            parse_status = "error"

        digest = hashlib.sha256(f"{seed}:{prompt}".encode()).hexdigest()[:12]
        payload = {
            "intent": "answer_user_query",
            "response_draft": draft,
            "confidence": confidence,
            "provider": self.provider_id,
        }
        return InferenceResult(
            raw_output=draft,
            parsed_output=payload,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=max(1, len(draft.split())),
            latency_seconds=time.perf_counter() - t0,
            model_id=request.model_id,
            provider_id=self.provider_id,
            sampling={"temperature": request.temperature},
            seed=seed,
            parse_status=parse_status,
            safety_status=safety,
            retry_history=(),
            provenance={"digest": digest, "mode": context.mode.value, "hrm_backed": True},
        )
