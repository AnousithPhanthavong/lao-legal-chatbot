"""
Lao Legal RAG — RotatingClient (real, multi-key, resilient)
===========================================================

Holds one `google.genai.Client` per API key and routes EVERY call through the
tested `call_with_resilience` wrapper (resilient_call.py). This is what makes a
single 503 (server overload) or 429 (key quota) rotate to a fresh key instead
of failing the call or hammering one exhausted key — the root cause of the
503/429 storm seen in the interactive notebook.

Why this module exists
----------------------
The notebook had drifted to a SINGLE-KEY `google.generativeai` adapter with no
retry. Under load that path: (a) dies on transient 503s, and (b) rate-limits
one key to 429 while 7 other keys sit idle. This class uses the NEW unified
`google.genai` SDK (the old one is deprecated) and all keys, with resilience.

It exposes two factory methods that return plain callables matching the
contracts the agent modules already expect:
    * `.embed_query_fn()`  -> EmbedQueryFn  (str -> 768-d L2-normalized list)
    * `.llm_fn()`          -> LLMFn         (str -> str)
so retrieval.py / decompose.py / title_boost.py need ZERO changes.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from resilient_call import call_with_resilience

EMBED_MODEL = "gemini-embedding-001"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
EMBED_DIM = 768


class RotatingClient:
    """One google.genai Client per key, all calls routed through resilience."""

    def __init__(self, api_keys: list[str], *, llm_model: str = DEFAULT_LLM_MODEL):
        if not api_keys:
            raise ValueError("RotatingClient needs at least one API key")
        # Import here so the module imports cleanly even where the SDK is absent
        # (e.g. during static checks); real use requires google-genai installed.
        from google import genai

        self._clients = [genai.Client(api_key=k) for k in api_keys]
        self._idx = 0
        self._llm_model = llm_model
        self._n = len(self._clients)

    # ------------------------------------------------------------------ #
    # Internal: build one zero-arg callable per key for a given operation
    # ------------------------------------------------------------------ #
    def _on_rotate(self, new_idx: int) -> None:
        self._idx = new_idx

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        """Resilient text generation across all keys."""
        model = model or self._llm_model
        fns = [
            (lambda c=c: c.models.generate_content(model=model, contents=prompt).text)
            for c in self._clients
        ]
        return call_with_resilience(
            fns, start_idx=self._idx, on_rotate=self._on_rotate
        )

    def embed(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        """Resilient embedding across all keys; returns L2-normalized 768-d."""
        # Build the config object if the real SDK is present; fall back to a
        # plain dict-like config for fakes/tests where google.genai is absent.
        try:
            from google.genai import types
            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            )
        except Exception:
            config = {"task_type": task_type, "output_dimensionality": EMBED_DIM}

        def _make(c):
            def _call():
                resp = c.models.embed_content(
                    model=EMBED_MODEL,
                    contents=text,
                    config=config,
                )
                return resp.embeddings[0].values
            return _call

        fns = [_make(c) for c in self._clients]
        vec = call_with_resilience(
            fns, start_idx=self._idx, on_rotate=self._on_rotate
        )
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) or 1.0)
        if v.shape[0] != EMBED_DIM:
            raise ValueError(
                f"embedder returned dim {v.shape[0]}, expected {EMBED_DIM}"
            )
        return v.tolist()

    # ------------------------------------------------------------------ #
    # Factories returning the plain callables the agent modules expect
    # ------------------------------------------------------------------ #
    def embed_query_fn(self) -> Callable[[str], list[float]]:
        """Return an EmbedQueryFn (query-side task_type)."""
        def embed_query(text: str) -> list[float]:
            return self.embed(text, task_type="RETRIEVAL_QUERY")
        return embed_query

    def llm_fn(self) -> Callable[[str], str]:
        """Return an LLMFn."""
        def llm(prompt: str) -> str:
            return self.generate(prompt)
        return llm

    # ------------------------------------------------------------------ #
    # Convenience constructor from the GEMINI_KEYS env var
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, var: str = "GEMINI_KEYS", **kw) -> "RotatingClient":
        raw = os.environ.get(var, "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise RuntimeError(
                f"No keys in env var {var!r}. Export GEMINI_KEY_n -> {var} first."
            )
        return cls(keys, **kw)

    def __len__(self) -> int:
        return self._n
