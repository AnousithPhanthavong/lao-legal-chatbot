"""
Lao Legal RAG — Robust Call Wrapper
=====================================
Wraps any llm/embed callable so EVERY call automatically retries on transient
server errors (503 overload, 429 rate limit, 500, 502, 504) with exponential
backoff. Rotates API keys across attempts when keys are provided.

The point: a 503 "model is busy" error is TEMPORARY and RANDOM — it should be
retried, not turned into a fake "couldn't synthesize" message. After all retries
are exhausted, it raises a CLEAR error naming the real cause, so the UI can show
"AI service busy, try again" instead of a misleading default.

Usage:
    from robust_call import make_robust_llm, RobustCallError

    # wrap your existing llm function
    robust_llm = make_robust_llm(my_llm_fn, max_retries=4)
    answer = robust_llm("my prompt")   # retries 503/429 automatically

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

import random
import time
from typing import Callable


# transient errors worth retrying (server-side, temporary)
_TRANSIENT_MARKERS = ("503", "429", "500", "502", "504",
                      "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overload",
                      "high demand", "try again", "rate limit",
                      "RESOURCE EXHAUSTED", "quota")


class RobustCallError(Exception):
    """Raised after all retries are exhausted. Names the real cause."""
    def __init__(self, message: str, last_error: Exception, attempts: int):
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts


def is_transient(error: Exception) -> bool:
    """True if the error is a temporary server problem worth retrying."""
    s = str(error)
    return any(m in s for m in _TRANSIENT_MARKERS)


def classify_error(error: Exception) -> str:
    """Human-readable cause for the UI."""
    s = str(error)
    if "503" in s or "UNAVAILABLE" in s or "high demand" in s or "overload" in s:
        return "overload"          # server busy
    if "429" in s or "RESOURCE_EXHAUSTED" in s or "rate limit" in s or "quota" in s:
        return "rate_limit"        # too many requests / quota
    if "500" in s or "502" in s or "504" in s:
        return "server_error"
    return "other"


def robust_call(
    fn: Callable,
    *args,
    max_retries: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    on_retry: Callable[[int, str, float], None] = None,
    **kwargs,
):
    """Call fn(*args, **kwargs), retrying transient errors with backoff.

    Raises RobustCallError after max_retries, naming the real cause.
    `on_retry(attempt, cause, delay)` is called before each wait (for logging).
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if not is_transient(e) or attempt == max_retries:
                # not retryable, or out of retries
                if is_transient(e):
                    cause = classify_error(e)
                    raise RobustCallError(
                        f"AI service unavailable after {attempt+1} attempts "
                        f"(cause: {cause}). The model is temporarily busy — "
                        f"please try again in a moment.",
                        last_error=e, attempts=attempt + 1,
                    ) from e
                raise  # genuine non-transient error — surface it as-is
            # transient: back off and retry
            cause = classify_error(e)
            delay = min(base_delay * (2 ** attempt), max_delay)
            delay += random.uniform(0, delay * 0.25)  # jitter
            if on_retry:
                on_retry(attempt + 1, cause, delay)
            time.sleep(delay)
    raise RobustCallError(  # pragma: no cover
        "AI service unavailable.", last_error=last_error, attempts=max_retries + 1)


def make_robust_llm(
    llm_fn: Callable[[str], str],
    *,
    max_retries: int = 4,
    base_delay: float = 2.0,
    verbose: bool = True,
) -> Callable[[str], str]:
    """Wrap an llm(prompt)->str function so it retries transient errors."""
    def _logger(attempt, cause, delay):
        if verbose:
            print(f"    [retry {attempt}] {cause} — waiting {delay:.1f}s...")

    def _wrapped(prompt: str) -> str:
        return robust_call(llm_fn, prompt,
                           max_retries=max_retries, base_delay=base_delay,
                           on_retry=_logger)
    return _wrapped


def make_robust_embed(
    embed_fn: Callable[[str], list],
    *,
    max_retries: int = 4,
    base_delay: float = 2.0,
    verbose: bool = True,
) -> Callable[[str], list]:
    """Wrap an embed(text)->vector function so it retries transient errors."""
    def _logger(attempt, cause, delay):
        if verbose:
            print(f"    [embed retry {attempt}] {cause} — waiting {delay:.1f}s...")

    def _wrapped(text: str) -> list:
        return robust_call(embed_fn, text,
                           max_retries=max_retries, base_delay=base_delay,
                           on_retry=_logger)
    return _wrapped
