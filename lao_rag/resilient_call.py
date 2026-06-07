"""
Lao Legal RAG — Resilient LLM/Embed Call Wrapper
================================================

Hardens the interactive call path against TRANSIENT failures (503 / overload /
timeout), which the existing interactive RotatingClient mis-handled: it only
rotated on 429/quota and RAISED immediately on 503, so a single transient server
error silently collapsed decomposition to a no-op.

The batch pipeline (phase_b_v2) already treats 503/unavailable/overload as
"slow down & retry" — this module brings that same discipline to the interactive
agent path, as ONE tested, reusable place instead of a buried notebook cell.

Design
------
* Pure policy, no SDK import. You inject the list of per-key callables and a
  sleep function, so it is fully unit-testable with fakes (no network, no keys).
* Error classification:
    - OVERLOAD (429/503/quota/unavailable/overload/timeout/deadline/resource)
        -> rotate to next key AND retry with exponential backoff + jitter
    - FATAL (anything else: bad request, auth, code bug)
        -> raise immediately (retrying a real bug just wastes quota)
* Bounded: at most `max_attempts` total tries; raises the last error if exhausted.

This mirrors the dependency-injection style of retrieval.py / decompose.py so the
agent's reliability layer is testable in CI.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

# Substrings (lowercased) that mark a TRANSIENT, retry-worthy error.
OVERLOAD_MARKERS: tuple[str, ...] = (
    "429", "503", "rate", "quota", "resource", "resource_exhausted",
    "unavailable", "overload", "deadline", "timeout", "500", "502", "504",
)


def is_overload_error(err: BaseException) -> bool:
    """True if the error looks transient (slow down / retry), not a real bug."""
    msg = str(err).lower()
    return any(m in msg for m in OVERLOAD_MARKERS)


class AllKeysExhausted(RuntimeError):
    """Raised when every key was tried up to the attempt cap and all failed."""


def call_with_resilience(
    fn_per_key: Sequence[Callable[[], T]],
    *,
    start_idx: int = 0,
    max_attempts: int | None = None,
    initial_backoff: float = 2.0,
    max_backoff: float = 30.0,
    jitter: float = 1.0,
    sleep: Callable[[float], None] = None,  # injected for testability
    on_rotate: Callable[[int], None] = None,  # optional: report new index
) -> T:
    """Call `fn_per_key[idx]()`, rotating + backing off on transient errors.

    Parameters
    ----------
    fn_per_key : one zero-arg callable per key. Calling it performs the real
        request using that key and returns the result (or raises).
    start_idx : which key to try first (carry your RotatingClient's _idx here).
    max_attempts : total tries across all keys. Default = 2 * number_of_keys
        (one full rotation, then a second pass after backoff).
    initial_backoff / max_backoff / jitter : exponential backoff schedule;
        backoff applies only on OVERLOAD errors, doubling each attempt, capped,
        plus uniform jitter in [0, jitter) to desynchronize.
    sleep : injected sleep (real time.sleep in prod, a fake recorder in tests).
    on_rotate : optional callback receiving the new key index after each rotate
        (use it to update your RotatingClient._idx so the next call continues
        from a good key).

    Returns the first successful result.
    Raises the original FATAL error immediately on a non-transient failure, or
    AllKeysExhausted wrapping the last transient error if attempts run out.
    """
    n = len(fn_per_key)
    if n == 0:
        raise ValueError("fn_per_key must contain at least one callable")
    if sleep is None:
        import time
        sleep = time.sleep
    if max_attempts is None:
        max_attempts = n * 2

    idx = start_idx % n
    last_error: BaseException | None = None
    backoff = initial_backoff

    for attempt in range(max_attempts):
        try:
            return fn_per_key[idx]()
        except BaseException as e:  # noqa: BLE001 - we re-raise fatals below
            last_error = e
            if not is_overload_error(e):
                # real bug / auth / bad request: retrying wastes quota
                raise
            # transient: rotate to next key
            idx = (idx + 1) % n
            if on_rotate is not None:
                on_rotate(idx)
            # back off only when we have wrapped around (tried everyone once)
            # or always? -> back off every transient failure, but the wrap adds
            # the bigger pause implicitly via the doubling schedule.
            wait = min(backoff, max_backoff) + random.uniform(0.0, jitter)
            sleep(wait)
            backoff = min(backoff * 2, max_backoff)

    raise AllKeysExhausted(
        f"all {n} keys failed after {max_attempts} attempts; "
        f"last error: {last_error}"
    ) from last_error
