"""FIXED — structured logging (auditable NFR). Use get_logger(), never print()."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Path where agent trace steps are written (one TraceStep JSON-Lines per step).
# The A3 agentic-feature gate reads this file to verify that evidence-gated
# re-search actually fires when top_score < weak_threshold.
_TRACE_PATH = Path("traces/run.jsonl")

# ── logger factory ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a structured-JSON logger.  Always use this; never use print()."""
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter(
                '{"ts":"%(asctime)s","lvl":"%(levelname)s","mod":"%(name)s","msg":"%(message)s"}'
            )
        )
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


_log = get_logger(__name__)

# ── hook registration ──────────────────────────────────────────────────────────

def register(hooks) -> None:  # type: ignore[annotation-unchecked]
    """Wire structured tracing at each seam (auditable trail).

    Attaches a handler to ``ON_STEP``, ``ON_TOOL_CALL``, and ``AFTER_ANSWER``.
    Each handler call appends one ``contracts.TraceStep``-shaped JSON-Lines
    record to ``traces/run.jsonl`` so the A3 agentic-feature check can read the
    full trajectory and verify that the re-search branch fires (path must
    depend on observations).

    The ``ctx`` dict that flows through the seam is expected to carry the
    fields described in ``contracts.TraceStep``::

        {
            "step":  <int>,
            "tool":  <str>,   # e.g. "retrieve", "decide", "answer"
            "args":  <dict>,  # e.g. {"query": "...", "k": 10}
            "obs":   <dict>,  # e.g. {"top_score": 0.31, "n": 10}
        }

    Unknown or partial dicts are written as-is so no step is ever silently
    dropped from the audit trail.
    """
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _trace(ctx: dict) -> dict:
        """Append ctx to traces/run.jsonl and emit a structured log line."""
        # Build a safe record — fill defaults for any missing TraceStep fields.
        record = {
            "step": ctx.get("step", -1),
            "tool": ctx.get("tool", "unknown"),
            "args": ctx.get("args", {}),
            "obs":  ctx.get("obs", {}),
        }
        # Append to the JSONL trace file.
        with _TRACE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        # Also emit a structured log line for the audit stream.
        _log.info(
            "trace_step",
            extra={
                "step": record["step"],
                "tool": record["tool"],
            },
        )
        return ctx  # always return ctx so the seam chain is not broken

    hooks.register(hooks.ON_STEP,      _trace)
    hooks.register(hooks.ON_TOOL_CALL, _trace)
    hooks.register(hooks.AFTER_ANSWER, _trace)

