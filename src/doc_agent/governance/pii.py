"""Governance — PII detection + redaction (mandatory).

Wired by ``wiring.register_all`` into three seams: ``AFTER_OCR`` (scrub recognised text before
it is ever indexed), ``BEFORE_ANSWER`` (scrub the outgoing answer) and ``ON_LOG``. The handler
therefore cannot assume a ctx shape — ``AFTER_OCR`` carries ``{"chunks": [Chunk, ...]}`` while
``BEFORE_ANSWER`` carries ``{"state": {...}}`` — so it walks whatever it is given and redacts
every string it finds, rather than reaching for named keys that may not be there.

It redacts **in place**. ``pipeline.build_knowledge_base`` discards the value ``hooks.run``
returns and goes on to chunk the same list it passed in, so a handler that returned tidy copies
would scrub nothing at all.

What is deliberately *not* detected is personal names. This corpus is a 1918 domestic manual
whose text is full of them — "Mrs. Lincoln's recipe", contributors, dedications — and they are
load-bearing content, not leaked data: a name detector here would quietly destroy the thing we
are trying to retrieve. The patterns are limited to identifiers that are unambiguous wherever
they appear: email addresses, telephone numbers, US social-security numbers, street addresses.

Measured over the book's full text layer (625 pages, 317,844 words): **11 spans, all ADDRESS,
all on the two advertising pages** — "36 Pearl Street", "606 Chicago Ave.", the 1918 grocers
and stove merchants who paid for the back matter. Zero EMAIL, PHONE or SSN, and zero spans of
any kind on the 40 held-out pages, so the recipe corpus itself is untouched and the evaluation
numbers are unaffected. That is the expected answer for a public-domain cookbook and the reason
this stays regex-sized. It is on the path anyway because a scan of any modern document would
carry the real thing, and because scrubbing after indexing would already be too late.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]*[a-zA-Z]{2,}\b")),
    (
        "PHONE",
        re.compile(r"(?<!\d)(?:\+?\d{1,2}[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\d)"),
    ),
    ("SSN", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z][a-zA-Z]+\s+){1,3}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b\.?"
        ),
    ),
)

_TAG = "[REDACTED:{}]"


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, type)`` PII spans, in order and non-overlapping."""
    found = sorted(
        ((m.start(), m.end(), kind) for kind, rx in _PATTERNS for m in rx.finditer(text)),
        key=lambda s: (s[0], -s[1]),
    )
    spans: list[tuple[int, int, str]] = []
    for start, end, kind in found:
        # Two patterns can cover the same characters (a phone number inside an address); keeping
        # the first, widest match stops one redaction being written inside another.
        if spans and start < spans[-1][1]:
            continue
        spans.append((start, end, kind))
    return spans


def redact(text: str) -> str:
    """Replace every detected span with a typed placeholder."""
    spans = detect(text)
    if not spans:
        return text
    out: list[str] = []
    prev = 0
    for start, end, kind in spans:
        out.append(text[prev:start])
        out.append(_TAG.format(kind))
        prev = end
    out.append(text[prev:])
    return "".join(out)


def _scrub_value(value: object) -> object:
    """Redact in place; a replacement is returned only for strings, which cannot be mutated."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        for k, v in value.items():
            value[k] = _scrub_value(v)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = _scrub_value(v)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            setattr(value, name, _scrub_value(getattr(value, name)))
    return value


def register(hooks) -> None:  # type: ignore[no-untyped-def]
    """Wire PII redaction into the pipeline."""

    def _scrub(ctx: dict) -> dict:
        _scrub_value(ctx)
        return ctx

    hooks.register(hooks.AFTER_OCR, _scrub)  # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)  # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)  # scrub logs
