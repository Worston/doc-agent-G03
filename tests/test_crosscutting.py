"""Cross-cutting features must work END TO END, not just exist in one file.
Un-skip and implement alongside the feature. CI runs these."""

import pytest


@pytest.mark.skip(reason="implement with grounding")
def test_grounding_unsupported_query_abstains():
    """An answer with no supporting evidence must abstain, not fabricate."""
    assert True


@pytest.mark.skip(reason="implement with security")
def test_injection_in_document_does_not_hijack():
    """A document containing 'ignore your instructions' must not change agent behaviour."""
    assert True


def test_pii_never_leaks_to_answer_or_log():
    """PII in the corpus must not appear in answers or logs."""
    from doc_agent import hooks
    from doc_agent.contracts import Chunk
    from doc_agent.governance import pii

    hooks.clear()
    pii.register(hooks)

    # AFTER_OCR: pipeline.build_knowledge_base discards what hooks.run returns and chunks the
    # same list it passed in, so the objects themselves must come back scrubbed.
    ocr_chunks = [Chunk(id="c0", doc_id="d", text="mail cook@example.com", page_ids=["p1"])]
    hooks.run(hooks.AFTER_OCR, {"chunks": ocr_chunks})
    assert ocr_chunks[0].text == "mail [REDACTED:EMAIL]"

    # BEFORE_ANSWER / ON_LOG carry a different shape entirely; the same handler must cope.
    state = {"answer": "reach him on 212-555-0143", "evidence": ["ssn 123-45-6789"]}
    hooks.run(hooks.BEFORE_ANSWER, {"state": state})
    assert "555-0143" not in state["answer"]
    assert "123-45-6789" not in state["evidence"][0]

    record = {"msg": "user at 350 Fifth Avenue"}
    hooks.run(hooks.ON_LOG, record)
    assert "Fifth Avenue" not in record["msg"]
    hooks.clear()


@pytest.mark.parametrize(
    "text,kinds",
    [
        ("write jane.doe+x@example.co.uk now", ["EMAIL"]),
        ("call (212) 555-0143 or +1 212.555.0143", ["PHONE", "PHONE"]),
        ("ssn 123-45-6789", ["SSN"]),
        ("at 350 Fifth Avenue, New York", ["ADDRESS"]),
        # The corpus is a domestic manual: personal names, quantities, oven temperatures and
        # page references are content, and a detector that eats them would destroy the corpus.
        ("Mrs. Lincoln of the Boston Cooking School", []),
        ("Take 2 cups flour, bake 350 degrees 25 minutes", []),
        ("1 pint Milk. 3 Eggs. See page 214.", []),
    ],
)
def test_pii_detects_identifiers_but_not_recipe_text(text, kinds):
    from doc_agent.governance import pii

    assert [k for _, _, k in pii.detect(text)] == kinds


def test_pii_spans_never_nest():
    """A phone number inside an address must not be redacted twice."""
    from doc_agent.governance import pii

    spans = pii.detect("write to 12 Bell Ave and call 212-555-0143")
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert "[REDACTED:[REDACTED" not in pii.redact("write to 12 Bell Ave and call 212-555-0143")


def test_pii_leaves_clean_text_byte_identical():
    """The 1918 corpus fires on nothing, so redaction must be the identity on it."""
    from doc_agent.governance import pii

    text = "Wash the silver in hot suds, rinse, and dry with a soft cloth."
    assert pii.redact(text) is text


@pytest.mark.skip(reason="implement with tracing")
def test_trace_covers_every_step():
    """Every agent step and tool call must appear in the audit trail."""
    assert True


@pytest.mark.skip(reason="implement with reproducibility")
def test_rerun_reproduces_metrics():
    """A seeded re-run reproduces reported metrics within tolerance."""
    assert True
