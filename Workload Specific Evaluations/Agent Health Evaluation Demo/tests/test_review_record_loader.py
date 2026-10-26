"""Example/unit tests for the ``ReviewRecord`` loader in ``persistence.py``.

Validates: Requirements 3.2, 3.3

These are concrete example-based tests (as opposed to the property-based
suites in ``test_persistence.py`` / ``test_models_persistence.py``), each
pinned to one realistic ``review_cases.jsonl``-shaped row:

  - ``parse_pipeline_output`` is tested directly as a pure function (no
    I/O) against a combined ``pipeline_output`` string built in the exact
    ``"\\n\\n".join(f"[{agent_name}]\\n{response_text}" ...)`` format that
    the sibling module's ``capture_for_review.py`` produces.
  - ``load_review_records`` is tested end-to-end against a tiny temporary
    JSONL fixture (one multi-agent row, one single-agent row), confirming
    the resulting ``ReviewRecord``s have the right ``record_id``,
    ``brief_text``, ``expected_behavior``, ``responses``, and
    ``agent_name`` (including the ``MULTI_AGENT_SENTINEL`` case).
  - ``load_review_records()`` is also called once with no arguments
    against the real sibling ``review_cases.jsonl`` file, read-only, as a
    one-shot integration-style example confirming the real end-to-end path
    still works against the current dataset.

Nothing in this module writes to, or otherwise modifies, the sibling
module's directory -- the temporary-fixture tests below construct their
own JSONL file under pytest's ``tmp_path`` and read it directly via
``persistence.read_sibling_file``/``json`` without touching
``SIBLING_MODULE_DIR``, and the integration-style test only calls the
read-only ``load_review_records()`` against the real sibling file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from models import ReviewRecord  # noqa: E402
from persistence import (  # noqa: E402
    MULTI_AGENT_SENTINEL,
    load_review_records,
    parse_pipeline_output,
)


# ---------------------------------------------------------------------------
# parse_pipeline_output: pure-function example tests
# ---------------------------------------------------------------------------

def test_parse_pipeline_output_splits_multi_agent_blocks():
    """A combined pipeline_output with three agent blocks, joined with
    "\\n\\n" exactly as capture_for_review.py builds it, splits into a
    dict with the right keys and trimmed response text for each agent."""
    pipeline_output = "\n\n".join(
        [
            "[market_trends]\nThe market grew 8% year over year.",
            "[customer_insights]\nUrban diners want faster service.",
            "[strategy_synth]\nPrioritize suburban expansion next.",
        ]
    )

    responses = parse_pipeline_output(pipeline_output)

    assert set(responses.keys()) == {
        "market_trends",
        "customer_insights",
        "strategy_synth",
    }
    assert responses["market_trends"] == "The market grew 8% year over year."
    assert responses["customer_insights"] == "Urban diners want faster service."
    assert responses["strategy_synth"] == "Prioritize suburban expansion next."


def test_parse_pipeline_output_single_agent_block():
    """A pipeline_output containing exactly one [agent_name] block yields a
    single-entry dict keyed by that agent's name."""
    pipeline_output = "[market_trends]\nThe market grew 8% year over year."

    responses = parse_pipeline_output(pipeline_output)

    assert responses == {"market_trends": "The market grew 8% year over year."}


def test_parse_pipeline_output_strips_surrounding_blank_lines():
    """Leading/trailing blank lines within a block are stripped from the
    returned response text."""
    pipeline_output = "[market_trends]\n\n  The market grew 8%.  \n\n"

    responses = parse_pipeline_output(pipeline_output)

    assert responses == {"market_trends": "The market grew 8%."}


# ---------------------------------------------------------------------------
# load_review_records: end-to-end example tests against a temp JSONL fixture
# ---------------------------------------------------------------------------

def _write_fixture_review_cases(tmp_path: Path, rows: list[dict]) -> Path:
    """Write ``rows`` as a JSONL file under ``tmp_path`` and return its
    path. This is a standalone fixture file -- it is never written under
    the sibling module's directory."""
    fixture_path = tmp_path / "review_cases.jsonl"
    with open(fixture_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")
    return fixture_path


def _load_records_from_fixture(fixture_path: Path) -> list[ReviewRecord]:
    """Exercise the same row-parsing logic as ``load_review_records``
    (``persistence.parse_review_cases_text``, which it delegates to),
    without going through ``read_sibling_file`` (which only ever reads
    from the real sibling module directory)."""
    from persistence import parse_review_cases_text

    return parse_review_cases_text(fixture_path.read_text(encoding="utf-8"))


def test_multi_agent_row_yields_multi_agent_sentinel(tmp_path):
    """A review_cases.jsonl-shaped row whose pipeline_output has more than
    one agent block produces a ReviewRecord with the right record_id,
    brief_text, expected_behavior, responses dict, and
    agent_name == MULTI_AGENT_SENTINEL."""
    pipeline_output = "\n\n".join(
        [
            "[market_trends]\nThe fast-casual market is growing 8% annually.",
            "[customer_insights]\nHealth-conscious diners want transparency.",
        ]
    )
    row = {
        "case_id": "live-brief-01-fastcasual",
        "expected_behavior": "normal_grounded_analysis",
        "brief_text": "Analyse the fast-casual restaurant market.",
        "pipeline_output": pipeline_output,
        "human_label": "",
        "human_rationale": "",
        "provenance": "live-run",
    }
    fixture_path = _write_fixture_review_cases(tmp_path, [row])

    records = _load_records_from_fixture(fixture_path)

    assert len(records) == 1
    record = records[0]
    assert record.record_id == "live-brief-01-fastcasual"
    assert record.brief_text == "Analyse the fast-casual restaurant market."
    assert record.expected_behavior == "normal_grounded_analysis"
    assert record.responses == {
        "market_trends": "The fast-casual market is growing 8% annually.",
        "customer_insights": "Health-conscious diners want transparency.",
    }
    assert record.agent_name == MULTI_AGENT_SENTINEL


def test_single_agent_row_yields_that_agents_name(tmp_path):
    """A review_cases.jsonl-shaped row whose pipeline_output has exactly
    one [agent_name] block produces a ReviewRecord whose agent_name is
    that agent's name, not MULTI_AGENT_SENTINEL."""
    row = {
        "case_id": "live-brief-02-single-agent",
        "expected_behavior": "normal_grounded_analysis",
        "brief_text": "Analyse the fast-casual restaurant market.",
        "pipeline_output": "[market_trends]\nThe market grew 8% year over year.",
        "human_label": "",
        "human_rationale": "",
        "provenance": "live-run",
    }
    fixture_path = _write_fixture_review_cases(tmp_path, [row])

    records = _load_records_from_fixture(fixture_path)

    assert len(records) == 1
    record = records[0]
    assert record.record_id == "live-brief-02-single-agent"
    assert record.responses == {
        "market_trends": "The market grew 8% year over year.",
    }
    assert record.agent_name == "market_trends"
    assert record.agent_name != MULTI_AGENT_SENTINEL


# ---------------------------------------------------------------------------
# load_review_records: one-shot integration-style example against the real
# sibling file (read-only, no mocking, no fixture)
# ---------------------------------------------------------------------------

def test_load_review_records_against_real_sibling_file():
    """load_review_records() with its default arguments reads the real
    sibling review_cases.jsonl end to end without error, exercising the
    actual read_sibling_file path once (complementing the fixture-based
    unit tests above). This test only reads from the sibling module; it
    never writes to it."""
    records = load_review_records()

    assert len(records) >= 1
    assert all(isinstance(r, ReviewRecord) for r in records)
    assert all(r.record_id for r in records)
    assert all(r.agent_name for r in records)
    # At least one real record should have a non-empty responses dict.
    assert any(len(r.responses) > 0 for r in records)
