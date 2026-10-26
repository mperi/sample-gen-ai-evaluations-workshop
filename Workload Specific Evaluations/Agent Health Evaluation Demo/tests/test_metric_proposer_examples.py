"""Example/unit tests for metric_proposer.py (Task 3.5).

Validates: Requirements 2.3, 2.5

Concrete, non-randomized pins covering:
  (1) one fully-populated raw proposal: validate_candidate accepts it, and
      filter_candidates produces exactly one CandidateHealthMetric with the
      correct agent_name stamped on.
  (2) one proposal missing a required field (rubric): validate_candidate
      rejects it, and filter_candidates reports missing_fields == ["rubric"].
  (3) a load_metrics round trip against a hand-edited file: write a few
      CandidateHealthMetrics to a distinct test-only scratch JSONL path
      under output/metrics/ via persistence.append_jsonl, simulate a human
      editing one field by rewriting the file with one row's rubric text
      changed, then confirm load_metrics(path) reflects the edit.

This is a distinct file from tests/test_metric_proposer.py (which holds
tasks 3.2/3.3's property tests for validate_candidate/filter_candidates) to
avoid any collision risk between the two test modules.

No AWS/Bedrock calls are made anywhere in this file -- only
validate_candidate/filter_candidates (pure functions) and
persistence.append_jsonl/metric_proposer.load_metrics (local file I/O
against a test-only scratch path) are exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from metric_proposer import filter_candidates, load_metrics, validate_candidate  # noqa: E402
from models import CandidateHealthMetric  # noqa: E402
from persistence import OUTPUT_DIR, append_jsonl  # noqa: E402

# Distinct test-only scratch path under output/metrics/, isolated from the
# real output/metrics/candidate_metrics.jsonl and from any other test
# module's scratch path.
_SCRATCH_CANDIDATE_METRICS_RELATIVE_PATH = "metrics/_test_metric_proposer_examples_candidate_metrics.jsonl"
_SCRATCH_CANDIDATE_METRICS_PATH = OUTPUT_DIR / _SCRATCH_CANDIDATE_METRICS_RELATIVE_PATH


@pytest.fixture(autouse=True)
def _cleanup_scratch_candidate_metrics_file():
    """Remove this module's own scratch candidate-metrics file before and
    after each test so repeated runs never pollute (or are polluted by) the
    real output/metrics/candidate_metrics.jsonl or any other test's scratch
    file."""
    _SCRATCH_CANDIDATE_METRICS_PATH.unlink(missing_ok=True)
    yield
    _SCRATCH_CANDIDATE_METRICS_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# (1) One fully-populated raw proposal
# ---------------------------------------------------------------------------

def test_fully_populated_proposal_is_accepted_with_agent_name_stamped_on():
    """A single fully-populated raw proposal is accepted by
    validate_candidate, and filter_candidates produces exactly one
    CandidateHealthMetric with the correct agent_name stamped on.

    **Validates: Requirements 2.3**
    """
    fully_populated = {
        "name": "groundedness",
        "definition": "Whether claims are grounded in source facts",
        "rubric": "0-1 scale, 1 = fully grounded",
        "traceability": "role statement: analyze market trends",
    }

    assert validate_candidate(fully_populated) is True

    accepted, rejected = filter_candidates([fully_populated], agent_name="market_trends")

    assert rejected == []
    assert len(accepted) == 1

    metric = accepted[0]
    assert isinstance(metric, CandidateHealthMetric)
    assert metric == CandidateHealthMetric(
        name="groundedness",
        definition="Whether claims are grounded in source facts",
        rubric="0-1 scale, 1 = fully grounded",
        traceability="role statement: analyze market trends",
        agent_name="market_trends",
    )
    assert metric.agent_name == "market_trends"


# ---------------------------------------------------------------------------
# (2) One proposal missing a required field (rubric)
# ---------------------------------------------------------------------------

def test_proposal_missing_rubric_is_rejected_with_missing_fields_recorded():
    """A proposal missing the required `rubric` field is rejected by
    validate_candidate, and filter_candidates records
    missing_fields == ["rubric"] on the resulting RejectedMetric.

    **Validates: Requirements 2.3**
    """
    missing_rubric = {
        "name": "latency",
        "definition": "How quickly the agent responds",
        "traceability": "responsibility: respond promptly",
    }

    assert validate_candidate(missing_rubric) is False

    accepted, rejected = filter_candidates([missing_rubric], agent_name="market_trends")

    assert accepted == []
    assert len(rejected) == 1

    rejection = rejected[0]
    assert rejection.raw == missing_rubric
    assert rejection.missing_fields == ["rubric"]


# ---------------------------------------------------------------------------
# (3) load_metrics round trip against a hand-edited file
# ---------------------------------------------------------------------------

def test_load_metrics_reflects_a_hand_edited_field():
    """Writing a few CandidateHealthMetrics to a scratch JSONL file, then
    simulating a human hand-editing one entry's rubric text directly in the
    file, and finally calling load_metrics(path) on it, reflects the edited
    value -- demonstrating the human-edited-override path described in
    Requirement 2.5.

    **Validates: Requirements 2.5**
    """
    first = CandidateHealthMetric(
        name="groundedness",
        definition="Whether claims are grounded in source facts",
        rubric="0-1 scale, 1 = fully grounded",
        traceability="role statement: analyze market trends",
        agent_name="market_trends",
    )
    second = CandidateHealthMetric(
        name="timeliness",
        definition="Whether the analysis reflects current data",
        rubric="0-1 scale, 1 = fully current",
        traceability="responsibility: track emerging trends",
        agent_name="market_trends",
    )
    third = CandidateHealthMetric(
        name="citation_quality",
        definition="Whether sources are properly cited",
        rubric="0-1 scale, 1 = every claim cited",
        traceability="constraint: must cite sources",
        agent_name="market_trends",
    )

    for metric in (first, second, third):
        append_jsonl(_SCRATCH_CANDIDATE_METRICS_RELATIVE_PATH, metric.to_dict())

    # Sanity check: before any hand-editing, the file round-trips exactly
    # as written.
    loaded_before_edit = load_metrics(str(_SCRATCH_CANDIDATE_METRICS_PATH))
    assert loaded_before_edit == [first, second, third]

    # Simulate a human directly editing the second entry's rubric text in
    # the persisted file (e.g. tightening the scoring scale's wording),
    # by rewriting the file with that one field changed and every other
    # line byte-for-byte unchanged.
    edited_second = CandidateHealthMetric(
        name=second.name,
        definition=second.definition,
        rubric="0-5 scale, 5 = fully current and sourced within the last quarter",
        traceability=second.traceability,
        agent_name=second.agent_name,
    )

    import json

    lines = [
        json.dumps(first.to_dict()),
        json.dumps(edited_second.to_dict()),
        json.dumps(third.to_dict()),
    ]
    _SCRATCH_CANDIDATE_METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loaded_after_edit = load_metrics(str(_SCRATCH_CANDIDATE_METRICS_PATH))

    assert len(loaded_after_edit) == 3
    assert loaded_after_edit[0] == first
    assert loaded_after_edit[2] == third

    edited_loaded = loaded_after_edit[1]
    assert edited_loaded.name == second.name
    assert edited_loaded.rubric == "0-5 scale, 5 = fully current and sourced within the last quarter"
    assert edited_loaded.rubric != second.rubric
    assert edited_loaded == edited_second
