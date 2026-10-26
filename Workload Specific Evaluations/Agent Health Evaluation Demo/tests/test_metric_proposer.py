"""Property-based test for metric_proposer.py's candidate validation.

**Property 5: Candidate metric validation accepts exactly the
fully-populated proposals**

Validates: Requirements 2.2

For any dict, ``validate_candidate`` returns true if and only if ``name``,
``definition``, ``rubric``, and ``traceability`` are all present and
non-empty (a missing key, a ``None`` value, and a whitespace-only string
all count as "not present" -- see ``metric_proposer._is_present``).

This test makes no AWS/Bedrock calls and performs no file I/O; it only
exercises ``validate_candidate`` as a pure function of a generated dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from metric_proposer import REQUIRED_FIELDS, validate_candidate  # noqa: E402


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

# A non-empty, non-whitespace-only string -- the "present" case (a).
_present_value = st.text(min_size=1).filter(lambda s: s.strip() != "")

# A whitespace-only string (including plain spaces, tabs, newlines, and the
# empty string) -- one of the "not present" cases (d).
_whitespace_only_value = st.one_of(
    st.just(""),
    st.text(alphabet=" \t\n\r", min_size=1, max_size=5),
)

# For a single required field, one of four independent cases:
#   (a) present: a random non-empty, non-whitespace string -> counts as present
#   (b) absent: the key is simply not included in the dict
#   (c) None: the key is included with value None
#   (d) whitespace-only: the key is included with a whitespace-only string
# Represented as a (is_present, value_or_sentinel) pair so the test can
# compute the expected overall result independently of validate_candidate.
_ABSENT = object()

_field_case = st.one_of(
    st.tuples(st.just(True), _present_value),
    st.tuples(st.just(False), st.just(_ABSENT)),
    st.tuples(st.just(False), st.just(None)),
    st.tuples(st.just(False), _whitespace_only_value),
)

# One case per required field, generated independently.
_candidate_cases = st.fixed_dictionaries({field: _field_case for field in REQUIRED_FIELDS})


def _build_raw_and_expected(cases: dict) -> tuple[dict, bool]:
    """Turn a {field: (is_present, value_or_ABSENT)} mapping into a raw dict
    (omitting ABSENT-marked fields) and the expected validate_candidate
    result (True iff every field's case was "present")."""
    raw: dict = {}
    all_present = True
    for field in REQUIRED_FIELDS:
        is_present, value = cases[field]
        if value is not _ABSENT:
            raw[field] = value
        all_present = all_present and is_present
    return raw, all_present


@settings(max_examples=100)
@given(cases=_candidate_cases)
def test_validate_candidate_accepts_exactly_fully_populated_proposals(cases):
    raw, expected = _build_raw_and_expected(cases)
    assert validate_candidate(raw) == expected


# ---------------------------------------------------------------------------
# Property-based test for metric_proposer.py's candidate filtering.
#
# **Property 6: Filtering partitions raw proposals into accepted and
# omitted with no loss or duplication**
#
# Validates: Requirements 2.3
#
# For any list of raw candidate-metric dicts, ``filter_candidates`` returns
# an accepted list containing exactly the dicts for which
# ``validate_candidate`` is true (converted to ``CandidateHealthMetric``)
# and a rejected list containing exactly the rest (as ``RejectedMetric``
# with their missing fields recorded); every input dict appears in exactly
# one of the two output lists, and relative order is preserved within each
# output list.
#
# This test makes no AWS/Bedrock calls and performs no file I/O; it only
# exercises ``filter_candidates``/``validate_candidate`` as pure functions
# of a generated list of dicts.
# ---------------------------------------------------------------------------

from metric_proposer import (  # noqa: E402
    filter_candidates,
    missing_required_fields,
    validate_candidate,
)
from models import CandidateHealthMetric, RejectedMetric  # noqa: E402

# A single raw candidate dict, randomly either fully-populated (always
# accepted) or partially-populated/empty (built from the same per-field
# present/absent/None/whitespace-only cases used for validate_candidate's
# own test, so it may happen to be fully-populated too -- that's fine, the
# property holds either way since we always classify via validate_candidate
# directly rather than assuming which bucket a given case lands in).
_raw_candidate = st.fixed_dictionaries({field: _field_case for field in REQUIRED_FIELDS}).map(
    lambda cases: _build_raw_and_expected(cases)[0]
)

# 0 to ~15 raw candidate dicts, in order.
_raw_candidates_list = st.lists(_raw_candidate, min_size=0, max_size=15)

_agent_name = st.text(min_size=1, max_size=20).filter(lambda s: s.strip() != "")


@settings(max_examples=100)
@given(raw_candidates=_raw_candidates_list, agent_name=_agent_name)
def test_filter_candidates_partitions_with_no_loss_or_duplication(raw_candidates, agent_name):
    accepted, rejected = filter_candidates(raw_candidates, agent_name)

    # No loss, no duplication: every input dict appears in exactly one of
    # the two output lists, and the combined count matches the input count.
    assert len(accepted) + len(rejected) == len(raw_candidates)

    expected_accepted_raw = [raw for raw in raw_candidates if validate_candidate(raw)]
    expected_rejected_raw = [raw for raw in raw_candidates if not validate_candidate(raw)]

    # accepted contains exactly the dicts for which validate_candidate is
    # True, converted to CandidateHealthMetric with the given agent_name,
    # in the same relative order as raw_candidates.
    assert len(accepted) == len(expected_accepted_raw)
    for metric, raw in zip(accepted, expected_accepted_raw):
        assert isinstance(metric, CandidateHealthMetric)
        assert metric == CandidateHealthMetric(
            name=raw["name"],
            definition=raw["definition"],
            rubric=raw["rubric"],
            traceability=raw["traceability"],
            agent_name=agent_name,
        )

    # rejected contains exactly the rest, as RejectedMetric with raw == d
    # and missing_fields matching missing_required_fields(d), in the same
    # relative order as raw_candidates.
    assert len(rejected) == len(expected_rejected_raw)
    for rejection, raw in zip(rejected, expected_rejected_raw):
        assert isinstance(rejection, RejectedMetric)
        assert rejection.raw == raw
        assert rejection.missing_fields == missing_required_fields(raw)
