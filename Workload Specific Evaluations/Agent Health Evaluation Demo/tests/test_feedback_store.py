"""Property-based test for Feedback_Store's latest-label lookup.

**Property 7: Feedback_Store append-then-lookup returns the most recently
written label for a record**

Validates: Requirements 3.5, 3.7

For any sequence of ``ExpectedStateLabel`` writes, including multiple writes
for the same ``record_id`` interspersed with writes for other
``record_id``s, ``latest_label_for_record`` for a given ``record_id``
returns the ``label_text`` from the last write made for that ``record_id``
in the sequence, and never returns a label belonging to a different
``record_id``.

This test exercises ``latest_label_for_record``'s pure in-memory lookup path
only: the generated ``ExpectedStateLabel`` list is passed directly as the
``entries`` argument, so there is no AWS call and no real file I/O (no
``append_jsonl``/``load_feedback`` involved).
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from feedback_store import latest_label_for_record  # noqa: E402
from models import ExpectedStateLabel  # noqa: E402


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A small fixed pool of record ids, so that a generated sequence naturally
# produces multiple writes for the same record_id interspersed with writes
# for other record_ids.
_RECORD_IDS = ["case-1", "case-2", "case-3", "case-4", "case-5"]

_record_id_strategy = st.sampled_from(_RECORD_IDS)

_label_text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

# A sequence of (record_id, label_text) pairs representing writes in order.
_writes_strategy = st.lists(
    st.tuples(_record_id_strategy, _label_text_strategy),
    min_size=1,
    max_size=30,
)


def _writes_to_entries(writes: list[tuple[str, str]]) -> list[ExpectedStateLabel]:
    """Build an ExpectedStateLabel per write, in order, with strictly
    incrementing timestamps so write order is unambiguous."""
    return [
        ExpectedStateLabel(
            record_id=record_id,
            agent_name="some_agent",
            label_text=label_text,
            timestamp=f"2024-01-01T00:{index:02d}:00Z",
        )
        for index, (record_id, label_text) in enumerate(writes)
    ]


# ---------------------------------------------------------------------------
# Property: latest_label_for_record returns the last write for each
# record_id, and never a label belonging to a different record_id
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(writes=_writes_strategy)
def test_latest_label_for_record_returns_most_recent_write(writes):
    entries = _writes_to_entries(writes)

    # The last label_text written for each distinct record_id present in
    # the sequence, computed independently of the function under test.
    expected_latest: dict[str, str] = {}
    for record_id, label_text in writes:
        expected_latest[record_id] = label_text

    for record_id, expected_label_text in expected_latest.items():
        result = latest_label_for_record(record_id, entries)

        assert result is not None
        assert result.record_id == record_id
        assert result.label_text == expected_label_text


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(writes=_writes_strategy)
def test_latest_label_for_record_never_returns_a_different_records_label(writes):
    entries = _writes_to_entries(writes)
    written_record_ids = {record_id for record_id, _ in writes}

    for record_id in written_record_ids:
        result = latest_label_for_record(record_id, entries)
        assert result is not None
        assert result.record_id == record_id


def test_latest_label_for_record_returns_none_for_unknown_record_id():
    """Pin for an id with no writes at all in the pool: returns None rather
    than a label belonging to some other record_id."""
    entries = _writes_to_entries([("case-1", "first"), ("case-2", "second")])

    result = latest_label_for_record("case-5", entries)

    assert result is None


# ---------------------------------------------------------------------------
# Property 8: Feedback_Store entries always retain record id, agent name,
# and timestamp regardless of entry type
#
# Validates: Requirements 6.2, 6.3
#
# For any valid ExpectedStateLabel OR TaskSuccessFeedback, appending it to
# the Feedback_Store and reading it back via load_feedback must yield an
# entry whose record_id, agent_name, and timestamp fields are present
# (non-None) and non-empty, regardless of which of the two entry types it
# is.
#
# All writes for this property go through a distinct test-only path under
# output/feedback/ (never the real FEEDBACK_STORE_PATH), cleaned up by a
# fixture before and after each test (and before each individual append,
# since Hypothesis's @given invokes the test body many times per test-
# function call), so repeated runs never pollute (or are polluted by) the
# real output/feedback/feedback_store.jsonl or any other test's scratch
# file in this module.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from persistence import OUTPUT_DIR, append_jsonl  # noqa: E402
from feedback_store import load_feedback  # noqa: E402
from models import TaskSuccessFeedback  # noqa: E402

_REQUIRED_FIELDS_STORE_RELATIVE_PATH = "feedback/_test_feedback_store_required_fields.jsonl"


@pytest.fixture(autouse=True)
def _cleanup_required_fields_store():
    """Remove this property's own scratch Feedback_Store file before and
    after each test so repeated runs never pollute (or are polluted by) the
    real output/feedback/feedback_store.jsonl or any other test's scratch
    file in this module."""
    scratch_path = OUTPUT_DIR / _REQUIRED_FIELDS_STORE_RELATIVE_PATH
    scratch_path.unlink(missing_ok=True)
    yield
    scratch_path.unlink(missing_ok=True)


_agent_name_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

_timestamp_strategy = _agent_name_strategy  # ISO 8601 in practice; this property only needs non-empty

_free_text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=200,
)

_expected_state_label_strategy = st.builds(
    ExpectedStateLabel,
    record_id=_record_id_strategy.map(str),
    agent_name=_agent_name_strategy,
    label_text=_free_text_strategy,
    timestamp=_timestamp_strategy,
)

_task_success_feedback_strategy = st.builds(
    TaskSuccessFeedback,
    record_id=_record_id_strategy.map(str),
    agent_name=_agent_name_strategy,
    passed=st.booleans(),
    rationale=_free_text_strategy,
    timestamp=_timestamp_strategy,
)

_feedback_entry_strategy = st.one_of(
    _expected_state_label_strategy,
    _task_success_feedback_strategy,
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(entry=_feedback_entry_strategy)
def test_feedback_store_entries_retain_required_fields_across_entry_types(entry):
    """For any ExpectedStateLabel or TaskSuccessFeedback, appending it to
    the Feedback_Store and reading it back yields an entry whose
    record_id, agent_name, and timestamp are present and non-empty,
    regardless of entry type.

    **Validates: Requirements 6.2, 6.3**

    Note: ``append_expected_state_label``/``append_task_success_feedback``
    always write to the real ``FEEDBACK_STORE_PATH`` and take no path
    override, so this test appends ``entry.to_dict()`` (which already
    carries the correct ``"entry_type"`` discriminator -- see
    ``models.py``) directly via ``persistence.append_jsonl`` against the
    distinct test-only path instead, mirroring the pattern used by
    ``feedback_store.py``'s own ``__main__`` sanity check.
    """
    scratch_path = OUTPUT_DIR / _REQUIRED_FIELDS_STORE_RELATIVE_PATH
    scratch_path.unlink(missing_ok=True)

    append_jsonl(_REQUIRED_FIELDS_STORE_RELATIVE_PATH, entry.to_dict())

    loaded = load_feedback(_REQUIRED_FIELDS_STORE_RELATIVE_PATH)

    assert len(loaded) == 1
    loaded_entry = loaded[0]

    assert isinstance(loaded_entry, (ExpectedStateLabel, TaskSuccessFeedback))

    assert loaded_entry.record_id is not None
    assert loaded_entry.record_id != ""

    assert loaded_entry.agent_name is not None
    assert loaded_entry.agent_name != ""

    assert loaded_entry.timestamp is not None
    assert loaded_entry.timestamp != ""


# ---------------------------------------------------------------------------
# Property 9: Task_Success_Feedback round-trips with its record reference
# intact
#
# Validates: Requirements 6.2
#
# For any valid TaskSuccessFeedback record, appending it to the
# Feedback_Store and reading it back yields an entry whose record_id,
# passed, and rationale fields equal the original exactly.
#
# Unlike the latest-label-lookup property above (which exercises the pure
# in-memory lookup path only), this property exercises the real
# append/read file round-trip: it appends via persistence.append_jsonl
# directly (append_task_success_feedback always writes to the real
# FEEDBACK_STORE_PATH and takes no path override) against a distinct
# test-only path under output/feedback/, then reads it back via
# load_feedback. The scratch file is cleaned up by an autouse fixture
# before and after each test so repeated/concurrent runs never pollute (or
# are polluted by) the real output/feedback/feedback_store.jsonl or any
# other test file's own scratch path.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from models import TaskSuccessFeedback  # noqa: E402
from persistence import OUTPUT_DIR, append_jsonl  # noqa: E402
from feedback_store import load_feedback as _load_feedback_for_roundtrip  # noqa: E402

_TASK_SUCCESS_ROUNDTRIP_STORE_RELATIVE_PATH = "feedback/_test_feedback_store_task_success_roundtrip.jsonl"


@pytest.fixture(autouse=True)
def _cleanup_task_success_roundtrip_store():
    """Remove this test's own scratch Feedback_Store file before and after
    each test so repeated runs never pollute (or are polluted by) the real
    output/feedback/feedback_store.jsonl or any other test's scratch file."""
    scratch_path = OUTPUT_DIR / _TASK_SUCCESS_ROUNDTRIP_STORE_RELATIVE_PATH
    scratch_path.unlink(missing_ok=True)
    yield
    scratch_path.unlink(missing_ok=True)


_task_success_text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

_task_success_feedback_roundtrip_strategy = st.builds(
    TaskSuccessFeedback,
    record_id=_task_success_text_strategy,
    agent_name=_task_success_text_strategy,
    passed=st.booleans(),
    rationale=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200),
    timestamp=_task_success_text_strategy,
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(feedback=_task_success_feedback_roundtrip_strategy)
def test_task_success_feedback_roundtrips_with_record_reference_intact(feedback):
    """For any valid TaskSuccessFeedback, appending it to the
    Feedback_Store and reading it back yields an entry whose record_id,
    passed, and rationale fields equal the original exactly.

    **Validates: Requirements 6.2**

    Note: the autouse fixture only runs once per test-function call, but
    Hypothesis's ``@given`` invokes this body many times per call, so the
    scratch file is also removed at the start of the body itself -- mirroring
    the Property 8 test above -- to keep each generated example's append
    isolated to exactly one entry.
    """
    scratch_path = OUTPUT_DIR / _TASK_SUCCESS_ROUNDTRIP_STORE_RELATIVE_PATH
    scratch_path.unlink(missing_ok=True)

    append_jsonl(_TASK_SUCCESS_ROUNDTRIP_STORE_RELATIVE_PATH, feedback.to_dict())

    loaded = _load_feedback_for_roundtrip(_TASK_SUCCESS_ROUNDTRIP_STORE_RELATIVE_PATH)

    assert len(loaded) == 1
    loaded_entry = loaded[0]

    assert isinstance(loaded_entry, TaskSuccessFeedback)
    assert loaded_entry.record_id == feedback.record_id
    assert loaded_entry.passed == feedback.passed
    assert loaded_entry.rationale == feedback.rationale


# ---------------------------------------------------------------------------
# Property 18: Disagreement finder returns exactly the record-matched,
# outcome-conflicting pairs
#
# Validates: Requirements 6.4
#
# For any generated list of TaskSuccessFeedback entries and any generated
# list of MetricResults (both with record_ids drawn from a small shared
# pool so matches and non-matches both occur naturally),
# find_disagreements(entries, results) returns exactly the pairs where:
# both share a record_id, AND (feedback.passed is False and
# result.score >= threshold) OR (feedback.passed is True and
# result.score < threshold).
#
# The expected set of disagreeing pairs is computed independently here (a
# Cartesian-product + filter over the same inputs), mirroring but not
# calling find_disagreements itself, and compared against the actual
# output as sets of hashable pair-identifiers (since dict/list ordering of
# pairs returned by the implementation need not match the order the
# expected set is built in).
# ---------------------------------------------------------------------------

from feedback_store import find_disagreements, DEFAULT_DISAGREEMENT_THRESHOLD  # noqa: E402
from models import MetricResult  # noqa: E402

# Small shared pool of record ids so generated feedback/result lists
# naturally produce both matching and non-matching record_ids.
_DISAGREEMENT_RECORD_IDS = ["rec-1", "rec-2", "rec-3", "rec-4"]

_disagreement_record_id_strategy = st.sampled_from(_DISAGREEMENT_RECORD_IDS)

_disagreement_text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=30,
)

_disagreement_feedback_strategy = st.builds(
    TaskSuccessFeedback,
    record_id=_disagreement_record_id_strategy,
    agent_name=_disagreement_text_strategy,
    passed=st.booleans(),
    rationale=_disagreement_text_strategy,
    timestamp=_disagreement_text_strategy,
)

_disagreement_metric_result_strategy = st.builds(
    MetricResult,
    record_id=_disagreement_record_id_strategy,
    metric_name=_disagreement_text_strategy,
    agent_name=_disagreement_text_strategy,
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    rationale=_disagreement_text_strategy,
)

_disagreement_entries_strategy = st.lists(_disagreement_feedback_strategy, max_size=8)
_disagreement_results_strategy = st.lists(_disagreement_metric_result_strategy, max_size=8)


def _feedback_identity(feedback: TaskSuccessFeedback) -> tuple:
    """A hashable identifier for a TaskSuccessFeedback, used to compare
    pairs as sets without relying on dataclass identity/equality nuances."""
    return (
        feedback.record_id,
        feedback.agent_name,
        feedback.passed,
        feedback.rationale,
        feedback.timestamp,
    )


def _result_identity(result: MetricResult) -> tuple:
    """A hashable identifier for a MetricResult, used to compare pairs as
    sets without relying on dataclass identity/equality nuances."""
    return (
        result.record_id,
        result.metric_name,
        result.agent_name,
        result.score,
        result.rationale,
    )


def _expected_disagreements(
    entries: list, results: list, threshold: float
) -> set:
    """Independently compute the expected disagreeing pairs via a
    Cartesian-product + filter over the same inputs, mirroring but not
    calling find_disagreements."""
    expected = set()
    for feedback in entries:
        for result in results:
            if feedback.record_id != result.record_id:
                continue
            failed_but_scored_pass = (not feedback.passed) and result.score >= threshold
            passed_but_scored_fail = feedback.passed and result.score < threshold
            if failed_but_scored_pass or passed_but_scored_fail:
                expected.add((_feedback_identity(feedback), _result_identity(result)))
    return expected


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    entries=_disagreement_entries_strategy,
    results=_disagreement_results_strategy,
)
def test_find_disagreements_returns_exactly_the_conflicting_matched_pairs(entries, results):
    """find_disagreements returns exactly the pairs that share a record_id
    and disagree about the pass/fail outcome, using the default threshold.

    **Validates: Requirements 6.4**
    """
    threshold = DEFAULT_DISAGREEMENT_THRESHOLD

    actual_pairs = find_disagreements(entries, results, threshold=threshold)
    actual = {
        (_feedback_identity(feedback), _result_identity(result))
        for feedback, result in actual_pairs
    }

    expected = _expected_disagreements(entries, results, threshold)

    assert actual == expected


# ---------------------------------------------------------------------------
# Property 19: Feedback_Store contents are unaffected by Evaluator_Run
# persistence operations
#
# Validates: Requirements 6.5
#
# For any sequence of Feedback_Store writes (a generated mix of
# ExpectedStateLabel/TaskSuccessFeedback entries, appended to a distinct
# test-only Feedback_Store path under output/feedback/), followed by one or
# more unrelated Evaluator_Run result-persistence operations (write_json
# and append_jsonl calls against a DIFFERENT distinct test-only path under
# output/results/, writing generated MetricResult-shaped dicts),
# load_feedback on the Feedback_Store's test-only path returns the exact
# same entries (count, content, order) before and after those unrelated
# result-persistence operations ran.
#
# Both the feedback-area scratch path and the results-area scratch path are
# distinct from every other test's scratch path in this module, and are
# cleaned up by an autouse fixture before and after each test (and at the
# start of each generated example's body, mirroring the Property 8/9
# fixtures above, since Hypothesis's @given invokes the test body many
# times per test-function call).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from persistence import OUTPUT_DIR, append_jsonl, write_json  # noqa: E402
from feedback_store import load_feedback as _load_feedback_for_independence  # noqa: E402

_INDEPENDENCE_FEEDBACK_STORE_RELATIVE_PATH = "feedback/_test_feedback_store_independence.jsonl"
_INDEPENDENCE_RESULTS_JSONL_RELATIVE_PATH = "results/_test_feedback_store_independence_metric_results.jsonl"
_INDEPENDENCE_RESULTS_JSON_RELATIVE_PATH = "results/_test_feedback_store_independence_metric_result.json"


@pytest.fixture(autouse=True)
def _cleanup_independence_scratch_paths():
    """Remove this property's own feedback-area and results-area scratch
    files before and after each test so repeated runs never pollute (or are
    polluted by) the real Feedback_Store, or any other test's scratch file
    in this module."""
    feedback_scratch_path = OUTPUT_DIR / _INDEPENDENCE_FEEDBACK_STORE_RELATIVE_PATH
    results_jsonl_scratch_path = OUTPUT_DIR / _INDEPENDENCE_RESULTS_JSONL_RELATIVE_PATH
    results_json_scratch_path = OUTPUT_DIR / _INDEPENDENCE_RESULTS_JSON_RELATIVE_PATH
    for scratch_path in (feedback_scratch_path, results_jsonl_scratch_path, results_json_scratch_path):
        scratch_path.unlink(missing_ok=True)
    yield
    for scratch_path in (feedback_scratch_path, results_jsonl_scratch_path, results_json_scratch_path):
        scratch_path.unlink(missing_ok=True)


_independence_feedback_entry_strategy = st.one_of(
    _expected_state_label_strategy,
    _task_success_feedback_strategy,
)

_independence_feedback_writes_strategy = st.lists(
    _independence_feedback_entry_strategy,
    min_size=1,
    max_size=15,
)

_independence_metric_result_dict_strategy = st.builds(
    lambda record_id, metric_name, agent_name, score, rationale: {
        "record_id": record_id,
        "metric_name": metric_name,
        "agent_name": agent_name,
        "score": score,
        "rationale": rationale,
    },
    record_id=_disagreement_record_id_strategy,
    metric_name=_disagreement_text_strategy,
    agent_name=_disagreement_text_strategy,
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    rationale=_disagreement_text_strategy,
)

_independence_result_writes_strategy = st.lists(
    _independence_metric_result_dict_strategy,
    min_size=1,
    max_size=10,
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    feedback_writes=_independence_feedback_writes_strategy,
    result_writes=_independence_result_writes_strategy,
)
def test_feedback_store_unaffected_by_evaluator_run_persistence(feedback_writes, result_writes):
    """For any sequence of Feedback_Store writes followed by any sequence
    of unrelated Evaluator_Run result-persistence operations, reading the
    Feedback_Store afterward returns the same entries (count, content,
    order) that were present beforehand, unchanged.

    **Validates: Requirements 6.5**
    """
    feedback_scratch_path = OUTPUT_DIR / _INDEPENDENCE_FEEDBACK_STORE_RELATIVE_PATH
    results_jsonl_scratch_path = OUTPUT_DIR / _INDEPENDENCE_RESULTS_JSONL_RELATIVE_PATH
    results_json_scratch_path = OUTPUT_DIR / _INDEPENDENCE_RESULTS_JSON_RELATIVE_PATH
    for scratch_path in (feedback_scratch_path, results_jsonl_scratch_path, results_json_scratch_path):
        scratch_path.unlink(missing_ok=True)

    # Write the Feedback_Store entries to their own distinct test-only path.
    for entry in feedback_writes:
        append_jsonl(_INDEPENDENCE_FEEDBACK_STORE_RELATIVE_PATH, entry.to_dict())

    before = _load_feedback_for_independence(_INDEPENDENCE_FEEDBACK_STORE_RELATIVE_PATH)

    # Unrelated Evaluator_Run result-persistence operations, against a
    # DIFFERENT distinct test-only path under output/results/ -- simulating
    # an Evaluator_Run writing its own MetricResult-shaped data via both
    # append_jsonl (metric_results.jsonl-style) and write_json (single-file
    # style), never touching the Feedback_Store's own path.
    for result_dict in result_writes:
        append_jsonl(_INDEPENDENCE_RESULTS_JSONL_RELATIVE_PATH, result_dict)
    if result_writes:
        write_json(_INDEPENDENCE_RESULTS_JSON_RELATIVE_PATH, result_writes[-1])

    after = _load_feedback_for_independence(_INDEPENDENCE_FEEDBACK_STORE_RELATIVE_PATH)

    assert len(after) == len(before)
    assert after == before
    for before_entry, after_entry in zip(before, after):
        assert type(before_entry) is type(after_entry)
        assert before_entry.to_dict() == after_entry.to_dict()


# ---------------------------------------------------------------------------
# Example/unit tests for feedback_store.py (Task 5.8)
#
# Validates: Requirements 6.3
#
# Concrete, non-randomized pins covering:
#   (1) a mixed-type feedback_store.jsonl-shaped file (both ExpectedStateLabel
#       and TaskSuccessFeedback entries present) loaded via load_feedback,
#       checked for length, per-entry type/order, and concrete field values.
#   (2) latest_label_for_record for a record_id that only has a
#       TaskSuccessFeedback entry (no ExpectedStateLabel at all) in the same
#       mixed file, confirming it returns None rather than conflating the
#       two entry types.
#
# Both tests share one distinct test-only scratch path under
# output/feedback/, isolated from every other test's scratch path in this
# module, cleaned up by an autouse fixture before and after each test.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from persistence import OUTPUT_DIR, append_jsonl  # noqa: E402
from feedback_store import load_feedback as _load_feedback_for_examples  # noqa: E402
from feedback_store import latest_label_for_record as _latest_label_for_examples  # noqa: E402
from models import ExpectedStateLabel as _ExpectedStateLabelForExamples  # noqa: E402
from models import TaskSuccessFeedback as _TaskSuccessFeedbackForExamples  # noqa: E402

_EXAMPLE_MIXED_STORE_RELATIVE_PATH = "feedback/_test_feedback_store_example_mixed_entries.jsonl"


@pytest.fixture(autouse=True)
def _cleanup_example_mixed_store():
    """Remove this example test's own scratch Feedback_Store file before and
    after each test so repeated runs never pollute (or are polluted by) the
    real output/feedback/feedback_store.jsonl or any other test's scratch
    file in this module."""
    scratch_path = OUTPUT_DIR / _EXAMPLE_MIXED_STORE_RELATIVE_PATH
    scratch_path.unlink(missing_ok=True)
    yield
    scratch_path.unlink(missing_ok=True)


def _write_example_mixed_store() -> None:
    """Append 3 ExpectedStateLabels and 2 TaskSuccessFeedback entries, in a
    fixed, known order, to the shared example scratch path."""
    append_jsonl(
        _EXAMPLE_MIXED_STORE_RELATIVE_PATH,
        _ExpectedStateLabelForExamples(
            record_id="example-case-1",
            agent_name="market_trends",
            label_text="Should have flagged the missing source.",
            timestamp="2024-02-01T00:00:00Z",
        ).to_dict(),
    )
    append_jsonl(
        _EXAMPLE_MIXED_STORE_RELATIVE_PATH,
        _ExpectedStateLabelForExamples(
            record_id="example-case-1",
            agent_name="market_trends",
            label_text="Second, most recent label for case-1.",
            timestamp="2024-02-01T00:01:00Z",
        ).to_dict(),
    )
    append_jsonl(
        _EXAMPLE_MIXED_STORE_RELATIVE_PATH,
        _ExpectedStateLabelForExamples(
            record_id="example-case-2",
            agent_name="customer_insights",
            label_text="Should have cited the Q3 report.",
            timestamp="2024-02-01T00:02:00Z",
        ).to_dict(),
    )
    append_jsonl(
        _EXAMPLE_MIXED_STORE_RELATIVE_PATH,
        _TaskSuccessFeedbackForExamples(
            record_id="example-case-3",
            agent_name="customer_insights",
            passed=True,
            rationale="Correctly summarized the customer feedback.",
            timestamp="2024-02-01T00:03:00Z",
        ).to_dict(),
    )
    append_jsonl(
        _EXAMPLE_MIXED_STORE_RELATIVE_PATH,
        _TaskSuccessFeedbackForExamples(
            record_id="example-case-4",
            agent_name="market_trends",
            passed=False,
            rationale="Missed the flawed premise in the brief.",
            timestamp="2024-02-01T00:04:00Z",
        ).to_dict(),
    )


def test_load_feedback_reads_mixed_entry_types_in_order_with_correct_fields():
    """A feedback_store.jsonl-shaped file with both ExpectedStateLabel and
    TaskSuccessFeedback entries present loads back as the correct length,
    correct concrete types in append order, and correct field values.

    **Validates: Requirements 6.3**
    """
    _write_example_mixed_store()

    entries = _load_feedback_for_examples(_EXAMPLE_MIXED_STORE_RELATIVE_PATH)

    assert len(entries) == 5

    assert [type(entry) for entry in entries] == [
        _ExpectedStateLabelForExamples,
        _ExpectedStateLabelForExamples,
        _ExpectedStateLabelForExamples,
        _TaskSuccessFeedbackForExamples,
        _TaskSuccessFeedbackForExamples,
    ]

    # Concrete pin on one ExpectedStateLabel entry's full field values.
    second_label = entries[1]
    assert isinstance(second_label, _ExpectedStateLabelForExamples)
    assert second_label.record_id == "example-case-1"
    assert second_label.agent_name == "market_trends"
    assert second_label.label_text == "Second, most recent label for case-1."
    assert second_label.timestamp == "2024-02-01T00:01:00Z"

    # Concrete pin on one TaskSuccessFeedback entry's full field values.
    first_feedback = entries[3]
    assert isinstance(first_feedback, _TaskSuccessFeedbackForExamples)
    assert first_feedback.record_id == "example-case-3"
    assert first_feedback.agent_name == "customer_insights"
    assert first_feedback.passed is True
    assert first_feedback.rationale == "Correctly summarized the customer feedback."
    assert first_feedback.timestamp == "2024-02-01T00:03:00Z"


def test_latest_label_for_record_returns_none_when_only_task_success_feedback_exists():
    """A record_id with a TaskSuccessFeedback entry but no ExpectedStateLabel
    entry in the same mixed file yields None from latest_label_for_record,
    rather than conflating the two entry types.

    **Validates: Requirements 6.3**
    """
    _write_example_mixed_store()

    entries = _load_feedback_for_examples(_EXAMPLE_MIXED_STORE_RELATIVE_PATH)

    # "example-case-3" has a TaskSuccessFeedback entry above, but no
    # ExpectedStateLabel entry anywhere in the mixed file.
    result = _latest_label_for_examples("example-case-3", entries)

    assert result is None
