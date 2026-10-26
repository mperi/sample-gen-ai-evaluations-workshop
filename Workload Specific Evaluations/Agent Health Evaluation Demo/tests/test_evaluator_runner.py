"""Property-based test for evaluator_runner.py's exhaustive scoring loop.

**Property 10: Evaluator_Run scores every record-metric combination
exactly once**

Validates: Requirements 4.1, 4.2, 4.3

For any list of ``ReviewRecord``s and ``CandidateHealthMetric``s and any
scorer that never raises, ``run_evaluation`` invokes the scorer exactly
once per ``(record, metric)`` pair and produces exactly
``len(records) * len(metrics)`` ``MetricResult``s, with every result's
``record_id``/``metric_name`` matching the pair it was produced from and no
pair duplicated or omitted.

``run_evaluation`` persists its results via ``persistence.append_jsonl``
against the module-level ``METRIC_RESULTS_PATH``/``METRIC_FAILURES_PATH``
constants, which have no override parameter. This test monkeypatches those
two module globals to distinct test-only paths under ``output/results/``
for the duration of each test (mirroring the pattern used by
``evaluator_runner.py``'s own ``__main__`` sanity check), and cleans those
scratch files up before and after every test (and at the start of each
generated example's body, since Hypothesis's ``@given`` invokes the test
body many times per test-function call) -- so repeated runs never pollute
(or are polluted by) the real
``output/results/metric_results.jsonl``/``.failures.jsonl``.

``run_evaluation`` also calls ``feedback_store.load_feedback()`` internally
with no argument; that resolves ``feedback_store.FEEDBACK_STORE_PATH`` at
call time, which ``tests/conftest.py`` redirects to a per-test scratch store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

import evaluator_runner  # noqa: E402
from evaluator_runner import run_evaluation  # noqa: E402
from models import CandidateHealthMetric, MetricResult, ReviewRecord  # noqa: E402
from persistence import OUTPUT_DIR  # noqa: E402

_SCRATCH_RESULTS_RELATIVE_PATH = "results/_test_evaluator_runner_scoring_results.jsonl"
_SCRATCH_FAILURES_RELATIVE_PATH = "results/_test_evaluator_runner_scoring_results.failures.jsonl"


@pytest.fixture(autouse=True)
def _patch_output_paths_and_cleanup():
    """Monkeypatch evaluator_runner's module-level output path constants to
    distinct test-only paths under output/results/ for the duration of each
    test, and remove those scratch files before and after each test so
    repeated runs never pollute (or are polluted by) the real
    output/results/metric_results.jsonl / .failures.jsonl.
    """
    original_results_path = evaluator_runner.METRIC_RESULTS_PATH
    original_failures_path = evaluator_runner.METRIC_FAILURES_PATH

    results_scratch_path = OUTPUT_DIR / _SCRATCH_RESULTS_RELATIVE_PATH
    failures_scratch_path = OUTPUT_DIR / _SCRATCH_FAILURES_RELATIVE_PATH
    results_scratch_path.unlink(missing_ok=True)
    failures_scratch_path.unlink(missing_ok=True)

    evaluator_runner.METRIC_RESULTS_PATH = _SCRATCH_RESULTS_RELATIVE_PATH
    evaluator_runner.METRIC_FAILURES_PATH = _SCRATCH_FAILURES_RELATIVE_PATH

    try:
        yield
    finally:
        evaluator_runner.METRIC_RESULTS_PATH = original_results_path
        evaluator_runner.METRIC_FAILURES_PATH = original_failures_path
        results_scratch_path.unlink(missing_ok=True)
        failures_scratch_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_name_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() != "")

_free_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=50,
)


def _unique_by(strategy, key, min_size, max_size):
    """A list of values drawn from strategy, deduplicated by key, with a
    lower bound re-enforced after dedup (Hypothesis may otherwise yield a
    shorter-than-min_size list after filtering duplicates)."""
    return st.lists(strategy, min_size=min_size, max_size=max_size).map(
        lambda items: list({key(item): item for item in items}.values())
    ).filter(lambda items: len(items) >= min_size)


_review_record_strategy = st.builds(
    ReviewRecord,
    record_id=_name_text,
    agent_name=_name_text,
    brief_text=_free_text,
    responses=st.just({}),
    expected_behavior=st.none(),
)

_candidate_metric_strategy = st.builds(
    CandidateHealthMetric,
    name=_name_text,
    definition=_free_text,
    rubric=_free_text,
    traceability=_free_text,
    agent_name=_name_text,
)

# 0-6 ReviewRecords with distinct record_ids, 0-6 CandidateHealthMetrics
# with distinct names.
_records_strategy = _unique_by(_review_record_strategy, lambda r: r.record_id, min_size=0, max_size=6)
_metrics_strategy = _unique_by(_candidate_metric_strategy, lambda m: m.name, min_size=0, max_size=6)


# ---------------------------------------------------------------------------
# Always-succeeding scorer, tracking every (record_id, metric_name) pair it
# is invoked with.
# ---------------------------------------------------------------------------

def _make_tracking_scorer():
    """Build a scorer that always succeeds with a fixed (score, rationale)
    and records every (record.record_id, metric.name) pair it is called
    with, in call order, via a closure-captured list."""
    calls: list[tuple[str, str]] = []

    def scorer(record, metric, reference_label):
        calls.append((record.record_id, metric.name))
        return (0.5, "ok")

    return scorer, calls


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(records=_records_strategy, metrics=_metrics_strategy)
def test_run_evaluation_scores_every_pair_exactly_once(records, metrics):
    results_scratch_path = OUTPUT_DIR / _SCRATCH_RESULTS_RELATIVE_PATH
    failures_scratch_path = OUTPUT_DIR / _SCRATCH_FAILURES_RELATIVE_PATH
    results_scratch_path.unlink(missing_ok=True)
    failures_scratch_path.unlink(missing_ok=True)

    scorer, calls = _make_tracking_scorer()

    results, failures = run_evaluation(metrics, records, scorer)

    expected_pairs = {
        (record.record_id, metric.name) for record in records for metric in metrics
    }

    # The scorer never raises, so there must be no failures, and the
    # scorer must have been invoked exactly once per (record, metric) pair
    # -- no pair duplicated or omitted.
    assert failures == []
    assert len(calls) == len(expected_pairs)
    assert set(calls) == expected_pairs

    # run_evaluation produces exactly len(records) * len(metrics)
    # MetricResults.
    assert len(results) == len(records) * len(metrics)

    result_pairs = [(result.record_id, result.metric_name) for result in results]

    # No pair duplicated or omitted among the produced results either.
    assert set(result_pairs) == expected_pairs
    assert len(set(result_pairs)) == len(result_pairs)

    # Every result's record_id/metric_name matches the pair it was
    # produced from (each MetricResult is well-formed for its pair).
    for result in results:
        assert isinstance(result, MetricResult)
        assert (result.record_id, result.metric_name) in expected_pairs
        assert result.score == 0.5
        assert result.rationale == "ok"


# ---------------------------------------------------------------------------
# Property 11: A failing scorer call is isolated to its own combination and
# does not abort the run.
# ---------------------------------------------------------------------------

# Pairs records/metrics with a same-length list of booleans (one per
# (record, metric) pair, in the same order as the iteration order used by
# run_evaluation itself: outer loop over records, inner loop over metrics)
# indicating whether that particular pair should be configured to raise.
# Deriving the booleans list's length from the already-drawn records/metrics
# via .flatmap guarantees it is always exactly len(records) * len(metrics)
# long, so zipping it against the pairs below can never mismatch in length.
_records_metrics_and_failure_flags_strategy = st.tuples(_records_strategy, _metrics_strategy).flatmap(
    lambda records_and_metrics: st.tuples(
        st.just(records_and_metrics[0]),
        st.just(records_and_metrics[1]),
        st.lists(
            st.booleans(),
            min_size=len(records_and_metrics[0]) * len(records_and_metrics[1]),
            max_size=len(records_and_metrics[0]) * len(records_and_metrics[1]),
        ),
    )
)


def _make_selective_failing_scorer(failing_pairs: set[tuple[str, str]]):
    """Build a scorer that raises ``RuntimeError`` for every
    ``(record.record_id, metric.name)`` pair in ``failing_pairs`` and
    otherwise always succeeds with a fixed ``(0.5, "ok")``."""

    def scorer(record, metric, reference_label):
        if (record.record_id, metric.name) in failing_pairs:
            raise RuntimeError(
                f"simulated scorer failure for {record.record_id}/{metric.name}"
            )
        return (0.5, "ok")

    return scorer


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(records_metrics_and_flags=_records_metrics_and_failure_flags_strategy)
def test_run_evaluation_isolates_failing_scorer_calls(records_metrics_and_flags):
    records, metrics, should_fail_flags = records_metrics_and_flags

    results_scratch_path = OUTPUT_DIR / _SCRATCH_RESULTS_RELATIVE_PATH
    failures_scratch_path = OUTPUT_DIR / _SCRATCH_FAILURES_RELATIVE_PATH
    results_scratch_path.unlink(missing_ok=True)
    failures_scratch_path.unlink(missing_ok=True)

    # Pair up (record, metric) combinations in the same outer-records /
    # inner-metrics order run_evaluation itself iterates in, with the
    # drawn per-pair failure flags, to decide an arbitrary subset of pairs
    # to configure as failing.
    all_pairs = [(record.record_id, metric.name) for record in records for metric in metrics]
    failing_pairs = {pair for pair, should_fail in zip(all_pairs, should_fail_flags) if should_fail}
    succeeding_pairs = set(all_pairs) - failing_pairs

    scorer = _make_selective_failing_scorer(failing_pairs)

    results, failures = run_evaluation(metrics, records, scorer)

    result_pairs = [(result.record_id, result.metric_name) for result in results]
    failure_pairs = [(failure.record_id, failure.metric_name) for failure in failures]

    # A MetricResult for every non-raising pair.
    assert set(result_pairs) == succeeding_pairs
    assert len(result_pairs) == len(succeeding_pairs)
    for result in results:
        assert isinstance(result, MetricResult)
        assert (result.record_id, result.metric_name) in succeeding_pairs
        assert result.score == 0.5
        assert result.rationale == "ok"

    # A MetricFailure, identified by record_id/metric_name, for every
    # raising pair.
    assert set(failure_pairs) == failing_pairs
    assert len(failure_pairs) == len(failing_pairs)

    # The failing scorer call is isolated to its own combination: it does
    # not abort the run, so every pair (failing or not) is accounted for
    # exactly once across the two lists combined.
    assert len(results) + len(failures) == len(records) * len(metrics)


# ---------------------------------------------------------------------------
# Property 12: Metric results persist and reload without loss, including
# failure entries.
#
# Validates: Requirements 4.5
#
# APPROACH: this reuses run_evaluation itself (the task 9.2/9.3 pattern --
# an always-succeeding tracking scorer, and a selective-failing scorer) to
# produce a real (results, failures) pair exactly the way a live
# Evaluator_Run would, rather than constructing MetricResult/MetricFailure
# objects directly and exercising persistence.write_json/append_jsonl +
# read_jsonl in isolation. run_evaluation's own persistence step (Req 4.5)
# is the thing this property must validate, so driving it through
# run_evaluation -- using the same failing-pairs mechanism as
# test_run_evaluation_isolates_failing_scorer_calls above to get a
# deterministic, non-empty failures list alongside the results list --
# covers the actual write path end-to-end (run_evaluation's two
# append_jsonl calls) rather than only the lower-level helpers, while still
# reading the persisted files back with persistence.read_jsonl directly to
# check the round trip, independent of run_evaluation's own return value.
# ---------------------------------------------------------------------------

from persistence import read_jsonl  # noqa: E402
from models import MetricFailure  # noqa: E402


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(records_metrics_and_flags=_records_metrics_and_failure_flags_strategy)
def test_run_evaluation_persists_results_and_failures_without_loss(records_metrics_and_flags):
    records, metrics, should_fail_flags = records_metrics_and_flags

    results_scratch_path = OUTPUT_DIR / _SCRATCH_RESULTS_RELATIVE_PATH
    failures_scratch_path = OUTPUT_DIR / _SCRATCH_FAILURES_RELATIVE_PATH
    results_scratch_path.unlink(missing_ok=True)
    failures_scratch_path.unlink(missing_ok=True)

    all_pairs = [(record.record_id, metric.name) for record in records for metric in metrics]
    # Deterministically fail every other pair (by position) so both the
    # results list and the failures list are exercised together whenever
    # at least two pairs exist, while still covering the all-succeed and
    # all-empty cases when there are fewer pairs.
    failing_pairs = {pair for pair, should_fail in zip(all_pairs, should_fail_flags) if should_fail}

    scorer = _make_selective_failing_scorer(failing_pairs)

    results, failures = run_evaluation(metrics, records, scorer)

    # run_evaluation persists via append_jsonl, which only creates the
    # file on first write -- when a list is empty, no file (and therefore
    # no line) is ever written for it, so the round trip through a missing
    # file must still yield an empty list (read_jsonl's documented
    # missing-file behavior), covering the empty-list case for both.
    reloaded_results_raw = read_jsonl(results_scratch_path)
    reloaded_failures_raw = read_jsonl(failures_scratch_path)

    reloaded_results = [MetricResult.from_dict(row) for row in reloaded_results_raw]
    reloaded_failures = [MetricFailure.from_dict(row) for row in reloaded_failures_raw]

    # Same length, same order, same field values as what run_evaluation
    # returned and (by Property 10/11) produced exactly once per pair --
    # no loss, no duplication, no reordering across the persist-then-load
    # round trip.
    assert reloaded_results == results
    assert reloaded_failures == failures

    if not results:
        assert reloaded_results == []
    if not failures:
        assert reloaded_failures == []


def test_run_evaluation_persists_empty_results_and_failures_without_loss():
    """Explicit empty-list case for both results and failures: no records
    and no metrics means run_evaluation produces (and persists) an empty
    results list and an empty failures list, and reloading each via
    read_jsonl against the (never-created) scratch files yields [] for
    both -- the degenerate round trip."""
    results_scratch_path = OUTPUT_DIR / _SCRATCH_RESULTS_RELATIVE_PATH
    failures_scratch_path = OUTPUT_DIR / _SCRATCH_FAILURES_RELATIVE_PATH
    results_scratch_path.unlink(missing_ok=True)
    failures_scratch_path.unlink(missing_ok=True)

    scorer, _calls = _make_tracking_scorer()

    results, failures = run_evaluation([], [], scorer)

    assert results == []
    assert failures == []
    assert read_jsonl(results_scratch_path) == []
    assert read_jsonl(failures_scratch_path) == []


# ---------------------------------------------------------------------------
# Property 13: Reference labels are passed to the scorer exactly when a
# label exists for that record.
#
# Validates: Requirements 4.6
#
# APPROACH: run_evaluation calls feedback_store.load_feedback() with no
# argument, which resolves feedback_store.FEEDBACK_STORE_PATH at call time.
# tests/conftest.py's autouse fixture redirects that constant (along with
# the results/metrics path constants) to a per-test scratch directory under
# output/_test_scratch/, so this test's append_expected_state_label call and
# run_evaluation's internal read both hit the scratch store, never the real
# output/feedback/feedback_store.jsonl. The scratch directory is removed
# after the test.
#
# Each Hypothesis example still writes its label against a fresh,
# UUID4-suffixed record_id ("test-9-5-reference-label-pass-through-<uuid4>")
# because the scratch store persists across the examples of one
# test-function call, and unique ids keep each example's lookup unambiguous.
# ---------------------------------------------------------------------------

import uuid  # noqa: E402

import feedback_store  # noqa: E402
from models import ExpectedStateLabel  # noqa: E402

# A fixed, obviously test-scoped prefix for the one record_id this test
# writes a real label for, so the fresh UUID4 suffix generated per example
# can never collide with any other example's record_id within the same
# scratch Feedback_Store.
_LABELED_RECORD_ID_PREFIX = "test-9-5-reference-label-pass-through-"


def _make_reference_label_tracking_scorer():
    """Build a scorer that always succeeds with a fixed (score, rationale)
    and records the exact `reference_label` argument it was called with,
    keyed by record_id, via a closure-captured dict."""
    captured_reference_labels: dict[str, object] = {}

    def scorer(record, metric, reference_label):
        captured_reference_labels[record.record_id] = reference_label
        return (0.5, "ok")

    return scorer, captured_reference_labels


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(label_text=_free_text, unlabeled_record_id=_name_text, metrics=_metrics_strategy)
def test_run_evaluation_passes_reference_label_exactly_when_one_exists(
    label_text, unlabeled_record_id, metrics
):
    results_scratch_path = OUTPUT_DIR / _SCRATCH_RESULTS_RELATIVE_PATH
    failures_scratch_path = OUTPUT_DIR / _SCRATCH_FAILURES_RELATIVE_PATH
    results_scratch_path.unlink(missing_ok=True)
    failures_scratch_path.unlink(missing_ok=True)

    # A fresh, unique record_id per example -- see the note above. The
    # label lands in conftest.py's scratch Feedback_Store, not the real one.
    labeled_record_id = _LABELED_RECORD_ID_PREFIX + str(uuid.uuid4())
    # Guard against the vanishingly unlikely case where Hypothesis draws an
    # unlabeled_record_id that collides with the labeled one (or shares its
    # reserved prefix), so the two records in this example are always
    # distinguishable by label presence alone.
    if unlabeled_record_id == labeled_record_id or unlabeled_record_id.startswith(
        _LABELED_RECORD_ID_PREFIX
    ):
        unlabeled_record_id = "not-" + unlabeled_record_id

    feedback_store.append_expected_state_label(
        ExpectedStateLabel(
            record_id=labeled_record_id,
            agent_name="test-agent",
            label_text=label_text,
            timestamp="2024-01-01T00:00:00Z",
        )
    )

    records = [
        ReviewRecord(
            record_id=labeled_record_id,
            agent_name="test-agent",
            brief_text="brief for labeled record",
            responses={},
            expected_behavior=None,
        ),
        ReviewRecord(
            record_id=unlabeled_record_id,
            agent_name="test-agent",
            brief_text="brief for unlabeled record",
            responses={},
            expected_behavior=None,
        ),
    ]

    scorer, captured_reference_labels = _make_reference_label_tracking_scorer()

    run_evaluation(metrics, records, scorer)

    if metrics:
        # The labeled record's every scorer call received the stored
        # label's exact label_text as reference_label.
        assert captured_reference_labels[labeled_record_id] == label_text
        # The unlabeled record's every scorer call received None, since no
        # ExpectedStateLabel was ever written for it.
        assert captured_reference_labels[unlabeled_record_id] is None
    else:
        # No metrics means no (record, metric) pairs at all, so the scorer
        # is never invoked for either record.
        assert captured_reference_labels == {}
