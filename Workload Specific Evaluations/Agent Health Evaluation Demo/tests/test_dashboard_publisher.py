"""Property-based test for dashboard_publisher.py's per-metric aggregation.

**Property 14: Per-metric aggregation never mixes results from different
metric names**

Validates: Requirements 5.1

For any generated list of ``MetricResult``s spanning multiple distinct
``metric_name``s, ``aggregate_by_metric(results)`` returns exactly one
aggregate value per distinct ``metric_name`` present in ``results``, and
that value is computed ONLY from that metric's own results (never mixed
with another metric's scores).

``aggregate_by_metric`` is a pure function -- no AWS call and no file I/O --
so this test exercises it directly against in-memory ``MetricResult``
lists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from dashboard_publisher import aggregate_by_metric  # noqa: E402
from models import MetricResult  # noqa: E402


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Small fixed pools for metric_name/record_id/agent_name so that a generated
# list of results naturally produces multiple results sharing the same
# metric_name (and the same record_id), exercising the grouping logic
# rather than just producing all-distinct singletons.
_METRIC_NAMES = ["groundedness", "completeness", "relevance", "coherence", "safety"]
_RECORD_IDS = ["case-1", "case-2", "case-3", "case-4"]
_AGENT_NAMES = ["market_trends", "customer_insights", "ops_agent"]

_metric_name_strategy = st.sampled_from(_METRIC_NAMES)
_record_id_strategy = st.sampled_from(_RECORD_IDS)
_agent_name_strategy = st.sampled_from(_AGENT_NAMES)

_score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_rationale_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=30,
)

_metric_result_strategy = st.builds(
    MetricResult,
    record_id=_record_id_strategy,
    metric_name=_metric_name_strategy,
    agent_name=_agent_name_strategy,
    score=_score_strategy,
    rationale=_rationale_strategy,
)

_results_strategy = st.lists(_metric_result_strategy, min_size=0, max_size=40)


def _expected_means(results: list[MetricResult]) -> dict[str, float]:
    """Independently compute the expected per-metric mean score by grouping
    ``results`` by ``metric_name`` and averaging their scores -- an
    implementation that does not share any logic with
    ``aggregate_by_metric`` itself, used as the oracle for this test."""
    scores_by_metric: dict[str, list[float]] = {}
    for result in results:
        scores_by_metric.setdefault(result.metric_name, []).append(result.score)
    return {
        metric_name: sum(scores) / len(scores)
        for metric_name, scores in scores_by_metric.items()
    }


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(results=_results_strategy)
def test_aggregate_by_metric_isolates_results_by_metric_name(results):
    aggregates = aggregate_by_metric(results)
    expected = _expected_means(results)

    # Exactly one aggregate value per distinct metric_name present in
    # results -- no metric missing, no extra metric invented.
    assert set(aggregates.keys()) == set(expected.keys())

    # Each aggregate value is computed only from that metric's own results.
    for metric_name, expected_mean in expected.items():
        assert aggregates[metric_name] == pytest.approx(expected_mean, abs=1e-9)


def test_aggregate_by_metric_empty_list_returns_empty_dict():
    """Explicit edge-case example: aggregate_by_metric([]) returns {}."""
    assert aggregate_by_metric([]) == {}


# ---------------------------------------------------------------------------
# Property 17: Per-record aggregate lookup returns exactly the metrics with
# supporting results for that record
#
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

from dashboard_publisher import aggregate_by_record  # noqa: E402


def _expected_record_means(results: list[MetricResult], record_id: str) -> dict[str, float]:
    """Independently compute the expected per-metric mean score for just
    ``record_id`` -- filters to ``record_id``'s own results, groups by
    ``metric_name``, and averages -- without calling ``aggregate_by_record``
    or ``aggregate_by_metric``, used as the oracle for this test."""
    scores_by_metric: dict[str, list[float]] = {}
    for result in results:
        if result.record_id != record_id:
            continue
        scores_by_metric.setdefault(result.metric_name, []).append(result.score)
    return {
        metric_name: sum(scores) / len(scores)
        for metric_name, scores in scores_by_metric.items()
    }


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(results=_results_strategy, record_id=_record_id_strategy)
def test_aggregate_by_record_returns_exactly_that_records_metrics(results, record_id):
    aggregates = aggregate_by_record(results, record_id)
    expected = _expected_record_means(results, record_id)

    # The key set is exactly the metric_names present among record_id's own
    # results -- no extra metric invented, none missing.
    assert set(aggregates.keys()) == set(expected.keys())

    # Each value is computed only from record_id's own results for that
    # metric (no leakage from other records).
    for metric_name, expected_mean in expected.items():
        assert aggregates[metric_name] == pytest.approx(expected_mean, abs=1e-9)


def test_aggregate_by_record_no_matching_results_returns_empty_dict():
    """Explicit edge-case example: a record_id with zero matching results
    returns {}."""
    results = [
        MetricResult(record_id="case-1", metric_name="groundedness", agent_name="market_trends",
                     score=0.8, rationale="Mostly grounded."),
        MetricResult(record_id="case-2", metric_name="completeness", agent_name="customer_insights",
                     score=0.6, rationale="Missed one point."),
    ]
    assert aggregate_by_record(results, "case-does-not-exist") == {}


# ---------------------------------------------------------------------------
# Property 15: Published metric data points are always labeled with the
# correct agent and metric name
#
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------

from dashboard_publisher import build_metric_datum  # noqa: E402


# Broad character range -- unicode letters, spaces, punctuation -- plus the
# empty string, since build_metric_datum has no validation that would
# reject unusual or empty agent/metric names. The datum must label the
# value with exactly what it was given, with no truncation, escaping, or
# field-swapping, no matter how unusual the string.
_unusual_text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=50,
)

_value_strategy = st.floats(allow_nan=False, allow_infinity=False)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    agent_name=_unusual_text_strategy,
    metric_name=_unusual_text_strategy,
    value=_value_strategy,
)
def test_build_metric_datum_labels_with_exact_agent_and_metric_name(agent_name, metric_name, value):
    datum = build_metric_datum(agent_name, metric_name, value)

    # The datum's MetricName and Value must equal the inputs exactly -- no
    # truncation or escaping that changes the value.
    assert datum["MetricName"] == metric_name
    assert datum["Value"] == value

    # Exactly two dimensions: AgentName and MetricName, each carrying the
    # exact input string -- no swapped fields, nothing dropped or added.
    dimensions = datum["Dimensions"]
    assert len(dimensions) == 2
    assert {"Name": "AgentName", "Value": agent_name} in dimensions
    assert {"Name": "MetricName", "Value": metric_name} in dimensions


def test_build_metric_datum_example_with_normal_inputs():
    """Explicit edge-case example: typical ASCII inputs produce the exact
    labeled shape documented in the module's docstring."""
    datum = build_metric_datum("market_trends", "groundedness", 0.6)
    assert datum == {
        "MetricName": "groundedness",
        "Dimensions": [
            {"Name": "AgentName", "Value": "market_trends"},
            {"Name": "MetricName", "Value": "groundedness"},
        ],
        "Value": 0.6,
        "Unit": "None",
    }


# ---------------------------------------------------------------------------
# Property 16: A failing publish call is isolated to its own metric and does
# not abort the remaining publishes
#
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------

from dashboard_publisher import publish_aggregates  # noqa: E402


class _StubCloudWatchClientWithFailures:
    """Stub CloudWatch client (no real AWS call): records every
    ``put_metric_data`` call and raises for any metric name in
    ``failing_metric_names``, succeeding for all others."""

    def __init__(self, failing_metric_names: set[str]):
        self.failing_metric_names = failing_metric_names
        self.calls: list[dict] = []

    def put_metric_data(self, Namespace, MetricData):
        metric_name = MetricData[0]["MetricName"]
        self.calls.append({"Namespace": Namespace, "MetricData": MetricData})
        if metric_name in self.failing_metric_names:
            raise RuntimeError(f"simulated publish failure for {metric_name!r}")


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    results=_results_strategy,
    failing_subset=st.sets(st.sampled_from(_METRIC_NAMES)),
)
def test_publish_aggregates_isolates_failures_per_metric(results, failing_subset):
    expected_aggregates = aggregate_by_metric(results)
    # Restrict the configured-to-fail set to metric names that will actually
    # be published this run, so the stub's failing set is a genuine subset of
    # the input metric names (an irrelevant failing name would never be
    # exercised and wouldn't test isolation).
    failing_metric_names = failing_subset & set(expected_aggregates.keys())

    stub_client = _StubCloudWatchClientWithFailures(failing_metric_names)
    report = publish_aggregates(results, "market_trends", cloudwatch_client=stub_client)

    # Every distinct metric_name in aggregate_by_metric(results) is attempted
    # exactly once via put_metric_data, regardless of any earlier failure --
    # no abort on the first raise.
    called_metric_names = [call["MetricData"][0]["MetricName"] for call in stub_client.calls]
    assert sorted(called_metric_names) == sorted(expected_aggregates.keys())
    assert len(called_metric_names) == len(set(called_metric_names))

    # Every input metric name appears in exactly one of the two outcome sets.
    assert set(report.published.keys()) | set(report.failures.keys()) == set(expected_aggregates.keys())
    assert set(report.published.keys()) & set(report.failures.keys()) == set()

    # `published` contains exactly the non-failing metric names, each mapped
    # to its correct aggregate value.
    non_failing = set(expected_aggregates.keys()) - failing_metric_names
    assert set(report.published.keys()) == non_failing
    for metric_name in non_failing:
        assert report.published[metric_name] == pytest.approx(expected_aggregates[metric_name], abs=1e-9)

    # `failures` contains exactly the failing metric names, each mapped to a
    # non-empty error-message string.
    assert set(report.failures.keys()) == failing_metric_names
    for metric_name in failing_metric_names:
        assert isinstance(report.failures[metric_name], str)
        assert len(report.failures[metric_name]) > 0


def test_publish_aggregates_one_failure_does_not_abort_remaining_example():
    """Explicit edge-case example: a single failing metric among several does
    not prevent the others from being published."""
    results = [
        MetricResult(record_id="case-1", metric_name="groundedness", agent_name="market_trends",
                     score=0.8, rationale="Mostly grounded."),
        MetricResult(record_id="case-1", metric_name="completeness", agent_name="market_trends",
                     score=0.6, rationale="Missed one point."),
        MetricResult(record_id="case-1", metric_name="relevance", agent_name="market_trends",
                     score=1.0, rationale="Fully relevant."),
    ]
    stub_client = _StubCloudWatchClientWithFailures({"completeness"})
    report = publish_aggregates(results, "market_trends", cloudwatch_client=stub_client)

    assert len(stub_client.calls) == 3
    assert report.published == {"groundedness": 0.8, "relevance": 1.0}
    assert set(report.failures.keys()) == {"completeness"}
    assert "completeness" in report.failures["completeness"]
