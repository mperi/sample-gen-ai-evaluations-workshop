"""Dashboard_Publisher for Agent_Health_Evaluation_Demo.

Phase 4 of the demo: publishes ``MetricResult`` aggregates to CloudWatch
custom metrics (namespace ``AgentHealthDemo``) so a workload owner can
monitor agent health over time rather than reading raw result files.

This module is split into two kinds of functions, matching design.md's
"Dashboard_Publisher" component section:

- Pure aggregation (``aggregate_by_metric``, ``aggregate_by_record``): no
  AWS dependency, safe to call from anywhere, including the Review_UI's
  per-record join (Requirement 5.5). (task 10.1)
- CloudWatch publishing (``build_metric_datum``, ``publish_aggregates``):
  calls CloudWatch's ``PutMetricData`` API. Authenticated via boto3's
  standard credential chain (SigV4/AWS_IAM), the paved path for
  service-to-service AWS calls — no bespoke auth. (task 10.4)

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import boto3

from model_config import AWS_REGION, CLOUDWATCH_NAMESPACE
from models import MetricResult


def aggregate_by_metric(results: List[MetricResult]) -> Dict[str, float]:
    """Group ``results`` by ``metric_name`` and compute the mean ``score``
    per group.

    Pure function, no AWS dependency — used by both the Dashboard_Publisher
    (to compute the per-metric aggregate it publishes to CloudWatch) and the
    Review_UI's per-record join via ``aggregate_by_record``.

    Args:
        results: the ``MetricResult``s to aggregate, e.g. as loaded from
            ``output/results/metric_results.jsonl``. May span any number of
            distinct ``record_id``s and ``metric_name``s.

    Returns:
        A dict mapping each distinct ``metric_name`` present in ``results``
        to the mean of that metric's own ``score`` values. Empty dict if
        ``results`` is empty.

    Validates: Requirement 5.1 (Property 14 — per-metric aggregation never
    mixes results from different metric names).
    """
    scores_by_metric: "defaultdict[str, List[float]]" = defaultdict(list)
    for result in results:
        scores_by_metric[result.metric_name].append(result.score)

    return {
        metric_name: sum(scores) / len(scores)
        for metric_name, scores in scores_by_metric.items()
    }


def aggregate_by_record(results: List[MetricResult], record_id: str) -> Dict[str, float]:
    """Compute the per-metric mean score for just one ``record_id``.

    Filters ``results`` down to those matching ``record_id``, then applies
    the same per-metric mean computation as ``aggregate_by_metric``. Used by
    the Review_UI to show, for a selected Review_Dataset record, each
    metric's aggregate score next to that record (Requirement 5.5).

    Args:
        results: the ``MetricResult``s to filter and aggregate.
        record_id: the ``ReviewRecord.record_id`` to restrict aggregation to.

    Returns:
        A dict mapping each distinct ``metric_name`` present among
        ``record_id``'s results to the mean of that metric's score values
        for ``record_id`` only. Empty dict if no result in ``results``
        matches ``record_id``.
    """
    matching_results = [result for result in results if result.record_id == record_id]
    return aggregate_by_metric(matching_results)


def build_metric_datum(agent_name: str, metric_name: str, value: float) -> Dict:
    """Construct one CloudWatch ``MetricDatum`` dict for a single aggregate.

    Pure function, no AWS call — just builds the dict shape. Labels the
    datum with both the ``AgentName`` and ``MetricName`` dimensions so the
    published data point is identifiable regardless of whether the later
    ``put_metric_data`` call that carries it succeeds (Requirement 5.2).

    Args:
        agent_name: the Peer_Agent_Config name the aggregate belongs to.
        metric_name: the Candidate_Health_Metric name being published.
        value: the aggregate value (e.g. the mean score from
            ``aggregate_by_metric``) to publish for this metric.

    Returns:
        A dict shaped like a single boto3 CloudWatch ``MetricDatum``:
        ``{"MetricName": ..., "Dimensions": [...], "Value": ..., "Unit": "None"}``.

    Validates: Requirement 5.2 (every published data point is labeled with
    the agent name and metric name it represents).
    """
    return {
        "MetricName": metric_name,
        "Dimensions": [
            {"Name": "AgentName", "Value": agent_name},
            {"Name": "MetricName", "Value": metric_name},
        ],
        "Value": value,
        "Unit": "None",
    }


@dataclass
class PublishReport:
    """Outcome of one ``publish_aggregates`` call.

    Attributes:
        published: metric names that were successfully published, mapped to
            the aggregate value sent for each.
        failures: metric names that failed to publish, mapped to a string
            describing the exception that was raised for that metric
            (Requirement 5.3). Does not include metrics that published
            successfully.
    """
    published: Dict[str, float] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)


def publish_aggregates(
    results: List[MetricResult],
    agent_name: str,
    cloudwatch_client=None,
) -> PublishReport:
    """Aggregate ``results`` per metric and publish each to CloudWatch.

    Computes per-metric means via ``aggregate_by_metric``, then calls
    ``cloudwatch_client.put_metric_data`` once per metric, each call wrapped
    in its own ``try/except`` so a failure publishing one metric's aggregate
    is recorded by metric name and does NOT abort publishing the remaining
    metrics (Requirement 5.3).

    ``cloudwatch_client`` defaults to a real
    ``boto3.client("cloudwatch", region_name=AWS_REGION)`` when not supplied,
    so unit tests can inject a stub/mock instead. Constructing that default
    client is intentionally left OUTSIDE the per-metric try/except: if it
    fails (e.g. no AWS credentials configured), that failure propagates as a
    clear configuration error rather than being swallowed — Dashboard_Publisher
    must not silently fall back to a mocked publishing path (Requirement 5.4).

    Args:
        results: the ``MetricResult``s to aggregate and publish.
        agent_name: the Peer_Agent_Config name these results belong to; used
            to label every published datum via ``build_metric_datum``.
        cloudwatch_client: an object exposing ``put_metric_data(Namespace=,
            MetricData=)``, e.g. a real ``boto3`` CloudWatch client or a test
            stub/mock. Defaults to a real boto3 client when omitted.

    Returns:
        A ``PublishReport`` recording which metrics published successfully
        and which failed (with their error messages).

    Validates: Requirements 5.2, 5.3, 5.4.
    """
    if cloudwatch_client is None:
        # Left outside the per-metric try/except below: a failure here (e.g.
        # missing credentials) is a configuration error and must propagate,
        # not be treated as a per-metric publish failure (Req 5.4).
        cloudwatch_client = boto3.client("cloudwatch", region_name=AWS_REGION)

    aggregates = aggregate_by_metric(results)
    report = PublishReport()

    for metric_name, value in aggregates.items():
        try:
            datum = build_metric_datum(agent_name, metric_name, value)
            cloudwatch_client.put_metric_data(
                Namespace=CLOUDWATCH_NAMESPACE,
                MetricData=[datum],
            )
        except Exception as exc:  # noqa: BLE001 - per-metric isolation (Req 5.3)
            report.failures[metric_name] = str(exc)
            continue
        report.published[metric_name] = value

    return report


if __name__ == "__main__":
    # Sanity check, run with: python dashboard_publisher.py
    results = [
        MetricResult(record_id="case-1", metric_name="groundedness", agent_name="market_trends",
                     score=0.8, rationale="Mostly grounded."),
        MetricResult(record_id="case-1", metric_name="completeness", agent_name="market_trends",
                     score=0.6, rationale="Missed one point."),
        MetricResult(record_id="case-2", metric_name="groundedness", agent_name="customer_insights",
                     score=0.4, rationale="Some unsupported claims."),
        MetricResult(record_id="case-2", metric_name="completeness", agent_name="customer_insights",
                     score=1.0, rationale="Fully complete."),
    ]

    def _rounded(d: Dict[str, float]) -> Dict[str, float]:
        return {k: round(v, 6) for k, v in d.items()}

    by_metric = aggregate_by_metric(results)
    assert _rounded(by_metric) == {"groundedness": 0.6, "completeness": 0.8}, by_metric
    print(f"[ok] aggregate_by_metric() computed per-metric means across both records: {by_metric}")

    by_record_case1 = aggregate_by_record(results, "case-1")
    assert _rounded(by_record_case1) == {"groundedness": 0.8, "completeness": 0.6}, by_record_case1
    print(f"[ok] aggregate_by_record('case-1') only averaged case-1's results "
          f"(no leakage from case-2): {by_record_case1}")

    by_record_case2 = aggregate_by_record(results, "case-2")
    assert _rounded(by_record_case2) == {"groundedness": 0.4, "completeness": 1.0}, by_record_case2
    print(f"[ok] aggregate_by_record('case-2') only averaged case-2's results: {by_record_case2}")

    empty_by_metric = aggregate_by_metric([])
    assert empty_by_metric == {}, empty_by_metric
    print("[ok] aggregate_by_metric([]) returned {} for an empty input list")

    no_match_by_record = aggregate_by_record(results, "case-does-not-exist")
    assert no_match_by_record == {}, no_match_by_record
    print("[ok] aggregate_by_record() returned {} for a record_id with no matches")

    # -- build_metric_datum -------------------------------------------------
    datum = build_metric_datum("market_trends", "groundedness", 0.6)
    assert datum == {
        "MetricName": "groundedness",
        "Dimensions": [
            {"Name": "AgentName", "Value": "market_trends"},
            {"Name": "MetricName", "Value": "groundedness"},
        ],
        "Value": 0.6,
        "Unit": "None",
    }, datum
    print(f"[ok] build_metric_datum() built a correctly labeled MetricDatum dict: {datum}")

    # -- publish_aggregates (stub cloudwatch client, no live AWS call) -----
    class _StubCloudWatchClient:
        """Fake cloudwatch client: records calls, fails one metric on purpose."""

        def __init__(self):
            self.calls = []

        def put_metric_data(self, Namespace, MetricData):
            self.calls.append({"Namespace": Namespace, "MetricData": MetricData})
            if MetricData[0]["MetricName"] == "completeness":
                raise RuntimeError("simulated CloudWatch throttling error")

    stub_client = _StubCloudWatchClient()
    publish_report = publish_aggregates(results, "market_trends", cloudwatch_client=stub_client)

    # One put_metric_data call per metric (groundedness, completeness).
    assert len(stub_client.calls) == 2, stub_client.calls
    print(f"[ok] publish_aggregates() called put_metric_data once per metric: "
          f"{[c['MetricData'][0]['MetricName'] for c in stub_client.calls]}")

    for call in stub_client.calls:
        assert call["Namespace"] == CLOUDWATCH_NAMESPACE, call
        metric_name = call["MetricData"][0]["MetricName"]
        expected_value = by_metric[metric_name]
        assert call["MetricData"][0] == build_metric_datum("market_trends", metric_name, expected_value), call
    print("[ok] every put_metric_data call used Namespace=CLOUDWATCH_NAMESPACE and a "
          "MetricData datum matching build_metric_datum()'s output exactly")

    assert publish_report.published == {"groundedness": by_metric["groundedness"]}, publish_report.published
    assert set(publish_report.failures) == {"completeness"}, publish_report.failures
    assert "simulated CloudWatch throttling error" in publish_report.failures["completeness"]
    print(f"[ok] publish_aggregates() recorded the simulated 'completeness' failure "
          f"without aborting 'groundedness': published={publish_report.published}, "
          f"failures={publish_report.failures}")

    print("\nAll dashboard_publisher sanity checks passed.")
