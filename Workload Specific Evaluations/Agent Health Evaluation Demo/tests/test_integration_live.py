"""Live, real-AWS-calling integration tests for Agent_Health_Evaluation_Demo.

Every test in this module is skipped by default and must be run explicitly
via:

    LIVE_AWS_TESTS=1 pytest tests/test_integration_live.py

These tests make genuine calls against real AWS services (AgentCore,
Bedrock, CloudWatch, etc. depending on the test) using whatever AWS
credentials/region are resolvable in the current environment. Each test
makes its real call exactly once -- never iterated or property-based --
consistent with the live-call posture used by the sibling
``Multiagent Shared Context Evaluation`` module's own live-call scripts
(e.g. ``run_behavior_live.py``).

None of these tests run as part of the normal unit/property test suite
(``pytest tests/``): the module-level ``LIVE_AWS_TESTS`` flag below
defaults to ``False`` unless the ``LIVE_AWS_TESTS`` environment variable is
explicitly set to ``"1"``, and every test is marked
``@pytest.mark.skipif(not LIVE_AWS_TESTS, ...)`` accordingly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from dashboard_publisher import publish_aggregates  # noqa: E402
from evaluator_runner import agentcore_scorer  # noqa: E402
from models import CandidateHealthMetric, MetricResult, ReviewRecord  # noqa: E402

LIVE_AWS_TESTS = os.environ.get("LIVE_AWS_TESTS") == "1"


@pytest.mark.skipif(not LIVE_AWS_TESTS, reason="Live AgentCore call; set LIVE_AWS_TESTS=1 to run")
def test_agentcore_scorer_live_call():
    """Exercise one real ``agentcore_scorer`` call against the live
    AgentCore evaluation service (Req 4.2, 4.7).

    Builds one simple ``ReviewRecord``/``CandidateHealthMetric`` pair, calls
    ``agentcore_scorer(record, metric, reference_label=None)`` directly
    (one real call, not iterated), and asserts the returned
    ``(score, rationale)`` tuple is well-formed: ``score`` a ``float`` in
    ``[0.0, 1.0]``, and ``rationale`` a non-empty ``str`` carrying this
    adapter's limitation-note prefix (see ``agentcore_scorer``'s docstring
    in ``evaluator_runner.py`` for the exact text and why it is present on
    every call).
    """
    record = ReviewRecord(
        record_id="live-integration-case-1",
        agent_name="market_trends",
        brief_text="Summarize the current market trend for electric vehicles.",
        responses={
            "market_trends": (
                "Electric vehicle sales have grown steadily over the past "
                "year, driven by falling battery costs and expanding "
                "charging infrastructure."
            ),
        },
    )
    metric = CandidateHealthMetric(
        name="groundedness",
        definition="The response is grounded in the brief and contains no fabricated claims.",
        rubric="Score 1.0 if every claim in the response is supported by the brief; 0.0 otherwise.",
        traceability="Derived from the agent's contract responsibility to report only sourced facts.",
        agent_name="market_trends",
    )

    score, rationale = agentcore_scorer(record, metric, reference_label=None)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    assert isinstance(rationale, str)
    assert rationale
    assert "agentcore_scorer adapter note" in rationale


@pytest.mark.skipif(not LIVE_AWS_TESTS, reason="Live CloudWatch PutMetricData call; set LIVE_AWS_TESTS=1 to run")
def test_publish_aggregates_live_call():
    """Exercise one real ``publish_aggregates`` call against the live
    CloudWatch ``PutMetricData`` API (Req 5.4).

    Builds a small set of ``MetricResult``s spanning two metric names, lets
    ``publish_aggregates`` construct a real
    ``boto3.client("cloudwatch", region_name=AWS_REGION)`` (``cloudwatch_client=None``),
    and makes one real ``put_metric_data`` call per metric under the
    ``AgentHealthDemo`` namespace -- additive custom metric data points,
    never deleting or modifying anything. Asserts the returned
    ``PublishReport.published`` contains both metric names with no
    ``.failures`` entries.
    """
    results = [
        MetricResult(record_id="live-integration-case-1", metric_name="groundedness",
                     agent_name="test_agent_health_demo", score=0.9,
                     rationale="Response is well grounded in the brief."),
        MetricResult(record_id="live-integration-case-1", metric_name="completeness",
                     agent_name="test_agent_health_demo", score=0.75,
                     rationale="Covers most of the required points."),
        MetricResult(record_id="live-integration-case-2", metric_name="groundedness",
                     agent_name="test_agent_health_demo", score=0.6,
                     rationale="One claim lacked a clear source."),
    ]

    report = publish_aggregates(results, agent_name="test_agent_health_demo", cloudwatch_client=None)

    assert set(report.published) == {"groundedness", "completeness"}
    assert report.failures == {}


@pytest.mark.skipif(not LIVE_AWS_TESTS, reason="Live Bedrock call; set LIVE_AWS_TESTS=1 to run")
def test_propose_metrics_live_call():
    """Exercise one real ``metric_proposer.propose_metrics`` call against the
    live Bedrock service (Req 2.1, 2.3, 2.4).

    Loads the already-persisted ``market_trends`` ``AgentContract`` from
    ``output/contracts/market_trends.json`` (produced by
    ``contract_deriver.py``) via ``persistence.read_json`` +
    ``AgentContract.from_dict``, then calls
    ``metric_proposer.propose_metrics(contract)`` directly (one real call,
    not iterated). Asserts the returned ``(accepted, rejected)`` tuple has
    at least one accepted metric, and that every accepted
    ``CandidateHealthMetric`` has non-empty ``name``/``definition``/
    ``rubric``/``traceability`` and the correct ``agent_name``.

    Note: ``propose_metrics`` persists accepted/rejected candidates via
    ``metric_proposer.CANDIDATE_METRICS_PATH``/``REJECTED_METRICS_PATH``.
    ``tests/conftest.py`` redirects both to a per-test scratch directory
    under ``output/_test_scratch/``, so this live call never appends to the
    real ``output/metrics/candidate_metrics.jsonl``.
    """
    from metric_proposer import propose_metrics
    from models import AgentContract
    from persistence import read_json, resolve_output_path

    contract_path = resolve_output_path("contracts", "market_trends.json")
    contract = AgentContract.from_dict(read_json(contract_path))

    accepted, rejected = propose_metrics(contract)

    assert len(accepted) >= 1

    for metric in accepted:
        assert metric.agent_name == "market_trends"
        assert metric.name.strip()
        assert metric.definition.strip()
        assert metric.rubric.strip()
        assert metric.traceability.strip()


# This test launches a local Streamlit subprocess rather than calling AWS,
# so it is not gated on `LIVE_AWS_TESTS`. It is instead gated on whether the
# `streamlit` CLI is actually importable/executable in this environment --
# consistent with this module's theme of "slow/special-setup checks that
# don't run by default", but scoped to the specific prerequisite this test
# actually needs rather than reusing the AWS-credential-focused flag for a
# check that makes no AWS calls at all.
_STREAMLIT_AVAILABLE = True
try:
    import streamlit as _streamlit  # noqa: F401
except ImportError:
    _STREAMLIT_AVAILABLE = False


def _find_free_port() -> int:
    """Return a free local TCP port by binding to port 0 and releasing it."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="streamlit is not installed; install via requirements.txt to run")
def test_review_app_streamlit_launch_live():
    """Launch ``review_app.py`` with a real ``streamlit run`` subprocess and
    confirm it starts cleanly (Req 3.1).

    Picks a free local port, launches
    ``streamlit run review_app.py --server.headless true --server.port <port>``
    via ``subprocess.Popen``, and polls (briefly, up to a few seconds) for
    either an HTTP 200 from ``http://localhost:<port>/`` or startup output
    on the process's stdout/stderr. Asserts no traceback appears in the
    captured output, and that the app became reachable over HTTP. The
    subprocess is terminated in a ``finally`` block regardless of outcome.

    This does not make any AWS calls itself -- ``review_app.py`` only calls
    AWS-free, read-only ``persistence.load_review_records()`` on start (the
    sibling module's ``review/review_cases.jsonl`` is read from local disk)
    -- so no destructive or billable action occurs here beyond the local
    process launch.
    """
    import subprocess
    import time
    import urllib.error
    import urllib.request

    port = _find_free_port()
    url = f"http://localhost:{port}/"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PACKAGE_DIR / "review_app.py"),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
        ],
        cwd=str(PACKAGE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        reachable = False
        deadline = time.time() + 20
        while time.time() < deadline:
            if proc.poll() is not None:
                # Process exited early -- stop polling, fall through to the
                # output/traceback assertions below.
                break
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    reachable = response.status == 200
                if reachable:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(0.5)

        # Give the process a brief moment to flush any startup output before
        # we inspect it, without blocking indefinitely on a still-running
        # server process.
        time.sleep(0.5)

        output = ""
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""

        assert "Traceback (most recent call last):" not in output

        if not reachable and proc.poll() is not None:
            pytest.fail(
                f"streamlit process exited early (code {proc.returncode}) "
                f"before becoming reachable at {url}; output:\n{output}"
            )

        assert reachable, f"streamlit app never became reachable at {url}"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
