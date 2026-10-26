"""Offline tests for run_pipeline.py (fake scorer + stub CloudWatch client).

No AWS calls: ``main`` is given an explicit ``scorer`` (which also skips the
up-front AgentCore client check) and a stub ``cloudwatch_client``. All writes
land in tests/conftest.py's scratch directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

import run_pipeline  # noqa: E402
from models import ReviewRecord  # noqa: E402


def _record(record_id, responses):
    return ReviewRecord(record_id=record_id, agent_name="multi", brief_text="b", responses=responses)


def test_select_agent_view_targets_requested_agent_and_skips_missing():
    records = [
        _record("r1", {"customer_insights": "ci text", "market_trends": "mt text"}),
        _record("r2", {"customer_insights": "ci only"}),
    ]
    selected, skipped = run_pipeline.select_agent_view(records, "market_trends")
    assert [r.record_id for r in selected] == ["r1"]
    assert selected[0].agent_name == "market_trends"
    # The scorer's exact-match lookup now returns market_trends' text,
    # even though it is not the first response in the dict.
    assert selected[0].responses[selected[0].agent_name] == "mt text"
    assert skipped == ["r2"]
    assert records[0].agent_name == "multi"  # originals untouched


def test_short_rationale_strips_adapter_note():
    text = "[agentcore_scorer adapter note: blah -- not a direct evaluation of that rubric.] The answer is good."
    assert run_pipeline.short_rationale(text) == "The answer is good."


class _StubCloudWatch:
    def __init__(self):
        self.calls = []

    def put_metric_data(self, **kwargs):
        self.calls.append(kwargs)


def test_main_end_to_end_with_fakes(tmp_path, _isolate_output_data_paths, monkeypatch):
    metrics_file = tmp_path / "metrics.jsonl"
    rows = [
        {"name": f"M{i}", "definition": "d", "rubric": "r", "traceability": "t", "agent_name": "market_trends"}
        for i in range(3)
    ] + [{"name": "Other", "definition": "d", "rubric": "r", "traceability": "t", "agent_name": "strategy_synth"}]
    metrics_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    records = [
        _record("r1", {"market_trends": "mt1", "customer_insights": "ci1"}),
        _record("r2", {"customer_insights": "ci2", "market_trends": "mt2"}),
        _record("r3", {"market_trends": "mt3"}),
    ]
    monkeypatch.setattr(run_pipeline.persistence, "load_review_records", lambda: records)

    scored_texts = []

    def fake_scorer(record, metric, reference_label):
        scored_texts.append(record.responses[record.agent_name])
        return 0.5, f"[x -- not a direct evaluation of that rubric.] ok {record.record_id}"

    stub = _StubCloudWatch()
    rc = run_pipeline.main(
        ["--records", "2", "--metrics", "2", "--metrics-file", str(metrics_file)],
        scorer=fake_scorer,
        cloudwatch_client=stub,
    )
    assert rc == 0
    assert sorted(set(scored_texts)) == ["mt1", "mt2"]  # market_trends text only
    assert len(scored_texts) == 4
    assert len(stub.calls) == 2
    dims = stub.calls[0]["MetricData"][0]["Dimensions"]
    assert {"Name": "AgentName", "Value": "market_trends"} in dims

    summaries = list((_isolate_output_data_paths / "runs").glob("*_summary.md"))
    assert len(summaries) == 1
    body = summaries[0].read_text(encoding="utf-8")
    assert "| r1 | M0 | 0.50 | ok r1 |" in body
    assert "published M0 = 0.500" in body
