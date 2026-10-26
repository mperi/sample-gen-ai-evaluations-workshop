"""Minimal end-to-end CLI for Phases 3 (score) and 4 (publish).

Loads the persisted candidate metrics for one agent and the first N review
records, scores every (record, metric) pair with the live AgentCore scorer,
publishes per-metric aggregates to CloudWatch, prints a readable summary,
and saves that summary to ``output/runs/<timestamp>_summary.md``.

Example (3 records x 4 metrics = 12 AgentCore calls, then 4 PutMetricData):

    python run_pipeline.py --records 3 --metrics 4 --agent market_trends

Which response gets scored
--------------------------
The sibling module's review records are multi-agent: each record carries a
response from every peer agent, and ``ReviewRecord.agent_name`` is the
``"multi"`` sentinel. ``agentcore_scorer`` scores
``record.responses[record.agent_name]`` and, when that key is missing, falls
back to the *first* response in the dict -- which happens to be
market_trends' today only because of the order agents appear in
``pipeline_output``. To make the run coherent regardless of that ordering,
this CLI hands the scorer a per-agent view of each record
(``agent_name`` set to ``--agent``), so the scorer's exact-match lookup
picks ``--agent``'s own response and the fallback is never used. Records
with no response for ``--agent`` are skipped and listed in the summary.

All writes go through ``persistence`` (results via ``run_evaluation``,
the summary via ``persistence.write_text``), so output-dir containment
still holds.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import textwrap
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence, Tuple

import dashboard_publisher
import evaluator_runner
import metric_proposer
import persistence
from models import CandidateHealthMetric, MetricFailure, MetricResult, ReviewRecord

# Relative to output/; module-level so tests can redirect it (tests/conftest.py).
RUNS_DIR = "runs"

_RATIONALE_PREFIX_END = "-- not a direct evaluation of that rubric.]"


def select_agent_view(
    records: Sequence[ReviewRecord], agent: str
) -> Tuple[List[ReviewRecord], List[str]]:
    """Return per-agent copies of ``records`` plus the ids that were skipped.

    Each returned record has ``agent_name == agent`` so ``agentcore_scorer``
    scores ``responses[agent]`` directly. ``responses`` is left intact.
    """
    selected: List[ReviewRecord] = []
    skipped: List[str] = []
    for record in records:
        if record.responses.get(agent):
            selected.append(dataclasses.replace(record, agent_name=agent))
        else:
            skipped.append(record.record_id)
    return selected, skipped


def short_rationale(rationale: str, width: int = 160) -> str:
    """Drop the scorer's fixed adapter-note prefix and truncate for display."""
    text = rationale
    if _RATIONALE_PREFIX_END in text:
        text = text.split(_RATIONALE_PREFIX_END, 1)[1]
    text = " ".join(text.split())
    return textwrap.shorten(text, width=width, placeholder="...") if text else "(no explanation)"


def build_summary(
    *,
    agent: str,
    timestamp: str,
    records: Sequence[ReviewRecord],
    skipped: Sequence[str],
    metrics: Sequence[CandidateHealthMetric],
    results: Sequence[MetricResult],
    failures: Sequence[MetricFailure],
    report: Optional[dashboard_publisher.PublishReport],
) -> str:
    """Render the run summary as Markdown (also what gets printed)."""
    lines: List[str] = []
    lines.append(f"# Agent health run: {agent} ({timestamp})")
    lines.append("")
    lines.append(f"- Scorer: `agentcore_scorer` (evaluator `{evaluator_runner.DEFAULT_EVALUATOR_ID}`)")
    lines.append(f"- Scored response: `responses[{agent!r}]` of each record")
    lines.append(f"- Records ({len(records)}): " + ", ".join(f"`{r.record_id}`" for r in records))
    if skipped:
        lines.append(f"- Skipped (no {agent} response): " + ", ".join(f"`{s}`" for s in skipped))
    lines.append(f"- Metrics ({len(metrics)}): " + ", ".join(m.name for m in metrics))
    lines.append(
        f"- Pairs: {len(records) * len(metrics)} attempted, "
        f"{len(results)} scored, {len(failures)} failed"
    )
    lines.append("")

    lines.append("## Scores")
    lines.append("")
    lines.append("| Record | Metric | Score | Rationale (adapter note stripped) |")
    lines.append("|---|---|---|---|")
    for r in results:
        rationale = short_rationale(r.rationale).replace("|", "\\|")
        lines.append(f"| {r.record_id} | {r.metric_name} | {r.score:.2f} | {rationale} |")
    lines.append("")

    if failures:
        lines.append("## Failures (verbatim)")
        lines.append("")
        for f in failures:
            lines.append(f"- `{f.record_id}` / {f.metric_name}: {f.error}")
        lines.append("")

    lines.append("## Per-metric aggregates (mean score)")
    lines.append("")
    lines.append("| Metric | Mean | N |")
    lines.append("|---|---|---|")
    aggregates = dashboard_publisher.aggregate_by_metric(list(results))
    for name, mean in aggregates.items():
        n = sum(1 for r in results if r.metric_name == name)
        lines.append(f"| {name} | {mean:.3f} | {n} |")
    lines.append("")

    lines.append("## CloudWatch publish")
    lines.append("")
    if report is None:
        lines.append("Skipped (`--skip-publish`).")
    else:
        lines.append(
            f"Namespace `{dashboard_publisher.CLOUDWATCH_NAMESPACE}`, "
            f"dimensions AgentName=`{agent}`, MetricName=<metric>."
        )
        lines.append("")
        for name, value in report.published.items():
            lines.append(f"- published {name} = {value:.3f}")
        for name, err in report.failures.items():
            lines.append(f"- FAILED {name}: {err}")
    lines.append("")

    lines.append("## Full rationales")
    lines.append("")
    for r in results:
        lines.append(f"### {r.record_id} / {r.metric_name} ({r.score:.2f})")
        lines.append("")
        lines.append(r.rationale)
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=int, default=3, help="number of review records to score (first N)")
    parser.add_argument("--metrics", type=int, default=4, help="number of candidate metrics to use (first M for --agent)")
    parser.add_argument("--agent", default="market_trends", help="agent whose metrics and responses to evaluate")
    parser.add_argument(
        "--metrics-file",
        default=str(persistence.OUTPUT_DIR / "metrics" / "candidate_metrics.jsonl"),
        help="candidate metrics JSONL to load",
    )
    parser.add_argument("--skip-publish", action="store_true", help="score only; do not call CloudWatch")
    args = parser.parse_args(argv)
    if args.records < 1 or args.metrics < 1:
        parser.error("--records and --metrics must be >= 1")
    return args


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    scorer: Optional[Callable] = None,
    cloudwatch_client=None,
) -> int:
    """CLI entry point. ``scorer``/``cloudwatch_client`` are injectable for tests;
    when ``scorer`` is None the live AgentCore client is validated up front."""
    args = parse_args(argv)

    if scorer is None:
        # Fail fast on a config problem (missing package, credentials, region)
        # before any pair is scored.
        evaluator_runner._get_agentcore_client()
        scorer = evaluator_runner.agentcore_scorer

    metrics = [m for m in metric_proposer.load_metrics(args.metrics_file) if m.agent_name == args.agent]
    metrics = metrics[: args.metrics]
    if not metrics:
        print(f"No candidate metrics for agent {args.agent!r} in {args.metrics_file}", file=sys.stderr)
        return 2

    all_records = persistence.load_review_records()[: args.records]
    records, skipped = select_agent_view(all_records, args.agent)
    if not records:
        print(f"None of the first {args.records} review records has a {args.agent!r} response", file=sys.stderr)
        return 2

    print(f"Scoring {len(records)} record(s) x {len(metrics)} metric(s) for {args.agent} ...", flush=True)
    results, failures = evaluator_runner.run_evaluation(metrics, records, scorer)

    report = None
    if not args.skip_publish and results:
        report = dashboard_publisher.publish_aggregates(
            results, agent_name=args.agent, cloudwatch_client=cloudwatch_client
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = build_summary(
        agent=args.agent,
        timestamp=timestamp,
        records=records,
        skipped=skipped,
        metrics=metrics,
        results=results,
        failures=failures,
        report=report,
    )
    # Print everything except the long per-pair rationale appendix.
    print(summary.split("## Full rationales", 1)[0].rstrip())
    summary_path = persistence.write_text(f"{RUNS_DIR}/{timestamp}_summary.md", summary)
    print(f"\nSummary saved to {summary_path}")
    return 1 if failures or (report is not None and report.failures) else 0


if __name__ == "__main__":
    sys.exit(main())
