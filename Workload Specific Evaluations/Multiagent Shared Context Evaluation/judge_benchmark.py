"""Judge benchmark: does the expected_behavior judge agree with human labels?

The behavior suite (behavior_eval.py) uses an LLM judge to decide whether the
pipeline conformed to each case's expected_behavior. But nothing validates the
JUDGE itself. This benchmark closes that loop: a set of hand-labeled cases with
a known pass/fail verdict, run through the judge, scored for agreement.

Dataset: datasets/judge_benchmark.jsonl. Each row:
  case_id, expected_behavior, brief_text, pipeline_output,
  human_label ("pass"|"fail"), human_rationale, provenance

Run (live, real Bedrock):
    python judge_benchmark.py

Programmatic (e.g. tests) with a custom/stub judge:
    from judge_benchmark import load_cases, score_judge, benchmark_report
    results = score_judge(my_judge, load_cases())
    print(benchmark_report(results))
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "datasets", "judge_benchmark.jsonl")
_REQUIRED = ("case_id", "expected_behavior", "brief_text", "pipeline_output", "human_label")


@dataclass
class BenchmarkCase:
    case_id: str
    expected_behavior: str
    brief_text: str
    pipeline_output: str
    human_label: str          # "pass" | "fail"
    human_rationale: str = ""
    provenance: str = ""


@dataclass
class BenchmarkResult:
    case_id: str
    expected_behavior: str
    human_label: str          # "pass" | "fail"
    judge_label: str          # "pass" | "fail"
    agree: bool
    judge_reasoning: str = ""


def load_cases(path: str = DEFAULT_PATH) -> List[BenchmarkCase]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Judge benchmark not found: {path}")
    cases: List[BenchmarkCase] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [k for k in _REQUIRED if k not in row]
            if missing:
                raise ValueError(f"Line {line_no} missing keys: {missing}")
            if row["human_label"] not in ("pass", "fail"):
                raise ValueError(
                    f"Line {line_no} ({row['case_id']}): human_label must be "
                    f"'pass' or 'fail', got {row['human_label']!r}")
            cases.append(BenchmarkCase(
                case_id=row["case_id"],
                expected_behavior=row["expected_behavior"],
                brief_text=row["brief_text"],
                pipeline_output=row["pipeline_output"],
                human_label=row["human_label"],
                human_rationale=row.get("human_rationale", ""),
                provenance=row.get("provenance", ""),
            ))
    return cases


def score_judge(judge, cases: Optional[List[BenchmarkCase]] = None) -> List[BenchmarkResult]:
    """Run the judge over each case and compare to the human label.

    ``judge`` must expose
    ``judge_expected_behavior(expected_behavior, brief_text, combined_response)``
    returning a dict with a boolean ``pass``.
    """
    if cases is None:
        cases = load_cases()
    results: List[BenchmarkResult] = []
    for c in cases:
        verdict = judge.judge_expected_behavior(
            c.expected_behavior, c.brief_text, c.pipeline_output)
        judge_label = "pass" if verdict.get("pass") else "fail"
        results.append(BenchmarkResult(
            case_id=c.case_id,
            expected_behavior=c.expected_behavior,
            human_label=c.human_label,
            judge_label=judge_label,
            agree=(judge_label == c.human_label),
            judge_reasoning=verdict.get("reasoning", ""),
        ))
    return results


def benchmark_report(results: List[BenchmarkResult]) -> str:
    """Markdown report: accuracy, confusion matrix (human vs judge), disagreements."""
    total = len(results)
    agree = sum(1 for r in results if r.agree)
    acc = (agree / total * 100) if total else 0.0

    # Confusion counts, treating "pass" as the positive class.
    tp = sum(1 for r in results if r.human_label == "pass" and r.judge_label == "pass")
    tn = sum(1 for r in results if r.human_label == "fail" and r.judge_label == "fail")
    fp = sum(1 for r in results if r.human_label == "fail" and r.judge_label == "pass")
    fn = sum(1 for r in results if r.human_label == "pass" and r.judge_label == "fail")

    lines = ["### Judge Benchmark", ""]
    lines.append(f"**Agreement with human labels:** {agree}/{total} ({acc:.0f}%)")
    lines.append("")
    lines.append("**Confusion matrix** (positive class = `pass`):")
    lines.append("")
    lines.append("| | judge: pass | judge: fail |")
    lines.append("|---|---|---|")
    lines.append(f"| **human: pass** | {tp} (TP) | {fn} (FN) |")
    lines.append(f"| **human: fail** | {fp} (FP) | {tn} (TN) |")
    lines.append("")

    # False positives are the dangerous ones: judge PASSes a bad output.
    if fp:
        lines.append(f"**False positives (judge passed a bad output):** {fp} "
                     "-- these are the highest-risk disagreements.")
    else:
        lines.append("**No false positives** -- the judge did not pass any output a human failed.")
    lines.append("")

    disagreements = [r for r in results if not r.agree]
    if disagreements:
        lines.append("**Disagreements:**")
        lines.append("")
        lines.append("| Case | Behavior | Human | Judge | Judge reasoning |")
        lines.append("|------|----------|-------|-------|-----------------|")
        for r in disagreements:
            reason = r.judge_reasoning.replace("|", "\\|")
            lines.append(f"| {r.case_id} | `{r.expected_behavior}` | {r.human_label} "
                         f"| {r.judge_label} | {reason} |")
    else:
        lines.append("**No disagreements** -- judge matched every human label.")

    return "\n".join(lines)


def main():
    from metrics_collector import LLMJudge
    region = os.getenv("AWS_REGION", "us-west-2")
    judge = LLMJudge(region=region)
    cases = load_cases()
    print(f"Scoring judge over {len(cases)} benchmark cases in {region} ...\n")
    results = score_judge(judge, cases)
    print(benchmark_report(results))


if __name__ == "__main__":
    main()
