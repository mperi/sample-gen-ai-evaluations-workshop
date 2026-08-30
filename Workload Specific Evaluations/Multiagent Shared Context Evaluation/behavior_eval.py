"""Batch expected-behavior evaluation for the sequential multiagent pipeline.

Runs every brief in the dataset through the pipeline and grades each output
against its ``expected_behavior`` (flag_invalid_premise, flag_missing_source,
require_source_citation, resist_prompt_injection, reconcile_contradiction, or
normal_grounded_analysis) using ``LLMJudge.judge_expected_behavior``.

Decoupled from the notebook: pass in a ``run_fn`` that executes one brief's
turns and returns the pipeline's responses. This keeps the agent wiring in the
notebook and the evaluation logic here.

Usage (in the notebook)
-----------------------
    from behavior_eval import run_behavior_suite, behavior_report
    from metrics_collector import LLMJudge

    def run_fn(brief):
        # run the brief's turns through the peers, return {agent_name: response}
        _, memory = run_session(brief.turns, brief.id, source_facts=brief.source_facts)
        return {e["agent"]: e["content"] for e in memory}

    results = run_behavior_suite(run_fn, LLMJudge(region=region))
    display(Markdown(behavior_report(results)))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from dataset import load_briefs, Brief


@dataclass
class BehaviorResult:
    brief_id: str
    scenario: str
    expected_behavior: str
    passed: bool
    reasoning: str


def _combine(responses: Dict[str, str]) -> str:
    """Flatten {agent_name: response} into one text blob for the judge."""
    return "\n\n".join(f"[{name}]\n{text}" for name, text in responses.items())


def run_behavior_suite(run_fn: Callable[[Brief], Dict[str, str]],
                       judge,
                       briefs: Optional[List[Brief]] = None) -> List[BehaviorResult]:
    """Run every brief through ``run_fn`` and grade against expected_behavior.

    ``run_fn(brief) -> {agent_name: response}``.
    ``judge`` must expose ``judge_expected_behavior(expected, brief_text, combined)``.
    """
    if briefs is None:
        briefs = load_briefs()

    results: List[BehaviorResult] = []
    for b in briefs:
        responses = run_fn(b)
        combined = _combine(responses)
        brief_text = "\n\n".join(b.turns)
        verdict = judge.judge_expected_behavior(
            b.expected_behavior, brief_text, combined)
        results.append(BehaviorResult(
            brief_id=b.id,
            scenario=b.scenario,
            expected_behavior=b.expected_behavior,
            passed=bool(verdict.get("pass")),
            reasoning=verdict.get("reasoning", ""),
        ))
    return results


def behavior_report(results: List[BehaviorResult]) -> str:
    """Markdown report: per-case verdicts + overall and per-behavior pass rates."""
    lines = ["### Expected-Behavior Conformance", ""]

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    rate = (passed / total * 100) if total else 0.0
    lines.append(f"**Overall pass rate:** {passed}/{total} ({rate:.0f}%)")
    lines.append("")

    # Per-behavior breakdown
    by_behavior: Dict[str, List[BehaviorResult]] = {}
    for r in results:
        by_behavior.setdefault(r.expected_behavior, []).append(r)
    lines.append("**By expected behavior:**")
    for behavior, group in sorted(by_behavior.items()):
        p = sum(1 for r in group if r.passed)
        lines.append(f"- `{behavior}`: {p}/{len(group)}")
    lines.append("")

    # Per-case table
    lines.append("| Brief | Expected behavior | Result | Reasoning |")
    lines.append("|-------|-------------------|--------|-----------|")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        reason = r.reasoning.replace("|", "\\|")
        lines.append(f"| {r.brief_id} | `{r.expected_behavior}` | {mark} | {reason} |")

    return "\n".join(lines)
