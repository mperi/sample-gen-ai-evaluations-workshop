"""Evaluation dataset loader for the sequential multiagent pipeline
(research brief -> market_trends -> customer_insights -> strategy_synth).

Option A ("input-grounded"): each brief carries the verifiable facts it states.
Those facts are threaded into the pipeline as ``source_facts`` so the
Analysis Groundedness judge can check whether agent outputs stay faithful to
the brief (and, for the flawed-premise brief, whether the agents flag it).

No gold/reference analyses yet — that is a future Option B iteration.

Usage
-----
    from dataset import load_briefs, facts_to_text

    for b in load_briefs():
        for turn_text in b.turns:
            collector.begin_turn(i, turn_text, source_facts=b.source_facts)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "datasets", "research_briefs.jsonl")

# Required top-level keys every row must define.
_REQUIRED = ("id", "scenario", "turns", "facts", "premise_valid")

# Machine-checkable expected behaviour for each case. Lets negative/adversarial
# cases be auto-scored instead of relying on prose notes.
EXPECTED_BEHAVIORS = {
    "normal_grounded_analysis",   # ordinary case: analyse faithfully to the facts
    "flag_invalid_premise",       # brief's premise is false -> agent should flag it
    "flag_missing_source",        # source/report the brief relies on is absent -> agent should flag the gap
    "require_source_citation",    # agent must cite a source; unsourced claims = hallucination failure
    "resist_prompt_injection",    # brief embeds injected instructions -> agent should ignore them
    "reconcile_contradiction",    # brief contains contradictory facts -> agent should surface, not silently pick
}


@dataclass
class Brief:
    """One evaluation case: a research brief and its verifiable facts."""
    id: str
    scenario: str
    industry: str
    turns: List[str]                 # one entry per conversation turn
    facts: Dict[str, Any]            # structured ground truth
    premise_valid: bool              # False = deliberately flawed premise
    expected_behavior: str = "normal_grounded_analysis"  # machine-checkable expectation
    notes: str = ""
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_adversarial(self) -> bool:
        """True for negative/adversarial cases (anything but the normal path)."""
        return self.expected_behavior != "normal_grounded_analysis"

    @property
    def source_facts(self) -> str:
        """Ground-truth reference string passed to the groundedness judge."""
        return facts_to_text(self.facts, self.premise_valid)


def facts_to_text(facts: Dict[str, Any], premise_valid: bool = True) -> str:
    """Render structured facts as a compact, judge-friendly reference block."""
    lines = ["Verified facts stated in the research brief:"]
    for key, val in facts.items():
        label = key.replace("_", " ")
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        lines.append(f"- {label}: {val}")
    if not premise_valid:
        lines.append(
            "- WARNING: the brief's core premise is flawed/fabricated. A correct "
            "analysis should flag that the described market does not exist rather "
            "than restating the claimed figures as fact."
        )
    return "\n".join(lines)


def _validate(row: Dict[str, Any], line_no: int) -> None:
    missing = [k for k in _REQUIRED if k not in row]
    if missing:
        raise ValueError(f"Brief on line {line_no} is missing keys: {missing}")
    if not isinstance(row["turns"], list) or not row["turns"]:
        raise ValueError(f"Brief '{row.get('id')}' (line {line_no}): 'turns' must be a non-empty list")
    if not isinstance(row["facts"], dict) or not row["facts"]:
        raise ValueError(f"Brief '{row.get('id')}' (line {line_no}): 'facts' must be a non-empty object")
    if not isinstance(row["premise_valid"], bool):
        raise ValueError(f"Brief '{row.get('id')}' (line {line_no}): 'premise_valid' must be a boolean")
    eb = row.get("expected_behavior", "normal_grounded_analysis")
    if eb not in EXPECTED_BEHAVIORS:
        raise ValueError(
            f"Brief '{row.get('id')}' (line {line_no}): unknown expected_behavior '{eb}'. "
            f"Allowed: {sorted(EXPECTED_BEHAVIORS)}")


def load_briefs(path: str = DEFAULT_PATH) -> List[Brief]:
    """Load and validate all briefs from the JSONL dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    briefs: List[Brief] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            _validate(row, line_no)
            briefs.append(Brief(
                id=row["id"],
                scenario=row["scenario"],
                industry=row.get("industry", ""),
                turns=row["turns"],
                facts=row["facts"],
                premise_valid=row["premise_valid"],
                expected_behavior=row.get("expected_behavior", "normal_grounded_analysis"),
                notes=row.get("notes", ""),
                _raw=row,
            ))
    ids = [b.id for b in briefs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate brief ids in dataset: {sorted(dupes)}")
    return briefs


def get_brief(brief_id: str, path: str = DEFAULT_PATH) -> Brief:
    """Fetch a single brief by id."""
    for b in load_briefs(path):
        if b.id == brief_id:
            return b
    raise KeyError(f"No brief with id '{brief_id}' in {path}")


if __name__ == "__main__":
    loaded = load_briefs()
    print(f"Loaded {len(loaded)} briefs from {DEFAULT_PATH}\n")
    n_adv = sum(1 for b in loaded if b.is_adversarial)
    print(f"({len(loaded) - n_adv} normal, {n_adv} negative/adversarial)\n")
    for b in loaded:
        flag = "" if not b.is_adversarial else f"  [{b.expected_behavior}]"
        print(f"  {b.id}: {b.scenario} ({b.industry}), {len(b.turns)} turn(s){flag}")
