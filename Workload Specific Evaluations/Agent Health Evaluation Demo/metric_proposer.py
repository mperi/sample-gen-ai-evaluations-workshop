"""Metric_Proposer for Agent_Health_Evaluation_Demo.

This module implements all of Phase 1b (see design.md's "Metric_Proposer"
component section): submitting an ``AgentContract`` to Bedrock and asking it
to propose a list of candidate health metrics (``propose_metrics``),
validating/filtering that raw LLM output into accepted
``CandidateHealthMetric`` records and rejected ``RejectedMetric`` records
(``validate_candidate``/``missing_required_fields``/``filter_candidates``,
task 3.1), persisting both lists under this package's own ``output/``
directory, and loading a (possibly hand-edited) persisted candidate-metrics
file back in for later phases (``load_metrics``).

Design on the ``agent_name`` question (see design.md's data model for
``CandidateHealthMetric``, which requires an ``agent_name`` field):
raw LLM proposals describe *metrics*, not *which agent they're for* — the
agent is an artifact of which ``AgentContract`` was submitted to the LLM in
the first place, a fact the caller already knows and the LLM response
itself does not need to repeat. So ``agent_name`` is **not** one of the
fields ``validate_candidate`` checks for "present and non-empty" (per
Requirement 2.2, which lists exactly ``name``/``definition``/``rubric``/
``traceability``); it is instead supplied by the caller to
``filter_candidates`` and stamped onto every accepted
``CandidateHealthMetric`` it constructs. This keeps ``validate_candidate``
a pure function of the raw dict's own four required fields, while still
guaranteeing ``filter_candidates`` can always build a valid
``CandidateHealthMetric`` (which requires ``agent_name``) whenever
``validate_candidate`` returns ``True``.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from contract_deriver import _invoke_model
from model_config import METRIC_PROPOSER_MODEL_ID
from models import AgentContract, CandidateHealthMetric, RejectedMetric
from persistence import append_jsonl, read_jsonl

logger = logging.getLogger("metric_proposer")

CANDIDATE_METRICS_PATH = "metrics/candidate_metrics.jsonl"
REJECTED_METRICS_PATH = "metrics/candidate_metrics.rejected.jsonl"

# The fields Requirement 2.2 requires every Candidate_Health_Metric to
# include. Order here is also the order missing_fields are reported in.
REQUIRED_FIELDS: Tuple[str, ...] = ("name", "definition", "rubric", "traceability")


def _is_present(raw: Dict[str, Any], field: str) -> bool:
    """True iff ``raw[field]`` exists and is not ``None``/empty/whitespace.

    A missing key, a ``None`` value, and a string containing only
    whitespace (including the empty string ``""``) all count as "not
    present". Non-string values (e.g. a number) that are merely truthy are
    treated as present without further inspection, since the required
    fields are documented as text (name/definition/rubric/traceability).
    """
    if field not in raw:
        return False
    value = raw[field]
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def missing_required_fields(raw: Dict[str, Any]) -> List[str]:
    """Return exactly which of ``REQUIRED_FIELDS`` are absent/empty in ``raw``.

    Preserves ``REQUIRED_FIELDS`` order. Returns an empty list when every
    required field is present and non-empty.
    """
    return [field for field in REQUIRED_FIELDS if not _is_present(raw, field)]


def validate_candidate(raw: Dict[str, Any]) -> bool:
    """True iff ``raw`` has non-empty values for all of Requirement 2.2's
    required fields: ``name``, ``definition``, ``rubric``, and
    ``traceability``.

    A missing key, a ``None`` value, or an empty/whitespace-only string for
    any of those four fields makes this return ``False``. ``agent_name`` is
    deliberately not checked here — see this module's docstring for why.

    Args:
        raw: a single raw candidate-metric proposal dict, as received from
            the LLM (before any ``agent_name`` is attached).

    Returns:
        ``True`` if all four required fields are present and non-empty,
        ``False`` otherwise.
    """
    return len(missing_required_fields(raw)) == 0


def filter_candidates(
    raw_candidates: List[Dict[str, Any]], agent_name: str
) -> Tuple[List[CandidateHealthMetric], List[RejectedMetric]]:
    """Partition raw candidate-metric proposals into accepted and rejected.

    Every dict in ``raw_candidates`` is checked with ``validate_candidate``
    and lands in exactly one of the two returned lists — there is no loss
    and no duplication, and relative order within each output list matches
    ``raw_candidates``'s order.

    Args:
        raw_candidates: raw LLM-proposed metric dicts, each expected to
            (if valid) carry ``name``/``definition``/``rubric``/
            ``traceability``.
        agent_name: the agent whose ``AgentContract`` these proposals were
            generated from. Stamped onto every accepted
            ``CandidateHealthMetric`` (the raw dicts themselves are not
            required to, and generally will not, carry this field). Not
            itself subject to the omitted-field check, since it is
            supplied by the caller rather than the LLM.

    Returns:
        A ``(accepted, rejected)`` tuple:
          - ``accepted``: one ``CandidateHealthMetric`` per raw dict that
            passed ``validate_candidate``, with ``agent_name`` filled in
            from the ``agent_name`` parameter.
          - ``rejected``: one ``RejectedMetric`` per raw dict that failed
            ``validate_candidate``, carrying the original dict verbatim as
            ``raw`` and exactly which of the four required fields were
            missing/empty as ``missing_fields``.
    """
    accepted: List[CandidateHealthMetric] = []
    rejected: List[RejectedMetric] = []

    for raw in raw_candidates:
        missing = missing_required_fields(raw)
        if not missing:
            accepted.append(
                CandidateHealthMetric(
                    name=raw["name"],
                    definition=raw["definition"],
                    rubric=raw["rubric"],
                    traceability=raw["traceability"],
                    agent_name=agent_name,
                )
            )
        else:
            rejected.append(RejectedMetric(raw=dict(raw), missing_fields=missing))

    return accepted, rejected


# ---------------------------------------------------------------------------
# LLM-calling half (task 3.4)
# ---------------------------------------------------------------------------

_METRIC_PROPOSAL_PROMPT_TEMPLATE = """You are proposing candidate health metrics for an AI agent named \
"{agent_name}", based on a structured contract describing what it is supposed to do.

Role:
{role}

Responsibilities:
{responsibilities}

Constraints:
{constraints}

Tools available:
{tools}

Propose a list of candidate health metrics that could be used to evaluate this agent's quality and \
behavior. Each metric must include:
- "name": a short, specific metric name.
- "definition": what the metric measures.
- "rubric": the scoring rubric or scale used to score it (e.g. "0-1 scale, 1 = fully grounded").
- "traceability": a reference to the specific role/responsibility/constraint statement above that \
motivated this metric.

Respond with JSON only, no commentary, as a JSON list of objects in this exact shape:
[{{"name": "...", "definition": "...", "rubric": "...", "traceability": "..."}}, ...]"""


def _format_bullet_list(items: List[str]) -> str:
    """Render ``items`` as a newline-separated "- " bulleted list, or a
    placeholder line when ``items`` is empty."""
    if not items:
        return "(none stated)"
    return "\n".join(f"- {item}" for item in items)


def _build_metric_proposal_prompt(contract: AgentContract) -> str:
    """Build the Bedrock prompt for ``propose_metrics``, embedding the
    contract's role, responsibilities, constraints, and tools."""
    tools_text = (
        "\n".join(f"- {t.name}: {t.purpose}" for t in contract.tools)
        if contract.tools else "(none)"
    )
    return _METRIC_PROPOSAL_PROMPT_TEMPLATE.format(
        agent_name=contract.agent_name,
        role=contract.role,
        responsibilities=_format_bullet_list(contract.responsibilities),
        constraints=_format_bullet_list(contract.constraints),
        tools=tools_text,
    )


def propose_metrics(
    contract: AgentContract, model_id: str = METRIC_PROPOSER_MODEL_ID
) -> Tuple[List[CandidateHealthMetric], List[RejectedMetric]]:
    """Ask Bedrock to propose candidate health metrics for ``contract``, then
    validate/filter and persist the results (Requirement 2.1, 2.3, 2.4).

    Builds a prompt embedding ``contract``'s role, responsibilities,
    constraints, and tools, calls Bedrock via ``contract_deriver._invoke_model``
    (reusing that module's markdown-fence-stripping JSON extraction and
    ``ThrottlingException`` backoff/retry pattern rather than duplicating it),
    and expects a raw JSON list of metric-proposal dicts back. That raw list
    is passed through ``filter_candidates`` using ``contract.agent_name``.

    Unlike ``contract_deriver.derive_contract``, there is no non-LLM
    fallback here: per design.md's Error Handling section, a Bedrock call
    that fails after exhausting ``_invoke_model``'s retries is a
    user-visible failure of Phase 1 and must raise rather than be silently
    swallowed.

    Accepted candidates are appended to ``output/metrics/candidate_metrics.jsonl``
    and rejected candidates to ``output/metrics/candidate_metrics.rejected.jsonl``
    (Requirement 2.4), one ``persistence.append_jsonl`` call per metric.

    Args:
        contract: the ``AgentContract`` to propose metrics for.
        model_id: the Bedrock model id to use; defaults to
            ``METRIC_PROPOSER_MODEL_ID``.

    Returns:
        The ``(accepted, rejected)`` tuple from ``filter_candidates``.

    Raises:
        ValueError: if the LLM response is not a JSON list.
        Exception: whatever ``_invoke_model`` raises after exhausting its
            retries (e.g. a ``RuntimeError`` wrapping the last
            ``ThrottlingException``, or any other non-retryable error such
            as malformed JSON or missing credentials).
    """
    prompt = _build_metric_proposal_prompt(contract)
    # A JSON list of several metric proposals (each with its own name/
    # definition/rubric/traceability) needs more headroom than
    # contract_deriver's single-object 1024-token default to avoid a
    # truncated/unterminated JSON response.
    parsed = _invoke_model(prompt, model_id=model_id, max_tokens=4096)

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected a JSON list of metric proposals from the LLM, got "
            f"{type(parsed).__name__}: {parsed!r}"
        )

    accepted, rejected = filter_candidates(parsed, contract.agent_name)

    for metric in accepted:
        append_jsonl(CANDIDATE_METRICS_PATH, metric.to_dict())
    for rejection in rejected:
        append_jsonl(REJECTED_METRICS_PATH, rejection.to_dict())

    return accepted, rejected


def load_metrics(path: str) -> List[CandidateHealthMetric]:
    """Load a (possibly hand-edited) persisted ``candidate_metrics.jsonl``
    file into a list of ``CandidateHealthMetric``s, for use by later phases
    (Requirement 2.5).

    Args:
        path: path to the JSONL file to load, e.g.
            ``"output/metrics/candidate_metrics.jsonl"``. Passed straight
            through to ``persistence.read_jsonl``, which opens the file in
            ``"r"`` mode only and never enforces output-dir containment, so
            a hand-edited copy anywhere on disk can be loaded.

    Returns:
        One ``CandidateHealthMetric`` per non-blank line in the file, in
        file order. Returns an empty list if the file does not exist.
    """
    return [CandidateHealthMetric.from_dict(row) for row in read_jsonl(path)]


if __name__ == "__main__":
    # Lightweight sanity check (see also tests/test_metric_proposer.py once
    # the property/example tests for this module are added in tasks 3.2/3.3).
    fully_populated = {
        "name": "groundedness",
        "definition": "Whether claims are grounded in source facts",
        "rubric": "0-1 scale, 1 = fully grounded",
        "traceability": "role statement: analyze market trends",
    }
    missing_rubric = {
        "name": "latency",
        "definition": "How quickly the agent responds",
        "rubric": "",
        "traceability": "responsibility: respond promptly",
    }
    empty_proposal: Dict[str, Any] = {}

    assert validate_candidate(fully_populated) is True
    assert validate_candidate(missing_rubric) is False
    assert validate_candidate(empty_proposal) is False

    raw_candidates = [fully_populated, missing_rubric, empty_proposal]
    accepted, rejected = filter_candidates(raw_candidates, agent_name="market_trends")

    assert len(accepted) == 1, accepted
    assert accepted[0].name == "groundedness"
    assert accepted[0].agent_name == "market_trends"

    assert len(rejected) == 2, rejected
    missing_rubric_rejection = next(r for r in rejected if r.raw == missing_rubric)
    assert missing_rubric_rejection.missing_fields == ["rubric"], missing_rubric_rejection.missing_fields

    empty_rejection = next(r for r in rejected if r.raw == empty_proposal)
    assert empty_rejection.missing_fields == ["name", "definition", "rubric", "traceability"], (
        empty_rejection.missing_fields
    )

    assert len(accepted) + len(rejected) == len(raw_candidates)

    print(f"accepted: {len(accepted)}, rejected: {len(rejected)}")
    print("All metric_proposer validation/filtering sanity checks passed.")

    # ------------------------------------------------------------------
    # Live sanity check for propose_metrics/load_metrics (task 3.4). Makes
    # a real Bedrock call against the already-persisted market_trends
    # contract (produced by contract_deriver.py, task 2.1).
    # ------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO)

    from persistence import read_json, resolve_output_path

    contract_path = resolve_output_path("contracts", "market_trends.json")
    market_trends_contract = AgentContract.from_dict(read_json(contract_path))
    print(f"\nLoaded contract for agent: {market_trends_contract.agent_name}")

    live_accepted, live_rejected = propose_metrics(market_trends_contract)

    assert len(live_accepted) >= 1, "expected at least one accepted candidate metric"
    for live_metric in live_accepted:
        assert live_metric.agent_name == "market_trends", live_metric
        assert live_metric.name.strip()
        assert live_metric.definition.strip()
        assert live_metric.rubric.strip()
        assert live_metric.traceability.strip()
    print(f"[ok] propose_metrics returned {len(live_accepted)} accepted and "
          f"{len(live_rejected)} rejected candidate(s) for 'market_trends'")
    for live_metric in live_accepted:
        print(f"      - {live_metric.name}: {live_metric.definition}")

    candidate_metrics_file = resolve_output_path("metrics", "candidate_metrics.jsonl")
    assert candidate_metrics_file.exists(), candidate_metrics_file
    print(f"[ok] {candidate_metrics_file} exists")

    if live_rejected:
        rejected_metrics_file = resolve_output_path("metrics", "candidate_metrics.rejected.jsonl")
        assert rejected_metrics_file.exists(), rejected_metrics_file
        print(f"[ok] {rejected_metrics_file} exists")

    loaded_metrics = load_metrics(str(candidate_metrics_file))
    assert len(loaded_metrics) >= len(live_accepted), (
        f"expected at least the {len(live_accepted)} metric(s) just appended, "
        f"got {len(loaded_metrics)}"
    )
    for live_metric in live_accepted:
        assert live_metric in loaded_metrics, live_metric
    print(f"[ok] load_metrics round-tripped {len(loaded_metrics)} persisted "
          f"candidate metric(s), including all newly proposed ones")

    print("\nAll propose_metrics/load_metrics live sanity checks passed.")
