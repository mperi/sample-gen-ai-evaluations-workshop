"""Data models for Agent_Health_Evaluation_Demo.

Plain ``dataclasses`` with explicit ``to_dict()``/``from_dict()`` methods for
JSON/JSONL round-tripping, following the sibling module's
(`Workload Specific Evaluations/Multiagent Shared Context Evaluation/dataset.py`)
dataclass conventions.

``ExpectedStateLabel`` and ``TaskSuccessFeedback`` both add an ``"entry_type"``
discriminator field to their ``to_dict()`` output (not a dataclass field
itself) so that ``feedback_store.py`` can tell which dataclass to reconstruct
when deserializing mixed rows from ``feedback_store.jsonl``. ``from_dict()``
on both strips that discriminator before reconstructing the dataclass.

See the design document's "Data Models" section for the authoritative
schemas: ``.kiro/specs/agent-health-evaluation-demo/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolDefinition:
    """One tool available to a peer agent: its name and stated purpose."""
    name: str
    purpose: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "purpose": self.purpose}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolDefinition":
        return cls(name=data["name"], purpose=data["purpose"])


@dataclass
class AgentContract:
    """A structured statement of an agent's role, responsibilities,
    constraints, and tools, derived from its system prompt."""
    agent_name: str                      # e.g. "market_trends"
    role: str                            # single-sentence stated role
    responsibilities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    tools: List[ToolDefinition] = field(default_factory=list)
    source_prompt: str = ""              # verbatim system prompt derived from

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "responsibilities": list(self.responsibilities),
            "constraints": list(self.constraints),
            "tools": [t.to_dict() for t in self.tools],
            "source_prompt": self.source_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentContract":
        return cls(
            agent_name=data["agent_name"],
            role=data["role"],
            responsibilities=list(data.get("responsibilities", [])),
            constraints=list(data.get("constraints", [])),
            tools=[ToolDefinition.from_dict(t) for t in data.get("tools", [])],
            source_prompt=data.get("source_prompt", ""),
        )


@dataclass
class CandidateHealthMetric:
    """One LLM-proposed health metric, traceable to an AgentContract statement."""
    name: str
    definition: str
    rubric: str                          # scoring rubric/scale description
    traceability: str                    # reference to the motivating contract statement
    agent_name: str                      # which agent's contract this metric was proposed for

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "definition": self.definition,
            "rubric": self.rubric,
            "traceability": self.traceability,
            "agent_name": self.agent_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateHealthMetric":
        return cls(
            name=data["name"],
            definition=data["definition"],
            rubric=data["rubric"],
            traceability=data["traceability"],
            agent_name=data["agent_name"],
        )


@dataclass
class RejectedMetric:
    """A raw candidate-metric proposal that was excluded for missing fields."""
    raw: Dict[str, Any]                  # the raw LLM proposal as received
    missing_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": dict(self.raw),
            "missing_fields": list(self.missing_fields),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RejectedMetric":
        return cls(
            raw=dict(data.get("raw", {})),
            missing_fields=list(data.get("missing_fields", [])),
        )


@dataclass
class ReviewRecord:
    """One reviewable interaction record, sourced read-only from the sibling
    module's review dataset."""
    record_id: str                       # sourced from review_cases.jsonl's case_id
    agent_name: str                      # single agent this record belongs to, when applicable
    brief_text: str
    responses: Dict[str, str] = field(default_factory=dict)   # {agent_name: response_text}
    expected_behavior: Optional[str] = None                   # tag from source data, if present

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "agent_name": self.agent_name,
            "brief_text": self.brief_text,
            "responses": dict(self.responses),
            "expected_behavior": self.expected_behavior,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewRecord":
        return cls(
            record_id=data["record_id"],
            agent_name=data["agent_name"],
            brief_text=data["brief_text"],
            responses=dict(data.get("responses", {})),
            expected_behavior=data.get("expected_behavior"),
        )


@dataclass
class ExpectedStateLabel:
    """A Domain_Expert-authored annotation stating what the agent should have
    done or known for a given record."""
    record_id: str
    agent_name: str
    label_text: str
    timestamp: str                       # ISO 8601

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_type": "expected_state_label",
            "record_id": self.record_id,
            "agent_name": self.agent_name,
            "label_text": self.label_text,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedStateLabel":
        data = {k: v for k, v in data.items() if k != "entry_type"}
        return cls(
            record_id=data["record_id"],
            agent_name=data["agent_name"],
            label_text=data["label_text"],
            timestamp=data["timestamp"],
        )


@dataclass
class TaskSuccessFeedback:
    """A Domain_Expert-authored pass/fail judgment plus rationale for a given
    record."""
    record_id: str
    agent_name: str
    passed: bool
    rationale: str
    timestamp: str                       # ISO 8601

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_type": "task_success_feedback",
            "record_id": self.record_id,
            "agent_name": self.agent_name,
            "passed": self.passed,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSuccessFeedback":
        data = {k: v for k, v in data.items() if k != "entry_type"}
        return cls(
            record_id=data["record_id"],
            agent_name=data["agent_name"],
            passed=data["passed"],
            rationale=data["rationale"],
            timestamp=data["timestamp"],
        )


@dataclass
class MetricResult:
    """One scored outcome of an Evaluator_Run for a given record/metric pair."""
    record_id: str
    metric_name: str
    agent_name: str
    score: float                         # normalized 0.0-1.0
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "metric_name": self.metric_name,
            "agent_name": self.agent_name,
            "score": self.score,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricResult":
        return cls(
            record_id=data["record_id"],
            metric_name=data["metric_name"],
            agent_name=data["agent_name"],
            score=data["score"],
            rationale=data["rationale"],
        )


@dataclass
class MetricFailure:
    """A failed scorer call for a given record/metric pair."""
    record_id: str
    metric_name: str
    error: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "metric_name": self.metric_name,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricFailure":
        return cls(
            record_id=data["record_id"],
            metric_name=data["metric_name"],
            error=data["error"],
        )


if __name__ == "__main__":
    # Lightweight round-trip sanity check for every dataclass. The full
    # property-based persistence round-trip suite is a separate later task.
    tool = ToolDefinition(name="search", purpose="look up market data")
    assert ToolDefinition.from_dict(tool.to_dict()) == tool

    contract = AgentContract(
        agent_name="market_trends",
        role="Analyze market trends",
        responsibilities=["Summarize trends", "Flag risks"],
        constraints=["Must cite sources"],
        tools=[tool],
        source_prompt="You are a market trends analyst...",
    )
    assert AgentContract.from_dict(contract.to_dict()) == contract

    metric = CandidateHealthMetric(
        name="groundedness",
        definition="Whether claims are grounded in source facts",
        rubric="0-1 scale, 1 = fully grounded",
        traceability="role statement: analyze market trends",
        agent_name="market_trends",
    )
    assert CandidateHealthMetric.from_dict(metric.to_dict()) == metric

    rejected = RejectedMetric(raw={"name": "latency"}, missing_fields=["definition", "rubric", "traceability"])
    assert RejectedMetric.from_dict(rejected.to_dict()) == rejected

    record = ReviewRecord(
        record_id="case-1",
        agent_name="market_trends",
        brief_text="Analyze the Q3 market.",
        responses={"market_trends": "The market grew 5%."},
        expected_behavior="normal_grounded_analysis",
    )
    assert ReviewRecord.from_dict(record.to_dict()) == record

    label = ExpectedStateLabel(
        record_id="case-1",
        agent_name="market_trends",
        label_text="Should have flagged missing source.",
        timestamp="2024-01-01T00:00:00Z",
    )
    label_dict = label.to_dict()
    assert label_dict["entry_type"] == "expected_state_label"
    assert ExpectedStateLabel.from_dict(label_dict) == label

    feedback = TaskSuccessFeedback(
        record_id="case-1",
        agent_name="market_trends",
        passed=False,
        rationale="Missed the flawed premise.",
        timestamp="2024-01-01T00:05:00Z",
    )
    feedback_dict = feedback.to_dict()
    assert feedback_dict["entry_type"] == "task_success_feedback"
    assert TaskSuccessFeedback.from_dict(feedback_dict) == feedback

    result = MetricResult(
        record_id="case-1",
        metric_name="groundedness",
        agent_name="market_trends",
        score=0.82,
        rationale="Mostly grounded, one unsupported claim.",
    )
    assert MetricResult.from_dict(result.to_dict()) == result

    failure = MetricFailure(
        record_id="case-1",
        metric_name="groundedness",
        error="ThrottlingException",
    )
    assert MetricFailure.from_dict(failure.to_dict()) == failure

    print("All model round-trip sanity checks passed.")
