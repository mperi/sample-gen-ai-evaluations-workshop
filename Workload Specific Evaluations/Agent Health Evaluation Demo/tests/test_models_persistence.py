"""Property-based tests for AgentContract / CandidateHealthMetric
persistence round-tripping.

**Property 4: Contract and candidate-metric persistence round-trips**

Validates: Requirements 1.5, 2.4, 2.5

For any valid ``AgentContract``, writing it via ``persistence.write_json``
(through ``AgentContract.to_dict()``) and reading it back via
``persistence.read_json`` + ``AgentContract.from_dict()`` must yield a value
equal to the original -- same field values, same order for list fields.

For any list of valid ``CandidateHealthMetric`` records (including the empty
list), writing each via ``persistence.append_jsonl`` (through
``.to_dict()``) to a JSONL path and reading back via
``persistence.read_jsonl`` + ``CandidateHealthMetric.from_dict()`` per line
must yield an equal list in the same order.

All writes in this test module go through ``persistence.resolve_output_path``
(transitively, via ``write_json``/``append_jsonl``), so every artifact lands
under this package's own ``output/`` directory -- specifically under a
dedicated ``output/_test_artifacts/`` subdirectory that is cleaned up by a
fixture after each test, so repeated runs never pollute the real
``output/contracts`` or ``output/metrics`` directories used by the other
phases.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from models import AgentContract, CandidateHealthMetric, ToolDefinition  # noqa: E402
from persistence import OUTPUT_DIR, read_json, read_jsonl, write_json  # noqa: E402
import persistence  # noqa: E402


# ---------------------------------------------------------------------------
# Test-artifact containment
# ---------------------------------------------------------------------------

# A dedicated subdirectory under output/ used only by this test module, so
# round-trip artifacts never land in (or collide with) output/contracts or
# output/metrics, which are used by the real contract_deriver/metric_proposer
# modules. Removed in its entirety by the fixture below after every test.
_TEST_ARTIFACTS_SUBDIR = "_test_artifacts_models_persistence"


@pytest.fixture(autouse=True)
def _cleanup_test_artifacts():
    """Remove this module's scratch output directory before and after each
    test so repeated test runs never pollute (or are polluted by) the real
    output/ directory."""
    scratch_dir = OUTPUT_DIR / _TEST_ARTIFACTS_SUBDIR
    shutil.rmtree(scratch_dir, ignore_errors=True)
    yield
    shutil.rmtree(scratch_dir, ignore_errors=True)


def _scratch_path(*parts: str) -> str:
    """Build a path (relative, under output/) rooted inside this module's
    scratch subdirectory, suitable for passing to write_json/append_jsonl."""
    return str(Path(_TEST_ARTIFACTS_SUBDIR, *parts))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Reasonably-constrained text: non-empty after stripping, no surrogate
# characters (which are unpaired and not valid UTF-8, so json.dump/json.load
# would round-trip them inconsistently -- an irrelevant edge case for this
# property, which is about ordinary field round-tripping, not UTF-8 edge
# cases). Leading/trailing whitespace-only strings are also excluded since
# they are not representative of real field content and add no coverage
# value here.
_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

# source_prompt / rubric / definition can reasonably be longer free text,
# including empty (an agent could in principle have an empty source prompt
# field once already derived -- the round-trip property itself doesn't
# depend on non-emptiness, only equality after round-tripping).
_free_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=200,
)

_agent_name = _text
_tool_strategy = st.builds(ToolDefinition, name=_text, purpose=_free_text)

_agent_contract_strategy = st.builds(
    AgentContract,
    agent_name=_agent_name,
    role=_text,
    responsibilities=st.lists(_text, max_size=5),
    constraints=st.lists(_text, max_size=5),
    tools=st.lists(_tool_strategy, max_size=4),
    source_prompt=_free_text,
)

_candidate_health_metric_strategy = st.builds(
    CandidateHealthMetric,
    name=_text,
    definition=_free_text,
    rubric=_free_text,
    traceability=_free_text,
    agent_name=_agent_name,
)


# ---------------------------------------------------------------------------
# Property: AgentContract round-trips through write_json/read_json
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(contract=_agent_contract_strategy)
def test_agent_contract_persistence_round_trip(contract):
    """Writing an AgentContract via write_json(contract.to_dict()) and
    reading it back via read_json + AgentContract.from_dict() yields a value
    equal to the original, including list-field order."""
    rel_path = _scratch_path(f"{uuid.uuid4().hex}.json")

    write_json(rel_path, contract.to_dict())
    resolved = persistence.resolve_output_path(*Path(rel_path).parts)
    loaded = AgentContract.from_dict(read_json(resolved))

    assert loaded == contract
    assert loaded.responsibilities == contract.responsibilities
    assert loaded.constraints == contract.constraints
    assert [t.to_dict() for t in loaded.tools] == [t.to_dict() for t in contract.tools]


# ---------------------------------------------------------------------------
# Property: CandidateHealthMetric list round-trips through
# append_jsonl/read_jsonl, including the empty-list case
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(metrics=st.lists(_candidate_health_metric_strategy, max_size=10))
def test_candidate_health_metric_list_persistence_round_trip(metrics):
    """Writing a list of CandidateHealthMetric records one-by-one via
    append_jsonl(metric.to_dict()) and reading them back via read_jsonl +
    CandidateHealthMetric.from_dict() per line yields an equal list in the
    same order, including when the input list is empty."""
    rel_path = _scratch_path(f"{uuid.uuid4().hex}.jsonl")
    resolved = persistence.resolve_output_path(*Path(rel_path).parts)

    for metric in metrics:
        persistence.append_jsonl(rel_path, metric.to_dict())

    loaded_rows = read_jsonl(resolved)
    loaded_metrics = [CandidateHealthMetric.from_dict(row) for row in loaded_rows]

    assert loaded_metrics == metrics


def test_candidate_health_metric_empty_list_round_trip_produces_no_rows():
    """Explicit pin for the empty-list case: writing zero metrics means the
    file is never created, and reading back a nonexistent JSONL path yields
    an empty list (per read_jsonl's documented missing-file behavior)."""
    rel_path = _scratch_path(f"{uuid.uuid4().hex}.jsonl")
    resolved = persistence.resolve_output_path(*Path(rel_path).parts)

    assert not resolved.exists()
    assert read_jsonl(resolved) == []
