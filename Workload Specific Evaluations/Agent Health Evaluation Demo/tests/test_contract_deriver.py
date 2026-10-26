"""Property-based tests for ``contract_deriver.derive_contract``.

**Property 1: Contract derivation preserves agent identity and never crashes
on non-empty prompts**

Validates: Requirements 1.1, 1.2

For any agent name and any non-empty system prompt string, deriving an
``AgentContract`` from them never raises, and the resulting contract's
``agent_name`` and ``source_prompt`` fields equal the inputs exactly, with
``role`` non-empty.

Mocking strategy for Property 1: ``derive_contract`` tries a live Bedrock
call (``_llm_extract`` -> ``_invoke_model``, which constructs a real
``boto3`` client) first, and only falls back to the deterministic
``_heuristic_extract`` splitter if that call raises. Running 100+ Hypothesis
examples against a real Bedrock endpoint would be slow, costly, and
non-deterministic. This test monkeypatches ``contract_deriver._llm_extract``
to always raise, forcing every example through the deterministic heuristic
fallback path. This is preferable to calling ``_heuristic_extract`` directly
because it still exercises ``derive_contract``'s real control flow (the
empty-prompt guard, the try/except fallback wiring, and the
``AgentContract`` construction) -- only the LLM network call itself is
stubbed out. The property under test is about ``derive_contract``'s
contract (identity preservation + non-crashing), not about which extraction
backend produced the role/responsibilities/constraints, so forcing the
fallback path is a faithful and deterministic way to validate it.

**Property 2: Tool definitions round-trip into the contract unchanged,
including the empty case**

Validates: Requirements 1.3, 1.4

For any generated list of ``(tool_name, tool_purpose)`` pairs (0 to ~8
items, including the empty list), constructing ``ToolDefinition`` objects
from them and calling ``derive_contract(agent_name, system_prompt,
tools=tool_defs)`` must produce an ``AgentContract`` whose ``tools`` list has
the same length and order as the input, with each entry's ``name``/
``purpose`` equal to the corresponding input pair. ``tools=None`` must
produce an empty ``tools`` list (Requirement 1.4's no-tools case).

This property is only about the ``tools`` parameter passing through
``derive_contract`` unchanged -- it has nothing to do with the quality of
the LLM-derived role/responsibilities/constraints extraction. To keep the
test fast and deterministic across 100+ Hypothesis examples (and to avoid
making live Bedrock calls or depending on AWS credentials being configured
in the test environment), this module monkeypatches
``contract_deriver._llm_extract`` with a fixed, instant stand-in that
returns a fixed ``(role, responsibilities, constraints)`` tuple. This
exercises the same code path a successful LLM call would (the ``tools``
field is copied verbatim regardless of which extraction path produced the
role/responsibilities/constraints), without the latency, flakiness, or
credential dependency of a real Bedrock call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from hypothesis import HealthCheck, given, settings, strategies as st

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

import contract_deriver  # noqa: E402
from contract_deriver import derive_contract  # noqa: E402
from models import AgentContract, ToolDefinition  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic stand-in for the LLM extraction path
# ---------------------------------------------------------------------------

_FIXED_ROLE = "Fixed role for testing."
_FIXED_RESPONSIBILITIES = ["Fixed responsibility one.", "Fixed responsibility two."]
_FIXED_CONSTRAINTS = ["Fixed constraint one."]


def _fake_llm_extract(agent_name, system_prompt):
    """Instant, deterministic stand-in for ``contract_deriver._llm_extract``.

    Ignores its arguments and returns a fixed tuple, so
    ``derive_contract`` never attempts a real Bedrock call during this
    test module's property runs.
    """
    return _FIXED_ROLE, list(_FIXED_RESPONSIBILITIES), list(_FIXED_CONSTRAINTS)


def _patch_llm_extract(monkeypatch):
    monkeypatch.setattr(contract_deriver, "_llm_extract", _fake_llm_extract)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Reasonably-constrained non-empty text for tool name/purpose fields, free
# of surrogate characters (not valid UTF-8, irrelevant to this property).
_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "")

_tool_pair_strategy = st.tuples(_text, _text)

_agent_name_strategy = _text
_system_prompt_strategy = st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != "")


# ---------------------------------------------------------------------------
# Property: tool definitions round-trip unchanged, including the empty case
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    agent_name=_agent_name_strategy,
    system_prompt=_system_prompt_strategy,
    tool_pairs=st.lists(_tool_pair_strategy, min_size=0, max_size=8),
)
def test_tool_definitions_round_trip_unchanged(monkeypatch, agent_name, system_prompt, tool_pairs):
    """Any list of (name, purpose) pairs, turned into ToolDefinitions and
    passed to derive_contract, comes back out as an equal-length,
    same-order list of tools with matching name/purpose fields."""
    _patch_llm_extract(monkeypatch)

    tool_defs = [ToolDefinition(name=name, purpose=purpose) for name, purpose in tool_pairs]

    contract = derive_contract(agent_name, system_prompt, tools=tool_defs)

    assert len(contract.tools) == len(tool_pairs)
    for returned_tool, (expected_name, expected_purpose) in zip(contract.tools, tool_pairs):
        assert returned_tool.name == expected_name
        assert returned_tool.purpose == expected_purpose


def test_tools_none_produces_empty_tools_list(monkeypatch):
    """Explicit pin for Requirement 1.4: a Peer_Agent_Config with no
    associated tool definitions (tools=None) produces an Agent_Contract
    whose tools list is empty."""
    _patch_llm_extract(monkeypatch)

    contract = derive_contract("market_trends", "You are a market analyst.", tools=None)

    assert contract.tools == []


def test_tools_omitted_defaults_to_empty_tools_list(monkeypatch):
    """Same as above but relying on the default parameter value rather than
    passing tools=None explicitly."""
    _patch_llm_extract(monkeypatch)

    contract = derive_contract("market_trends", "You are a market analyst.")

    assert contract.tools == []


# ---------------------------------------------------------------------------
# Property 3: Deriving contracts for a batch of distinct agents produces one
# distinct contract per agent
#
# Validates: Requirements 1.6
#
# derive_all -> derive_contract -> _llm_extract (live Bedrock call) with a
# fallback to _heuristic_extract on any failure. Making 100+ live Bedrock
# calls per test run would be slow, costly, and flaky (throttling). Since
# this property is about derive_all's batching/ordering/distinctness
# behavior -- not about what _llm_extract specifically returns -- we
# monkeypatch contract_deriver._llm_extract to a fast, deterministic stub
# that echoes back a role derived from the inputs. This forces the same
# code path derive_contract would take on an LLM success, without any
# network call, while still exercising the real derive_contract/derive_all
# logic (identity passthrough, ordering, list construction).
# ---------------------------------------------------------------------------

def _fake_llm_extract_for_batch(agent_name, system_prompt):
    """Deterministic stand-in for contract_deriver._llm_extract: no network
    call, just echoes back values derived from its inputs."""
    return (
        f"role-for-{agent_name}",
        [f"responsibility-of-{agent_name}"],
        [f"constraint-of-{agent_name}"],
    )


# Agent names: non-empty strings safe to use as dict/identity keys. Avoid
# leading/trailing whitespace so that "distinct after strip" isn't a
# confusing edge case -- distinctness here is about the literal agent_name
# strings passed through unchanged.
_batch_agent_name_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=20,
).map(lambda s: s.strip()).filter(lambda s: s != "")

# System prompts: any non-empty, non-whitespace-only string (derive_contract
# raises ValueError on empty/whitespace-only prompts, so batches must avoid
# those to test the distinct-contracts property itself).
_batch_system_prompt_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")

# A list of (agent_name, system_prompt) pairs with pairwise-distinct agent
# names, built by drawing a set of unique names first and then pairing each
# with an independently generated prompt.
_distinct_peer_configs = st.lists(
    _batch_agent_name_strategy, min_size=1, max_size=8, unique=True
).flatmap(
    lambda names: st.builds(
        lambda prompts: list(zip(names, prompts)),
        st.lists(_batch_system_prompt_strategy, min_size=len(names), max_size=len(names)),
    )
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(peer_configs=_distinct_peer_configs)
def test_derive_all_produces_one_distinct_contract_per_agent(peer_configs):
    """For any list of (agent_name, system_prompt) pairs with distinct
    agent names, derive_all produces a list of contracts of the same
    length, with agent_name fields equal to the inputs in order, and no
    two output contracts sharing an agent_name."""
    with patch.object(contract_deriver, "_llm_extract", side_effect=_fake_llm_extract_for_batch):
        contracts = contract_deriver.derive_all(peer_configs)

    # Same length as the input batch.
    assert len(contracts) == len(peer_configs)

    # agent_name fields equal the inputs, in the same order.
    expected_names = [name for name, _ in peer_configs]
    actual_names = [c.agent_name for c in contracts]
    assert actual_names == expected_names

    # No two output contracts share an agent_name (follows from the inputs
    # having distinct names and identity being preserved, but assert it
    # directly as the property states it).
    assert len(set(actual_names)) == len(contracts)


# ---------------------------------------------------------------------------
# Property 1: identity preservation + never crashes (forced heuristic path)
# ---------------------------------------------------------------------------

# agent_name: reasonable text, no surrogate characters (unpaired surrogates
# aren't valid UTF-8 and are irrelevant to this property, which is about
# ordinary identity preservation, not UTF-8 edge cases). Allowed to be
# empty/whitespace-only -- agent_name has no emptiness precondition in
# derive_contract, unlike system_prompt.
_property1_agent_name_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=50,
)

# system_prompt: non-empty after stripping, since derive_contract explicitly
# raises ValueError for an empty/whitespace-only prompt (that behavior is
# out of scope for this property and is covered separately by
# test_derive_contract_raises_on_empty_prompt below).
_property1_system_prompt_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=500,
).filter(lambda s: s.strip() != "")


def _always_fail_llm_extract(agent_name, system_prompt):
    raise RuntimeError("forced failure: exercising the heuristic fallback path")


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(agent_name=_property1_agent_name_strategy, system_prompt=_property1_system_prompt_strategy)
def test_derive_contract_preserves_identity_and_never_crashes(monkeypatch, agent_name, system_prompt):
    """For any agent_name and any non-empty system_prompt, derive_contract
    never raises, and the resulting AgentContract's agent_name/source_prompt
    equal the inputs exactly, with a non-empty role.

    _llm_extract is monkeypatched to always raise so every example runs
    through the deterministic heuristic fallback (see module docstring).
    ``monkeypatch`` is function-scoped and reused (not reset) across
    Hypothesis examples within a single test call, but that is safe here:
    every example re-applies the same idempotent setattr before invoking
    derive_contract, and the patch is undone once at the end of the test
    function regardless of how many examples ran.
    """
    monkeypatch.setattr(contract_deriver, "_llm_extract", _always_fail_llm_extract)

    contract = contract_deriver.derive_contract(agent_name, system_prompt)

    assert isinstance(contract, AgentContract)
    assert contract.agent_name == agent_name
    assert contract.source_prompt == system_prompt
    assert contract.role.strip() != ""


def test_derive_contract_raises_on_empty_prompt():
    """Explicit pin for the documented out-of-scope precondition: an empty
    or whitespace-only system_prompt raises ValueError before any
    extraction (LLM or heuristic) is attempted."""
    import pytest

    with pytest.raises(ValueError):
        contract_deriver.derive_contract("some_agent", "")

    with pytest.raises(ValueError):
        contract_deriver.derive_contract("some_agent", "   \n\t  ")


# ---------------------------------------------------------------------------
# Example/unit tests (non-property-based)
#
# Validates: Requirements 1.3, 1.4
#
# These two tests pin down concrete, documented examples of derive_contract's
# behavior -- readable, fixed inputs/outputs rather than the property tests'
# randomly-generated tool pairs and prompts above. Both use the file's
# established _patch_llm_extract/monkeypatch helper to avoid a live Bedrock
# call, consistent with every other test in this module (no live call is
# made here).
# ---------------------------------------------------------------------------

def test_derive_contract_with_explicit_tool_definitions(monkeypatch):
    """Concrete example for Requirement 1.3: a Peer_Agent_Config with
    explicit tool definitions produces an AgentContract whose tools list
    matches exactly, pinning down the exact schema/shape for a documented,
    readable example (a "search" tool and a "calculator" tool)."""
    _patch_llm_extract(monkeypatch)

    search_tool = ToolDefinition(
        name="search",
        purpose="Search the web for current market data and news articles.",
    )
    calculator_tool = ToolDefinition(
        name="calculator",
        purpose="Perform arithmetic calculations on market figures.",
    )

    contract = derive_contract(
        "market_trends",
        "You are a market analyst.",
        tools=[search_tool, calculator_tool],
    )

    assert contract.tools == [search_tool, calculator_tool]
    assert [t.to_dict() for t in contract.tools] == [
        {"name": "search", "purpose": "Search the web for current market data and news articles."},
        {"name": "calculator", "purpose": "Perform arithmetic calculations on market figures."},
    ]


def test_derive_contract_with_no_tools_on_real_market_trends_prompt(monkeypatch):
    """Concrete example for Requirement 1.4: deriving a contract for a
    realistic peer-agent-style call -- the sibling module's real
    "market_trends" system prompt -- with tools omitted entirely produces an
    AgentContract with an empty tools list and a populated role/
    responsibilities/constraints.

    Uses the file's established _patch_llm_extract/monkeypatch helper (no
    live Bedrock call), consistent with the rest of this test module, so the
    role/responsibilities/constraints asserted below are the fixed stand-in
    values (_FIXED_ROLE / _FIXED_RESPONSIBILITIES / _FIXED_CONSTRAINTS), not
    a real LLM extraction of the prompt's content. What this test pins down
    is derive_contract's behavior on a real, realistic prompt string with no
    tools -- not the quality of the extraction itself (that is out of scope
    for this test, same as the property tests above).
    """
    _patch_llm_extract(monkeypatch)

    market_trends_prompt = (
        "You are a Market Trends Analyst specialising in the restaurant and food service industry. "
        "Analyse market trends, competitive landscape, market size, growth rates, and emerging "
        "opportunities for the given topic. Be specific with numbers, brand names, and market "
        "segments. Keep your analysis concise (2-3 paragraphs)."
    )

    contract = derive_contract("market_trends", market_trends_prompt)

    assert contract.tools == []
    assert contract.agent_name == "market_trends"
    assert contract.source_prompt == market_trends_prompt
    assert contract.role.strip() != ""
    assert len(contract.responsibilities) > 0
    assert all(isinstance(r, str) and r.strip() for r in contract.responsibilities)
    assert all(isinstance(c, str) and c.strip() for c in contract.constraints)
