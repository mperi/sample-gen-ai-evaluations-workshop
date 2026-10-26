"""Contract_Deriver for Agent_Health_Evaluation_Demo.

Introspects a Peer_Agent_Config's system prompt (and any associated tool
definitions) and produces an ``AgentContract`` stating the agent's role,
responsibilities, and constraints, without requiring a separately authored
requirements document.

``derive_contract`` calls Bedrock (via ``METRIC_PROPOSER_MODEL_ID``) using the
same ``invoke_model`` + markdown-fence-stripped-JSON + ``ThrottlingException``
backoff/retry pattern already used by the sibling module's
``metrics_collector.py`` (``LLMJudge._call``), for consistency with that
module's conventions. If the LLM call fails for any reason (throttling,
malformed response, missing credentials, etc.) this module falls back to a
deterministic, non-LLM heuristic splitter so a contract is still produced —
the Contract_Deriver should never produce an empty contract purely because of
a transient Bedrock error (see design.md's Error Handling section).

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3

from model_config import AWS_REGION, METRIC_PROPOSER_MODEL_ID, PEER_CONFIGS
from models import AgentContract, ToolDefinition
from persistence import write_json

logger = logging.getLogger("contract_deriver")


# ---------------------------------------------------------------------------
# Bedrock call (mirrors metrics_collector.LLMJudge._call's pattern)
# ---------------------------------------------------------------------------

def _invoke_model(prompt: str, region: str = AWS_REGION,
                   model_id: str = METRIC_PROPOSER_MODEL_ID,
                   max_retries: int = 5, max_tokens: int = 1024) -> dict:
    """Call Bedrock's ``invoke_model`` and parse a JSON response, reusing the
    sibling module's markdown-fence-stripping + ``ThrottlingException``
    backoff/retry pattern (see ``metrics_collector.LLMJudge._call``).

    Shared verbatim by ``contract_deriver.derive_contract`` (single
    role/responsibilities/constraints object, fits comfortably in the
    1024-token default) and ``metric_proposer.propose_metrics`` (a JSON
    *list* of several metric proposals, each with its own name/definition/
    rubric/traceability, which needs a larger ``max_tokens`` budget to avoid
    truncating mid-response — see that caller for the larger value it
    passes).

    Raises on failure (after exhausting retries, or on a malformed/empty
    response) rather than swallowing the error, so callers (``derive_contract``)
    can decide to fall back to the deterministic heuristic splitter.
    """
    client = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    })

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.invoke_model(
                modelId=model_id, body=body,
                contentType="application/json", accept="application/json")
            text = json.loads(resp["body"].read())["content"][0]["text"]
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except client.exceptions.ThrottlingException as e:
            last_error = e
            wait = 2 ** attempt + 1
            logger.warning(f"Throttled (attempt {attempt + 1}/{max_retries}), waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            last_error = e
            if "ServiceUnavailable" in str(type(e).__name__) or "Too many connections" in str(e):
                wait = 2 ** attempt + 1
                logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                # Not a retryable error (e.g. malformed JSON, missing key,
                # credentials error) — raise immediately so the caller can
                # fall back rather than burning through retries.
                raise

    raise RuntimeError(
        f"Bedrock invoke_model failed after {max_retries} retries: {last_error}"
    )


_CONTRACT_EXTRACTION_PROMPT_TEMPLATE = """You are analyzing the system prompt of an AI agent named "{agent_name}" to \
extract a structured contract describing what it is supposed to do.

System prompt:
\"\"\"{system_prompt}\"\"\"

Extract:
- "role": a single sentence stating the agent's overall role/purpose.
- "responsibilities": a list of short strings, each one specific thing the \
agent is responsible for doing.
- "constraints": a list of short strings, each one constraint or rule the \
agent must follow (if none are stated, return an empty list).

Respond with JSON only, no commentary, in this exact shape:
{{"role": "<one sentence>", "responsibilities": ["...", "..."], "constraints": ["...", "..."]}}"""


def _llm_extract(agent_name: str, system_prompt: str) -> Tuple[str, List[str], List[str]]:
    """Call Bedrock to extract (role, responsibilities, constraints) from a
    system prompt. Raises on any failure; callers should catch and fall back.
    """
    prompt = _CONTRACT_EXTRACTION_PROMPT_TEMPLATE.format(
        agent_name=agent_name, system_prompt=system_prompt)
    parsed = _invoke_model(prompt)

    role = parsed["role"]
    responsibilities = list(parsed["responsibilities"])
    constraints = list(parsed["constraints"])

    if not isinstance(role, str) or not role.strip():
        raise ValueError("LLM extraction returned an empty role")

    return role.strip(), responsibilities, constraints


# ---------------------------------------------------------------------------
# Deterministic heuristic fallback (no LLM call)
# ---------------------------------------------------------------------------

_ROLE_PATTERN = re.compile(r"\bYou are an?\b.*?[.\n]", re.IGNORECASE | re.DOTALL)

# Cue phrases that mark a sentence as a constraint rather than a plain
# responsibility.
_CONSTRAINT_CUES = (
    "must", "should not", "shall not", "never", "always", "only",
    "do not", "don't", "constraint", "required to", "not allowed",
)


def _split_into_sentences(text: str) -> List[str]:
    """Very small sentence splitter: splits on '.', '!', '?' followed by
    whitespace, and on newlines, trimming and dropping empties."""
    # Normalize bullet markers ("- ", "* ") to sentence boundaries too.
    normalized = re.sub(r"\n\s*[-*]\s*", "\n", text)
    pieces = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [p.strip() for p in pieces if p.strip()]


def _heuristic_extract(agent_name: str, system_prompt: str) -> Tuple[str, List[str], List[str]]:
    """Deterministic, non-LLM fallback: derive (role, responsibilities,
    constraints) from the raw prompt text using simple textual cues.

    - ``role``: the "You are a..."/"You are an..." sentence if present,
      otherwise the prompt's first sentence.
    - ``constraints``: sentences (other than the role sentence) containing a
      constraint cue phrase (e.g. "must", "should not", "constraints:").
    - ``responsibilities``: all remaining non-role sentences.
    """
    sentences = _split_into_sentences(system_prompt)
    if not sentences:
        # system_prompt is non-empty (checked by caller) but has no sentence
        # boundaries at all (e.g. a single word) — treat the whole text as
        # both the role and the only sentence to classify.
        sentences = [system_prompt.strip()]

    role_match = _ROLE_PATTERN.search(system_prompt)
    if role_match:
        role = role_match.group(0).strip().rstrip(".") + "."
    else:
        role = sentences[0]

    responsibilities: List[str] = []
    constraints: List[str] = []
    for sentence in sentences:
        # Skip the sentence we already used as the role statement.
        if sentence == role or sentence == role.rstrip("."):
            continue
        lowered = sentence.lower()
        if any(cue in lowered for cue in _CONSTRAINT_CUES):
            constraints.append(sentence)
        else:
            responsibilities.append(sentence)

    if not responsibilities and not constraints:
        # The prompt was a single sentence that became the role — there is
        # nothing left to classify. Keep responsibilities/constraints empty
        # rather than fabricating content.
        pass

    return role, responsibilities, constraints


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def derive_contract(agent_name: str, system_prompt: str,
                     tools: Optional[List[ToolDefinition]] = None) -> AgentContract:
    """Derive an ``AgentContract`` from ``agent_name``'s system prompt.

    Tries the Bedrock-based extraction first; falls back to a deterministic
    heuristic splitter if the LLM call raises for any reason (throttling,
    malformed response, missing credentials, etc.), so a contract is always
    produced for a well-formed, non-empty prompt.

    Args:
        agent_name: the Peer_Agent_Config's name, e.g. "market_trends".
        system_prompt: the Peer_Agent_Config's system prompt text.
        tools: tool definitions to copy verbatim into the contract's
            ``tools`` field. Defaults to an empty list if ``None``.

    Returns:
        The derived ``AgentContract``.

    Raises:
        ValueError: if ``system_prompt`` is empty or whitespace-only. This
            check happens before any LLM call or fallback is attempted.
    """
    if not system_prompt or not system_prompt.strip():
        raise ValueError("system_prompt must not be empty or whitespace-only")

    try:
        role, responsibilities, constraints = _llm_extract(agent_name, system_prompt)
    except Exception as e:
        logger.warning(
            f"LLM contract extraction failed for agent '{agent_name}' "
            f"({type(e).__name__}: {e}); falling back to heuristic splitter."
        )
        role, responsibilities, constraints = _heuristic_extract(agent_name, system_prompt)

    return AgentContract(
        agent_name=agent_name,
        role=role,
        responsibilities=responsibilities,
        constraints=constraints,
        tools=list(tools) if tools else [],
        source_prompt=system_prompt,
    )


def derive_all(peer_configs: List[Tuple[str, str]],
                tools_by_agent: Optional[Dict[str, List[ToolDefinition]]] = None
                ) -> List[AgentContract]:
    """Batch-derive one ``AgentContract`` per ``(agent_name, system_prompt)``
    pair in ``peer_configs``, in order.

    Args:
        peer_configs: a list of ``(agent_name, system_prompt)`` tuples, e.g.
            the sibling module's ``PEER_CONFIGS``.
        tools_by_agent: optional mapping of agent name to that agent's tool
            definitions. An agent missing from this mapping (or ``None``
            overall) is derived with no tools.

    Returns:
        A list of ``AgentContract``s, one per input pair, in the same order.
    """
    tools_by_agent = tools_by_agent or {}
    return [
        derive_contract(agent_name, system_prompt, tools_by_agent.get(agent_name))
        for agent_name, system_prompt in peer_configs
    ]


def load_peer_configs_readonly() -> List[Tuple[str, str]]:
    """Return the sibling module's ``PEER_CONFIGS``, as already imported
    read-only via this package's own ``model_config.py`` re-export.

    Never writes to the sibling's ``model_config.py`` or any other sibling
    file; this is a plain re-export of an already-imported value.
    """
    return list(PEER_CONFIGS)


def derive_and_persist_all(
        peer_configs: Optional[List[Tuple[str, str]]] = None,
        tools_by_agent: Optional[Dict[str, List[ToolDefinition]]] = None
        ) -> List[AgentContract]:
    """Convenience wrapper: derive contracts for ``peer_configs`` (defaulting
    to ``load_peer_configs_readonly()``) and persist each one to
    ``output/contracts/<agent_name>.json`` via ``persistence.write_json``.

    Returns the list of derived contracts (same order as the input).
    """
    if peer_configs is None:
        peer_configs = load_peer_configs_readonly()

    contracts = derive_all(peer_configs, tools_by_agent)
    for contract in contracts:
        write_json(f"contracts/{contract.agent_name}.json", contract.to_dict())
    return contracts


if __name__ == "__main__":
    # Sanity check: derive contracts for the three real peer agents and
    # persist them to output/contracts/. Bedrock access requires live AWS
    # credentials; if those aren't available in this execution environment,
    # derive_contract's internal fallback kicks in automatically and this
    # check still produces and verifies three distinct contracts — it just
    # reports which path (LLM vs. heuristic fallback) was actually taken.
    logging.basicConfig(level=logging.INFO)

    peer_configs = load_peer_configs_readonly()
    print(f"Loaded {len(peer_configs)} peer configs: {[name for name, _ in peer_configs]}")

    contracts = derive_and_persist_all(peer_configs)

    assert len(contracts) == len(peer_configs) == 3, (
        f"expected 3 distinct contracts, got {len(contracts)}"
    )
    agent_names = [c.agent_name for c in contracts]
    assert agent_names == [name for name, _ in peer_configs], agent_names
    assert len(set(agent_names)) == 3, "expected 3 distinct agent names"

    for contract in contracts:
        assert contract.role.strip(), f"{contract.agent_name} has an empty role"
        persisted_path = f"output/contracts/{contract.agent_name}.json"
        print(f"\n=== {contract.agent_name} ===")
        print(f"role: {contract.role}")
        print(f"responsibilities ({len(contract.responsibilities)}): {contract.responsibilities}")
        print(f"constraints ({len(contract.constraints)}): {contract.constraints}")
        print(f"persisted to: {persisted_path}")

    print("\nAll contract derivation sanity checks passed "
          "(three distinct contracts produced and persisted).")
