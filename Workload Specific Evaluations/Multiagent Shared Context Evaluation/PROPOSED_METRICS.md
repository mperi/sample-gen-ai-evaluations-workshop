# Proposed Metrics: Market Trends Analyst Agent

## Context

The `market_trends` agent (labelled "market_trends_analyst" in the notebook
architecture diagrams) is currently evaluated by `MetricsCollector` in
`metrics_collector.py`. That module is, by design, a **shared-context /
coordination evaluation**: every judge scores how context *flows and is handled*
across agents, not the *domain quality* of what each agent produces.

Mapping the current metrics to the questions we care about:

| Question | Covered by | Notes |
|----------|------------|-------|
| 1. Does it take upstream inputs (research brief / handover contract) accurately? | `handoff_completeness` + `context_freshness` | Handoff completeness checks the incoming query carried all needed facts; freshness checks the agent isn't working off stale memory. |
| 2. Does it pass outputs downstream accurately? | `write_accuracy` (memory write accuracy) | Checks the written response is accurate **relative to its input** — catches fabrication/contradiction, not domain correctness. |
| 3. Does it leverage shared context well (usage, efficiency)? | `context_utilization`, `redundant_context`, `ccr` | Usage, redundancy, and compression respectively. |
| **4. Is it analyzing trends accurately?** | **Nothing today** | **Gap** — no groundedness or task-quality judge exists. |

An agent could faithfully restate a wrong premise from the handoff and still
score 5 on `write_accuracy`, because accuracy is judged only against the input,
never against source data or an analytical rubric.

## Proposed New Metrics / Capabilities

### 1. Analysis Groundedness (reference-based)
- **What it measures:** Whether the specific claims in the analysis (market
  size, growth rate, competitor set, segment definitions) are supported by the
  provided source data / handoff facts, and whether any numbers were fabricated
  or altered.
- **Rationale:** This is the most direct answer to "is it analyzing trends
  accurately?" `write_accuracy` only checks internal consistency with the input;
  groundedness checks claims against a reference/source of truth so an agent
  can't launder a wrong input into a confident output.
- **How:** Add `LLMJudge.judge_analysis_groundedness(source_facts, response,
  agent_name)` returning `{"score": 1-5, "reasoning": ..., "unsupported_claims": [...]}`.
  Requires a `source_facts` / reference input threaded through the turn record.

### 2. Analysis Quality Rubric (task quality)
- **What it measures:** Analytical soundness on domain dimensions — evidence
  support, correct use of the numbers, depth/insight of trend calls, absence of
  unsupported logical leaps.
- **Rationale:** Groundedness catches wrong facts; it does not reward *good*
  analysis. A response can be fully grounded yet shallow. A rubric judge scores
  whether the trend analysis is actually useful, which is what "good job" means
  for this agent's core task.
- **How:** Add `LLMJudge.judge_analysis_quality(handoff, response, agent_name)`
  with a 1-5 rubric returning `{"score", "reasoning", "weaknesses": [...]}`.

### 3. (Optional) Trend Consistency Over Turns
- **What it measures:** When the brief is revised across turns (e.g. US →
  North America expansion in the sequential notebook), do the analyst's
  conclusions update coherently rather than contradicting the prior turn without
  cause?
- **Rationale:** The existing `state_consistency` judge is *cross-agent within a
  turn*. It does not check a single agent's coherence *across* turns as the
  brief evolves — a distinct failure mode for an analyst that revises work.
- **How:** A per-agent, cross-turn judge comparing the agent's response at turn
  N against N-1 given the delta in the brief. Lower priority; propose as a
  follow-up.

## Implementation Plan

1. **`metrics_collector.py`**
   - Add judge methods #1 and #2 (and optionally #3) to `LLMJudge`, matching the
     existing prompt/return-shape style.
   - Extend `AgentRecord` (and/or `TurnRecord`) to carry a `source_facts` /
     reference field needed by the groundedness judge.
   - Call the new judges in `MetricsCollector.evaluate_all()` (with the same
     `time.sleep` pacing used for existing judges).
   - Surface new columns/sections in `context_metrics_report()` and include the
     new dimensions in `comparison_report()`.
2. **Notebook (`04-peer-to-peer-sequential.ipynb` and/or peers)**
   - Provide the `source_facts` reference when beginning a turn so the
     groundedness judge has something to compare against.
   - Re-run the metrics cells to confirm the new columns render.
3. **Validation**
   - Run against an existing session; confirm new scores populate and reports
     render without breaking existing metrics.

## PR / Change Tracking Plan

- Pre-existing local modifications to `04-peer-to-peer-sequential.ipynb` and
  `model_config.py` are unrelated to this work — keep them out of this PR
  (separate branch / don't stage them).
- Create a dedicated feature branch, e.g. `feat/market-trends-analysis-quality`.
- Commit this proposal first, then the implementation, so the rationale is
  captured in history.
- Open the PR with: the gap being closed (no trend-accuracy metric), the new
  metrics and their rationale (this doc), and validation notes from the re-run.
