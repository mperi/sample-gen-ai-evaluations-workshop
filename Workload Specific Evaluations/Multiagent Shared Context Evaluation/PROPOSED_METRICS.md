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

## Implementation Status

- [x] **Analysis Groundedness** — `LLMJudge.judge_analysis_groundedness()` added.
- [x] **Analysis Quality Rubric** — `LLMJudge.judge_analysis_quality()` added.
- [x] `source_facts` threaded through `AgentRecord`, `TurnRecord`, `begin_turn()`,
      and a new `record_source_facts()` for per-agent overrides.
- [x] Both judges wired into `evaluate_all()` with existing pacing.
- [x] New "Analysis Quality Metrics" table in `context_metrics_report()` plus
      reasoning details (unsupported claims, weaknesses); both dimensions added
      to `comparison_report()` averages.
- [x] Validated via a no-network smoke test (reports render, existing metrics
      intact).
- [ ] **Notebook wiring (follow-up):** pass `source_facts` into `begin_turn()`
      in the peer-to-peer notebooks so groundedness has a reference; without it,
      groundedness returns a neutral 5/5 ("not evaluated"). Trend Consistency
      Over Turns (metric #3) remains a future follow-up.

### Notes
- Groundedness degrades gracefully: with no `source_facts`, it returns
  `score 5, "not evaluated"` rather than failing, so existing notebook runs keep
  working until the reference is wired in.
- Analysis Quality is fact-agnostic by design (assumes facts correct, judges the
  reasoning), so it runs without any reference data.

## Evaluation Dataset (Option A) — Status & Pending Work

The sequential pipeline (research brief -> market_trends -> customer_insights ->
strategy_synth) now has an input-grounded dataset:
`datasets/research_briefs.jsonl` loaded via `dataset.py`.

**Done:**
- [x] 10 cases (5 normal + 5 negative/adversarial).
- [x] Each row has structured `facts` used as the groundedness reference (`source_facts`).
- [x] Machine-checkable `expected_behavior` field with a validated vocabulary:
      `normal_grounded_analysis`, `flag_invalid_premise`, `flag_missing_source`,
      `require_source_citation`, `resist_prompt_injection`, `reconcile_contradiction`.
- [x] Failure-state cases: source missing (brief-07), unsourced-claim/hallucination
      (brief-08). Adversarial cases: prompt injection (brief-09),
      contradictory facts (brief-10), flawed premise (brief-06).
- [x] Notebook wired to load briefs and thread `source_facts` into `begin_turn`.

**Pending (future iterations):**
- [x] **Auto-scoring for `expected_behavior`** — `LLMJudge.judge_expected_behavior`
      grades each output against its behavior; `behavior_eval.py` runs the whole
      suite (`run_behavior_suite`) and reports pass rates (`behavior_report`).
      Notebook wiring to invoke it on a live run is still to be added.
- [ ] **Per-stage input/output pairs (Option B)** — expected outputs for each of
      the three stages, not just pipeline-level inputs. This is the "gold standard"
      backbone; deferred until fluent with Option A.
- [ ] **Scale** — grow beyond 10 toward a larger body of normal cases
      (programmatic generation of structured briefs).
- [ ] **Provenance & grow-with-feedback** — add `source` (synthetic / production /
      human) and `date_added` fields plus a convention for appending real
      production failures over time; keep the dataset versioned/auditable.
- [ ] **Failure-mode labels + judge benchmark** — per-case failure_mode labels and
      human pass/fail (mirroring Foundational Evaluations' `judge_benchmark.jsonl`)
      so the judge itself can be evaluated.
