"""Evaluator_Runner for Agent_Health_Evaluation_Demo (Phase 3).

``run_evaluation`` iterates the full Cartesian product of ``records x
metrics`` (Req 4.1, 4.2, 4.3). For each ``(record, metric)`` pair it looks up
``feedback_store.latest_label_for_record(record.record_id)`` and passes that
label's ``label_text`` (or ``None`` if no label exists) to ``scorer`` as
``reference_label`` (Req 4.6). Each call is wrapped in a per-pair
``try/except Exception``: success appends a ``MetricResult``, failure
appends a ``MetricFailure`` (record id, metric name, error message) and the
loop continues onto the next pair rather than aborting the run (Req 4.4).
When the run completes, both lists are persisted via ``persistence.py`` to
``output/results/metric_results.jsonl`` and
``output/results/metric_results.failures.jsonl`` (Req 4.5).

Scorer contract (decided in task 9.1, conformed to by ``agentcore_scorer``
below):
    ``scorer(record, metric, reference_label)`` MUST return a 2-tuple
    ``(score, rationale)`` -- ``score`` a ``float`` and ``rationale`` a
    ``str`` -- on success, or raise any ``Exception`` on failure. A plain
    tuple return is the simplest possible contract: it needs no extra type
    (design.md's ``ScoreOutcome`` does not exist as a dataclass), and it is
    trivially satisfied by both a hand-written fake scorer (as in this
    module's sanity check below) and the real ``agentcore_scorer``.
    ``_coerce_score_outcome`` additionally accepts an object/dict shaped
    result (attributes or ``"score"``/``"rationale"`` keys) so a scorer is
    free to return a richer ``ScoreOutcome``-like value instead of a bare
    tuple without requiring a change to this loop.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional, Tuple

import feedback_store
from models import CandidateHealthMetric, MetricFailure, MetricResult, ReviewRecord
from persistence import append_jsonl

logger = logging.getLogger(__name__)

# Where Evaluator_Run output is persisted, relative to this package's own
# output/ directory (resolved by persistence.append_jsonl via
# resolve_output_path). Req 4.5.
METRIC_RESULTS_PATH = "results/metric_results.jsonl"
METRIC_FAILURES_PATH = "results/metric_results.failures.jsonl"

# Scorer contract: a callable taking (record, metric, reference_label) and
# returning either a (score, rationale) tuple or a ScoreOutcome-like
# object/dict (see _coerce_score_outcome), or raising on failure.
Scorer = Callable[[ReviewRecord, CandidateHealthMetric, Optional[str]], Any]


def _coerce_score_outcome(raw: Any) -> Tuple[float, str]:
    """Coerce a scorer's raw return value into a ``(score, rationale)``
    pair.

    Accepts, in order of preference:
      - a ``(score, rationale)`` tuple/list of length 2 (the primary,
        simplest contract documented at the top of this module);
      - an object exposing ``.score`` and ``.rationale`` attributes (e.g. a
        future ``ScoreOutcome`` dataclass, if one is ever introduced);
      - a dict exposing ``"score"`` and ``"rationale"`` keys.

    Args:
        raw: whatever a ``scorer`` callable returned on success.

    Returns:
        A ``(score, rationale)`` tuple with ``score`` coerced to ``float``
        and ``rationale`` coerced to ``str``.

    Raises:
        ValueError: if ``raw`` does not match any of the accepted shapes.
    """
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        score, rationale = raw
        return float(score), str(rationale)

    if hasattr(raw, "score") and hasattr(raw, "rationale"):
        return float(raw.score), str(raw.rationale)

    if isinstance(raw, dict) and "score" in raw and "rationale" in raw:
        return float(raw["score"]), str(raw["rationale"])

    raise ValueError(
        f"Scorer returned an unrecognized result shape (expected a "
        f"(score, rationale) tuple, or an object/dict with 'score' and "
        f"'rationale'): {raw!r}"
    )


class AgentCoreConfigurationError(RuntimeError):
    """Raised when the AgentCore evaluation client cannot be constructed at
    all (e.g. the ``bedrock-agentcore`` package is not installed, or AWS
    credentials cannot be resolved).

    This is intentionally NOT a per-pair ``MetricFailure``: Requirement 4.7
    calls for a "no mocked fallback" posture, so a configuration problem
    that would prevent *every* ``agentcore_scorer`` call from working must
    abort the whole Evaluator_Run with a clear, loud error instead of being
    caught by ``run_evaluation``'s per-pair ``try/except`` and silently
    recorded as hundreds of individual ``MetricFailure`` entries. Raising
    this from module import time / client construction time (before the
    per-pair loop ever starts calling the scorer) is what achieves that.
    """


# Constructed lazily (module import must not require AWS credentials or the
# bedrock-agentcore package to be importable, so unit tests that only
# exercise run_evaluation's orchestration loop with a fake scorer are
# unaffected) and cached for the process lifetime, since EvaluationClient
# construction is a boto3-session-creation-level cost, not a per-call one.
_agentcore_client: Any = None


def _get_agentcore_client() -> Any:
    """Construct (once) and return the ``bedrock_agentcore.evaluation``
    ``EvaluationClient`` used by ``agentcore_scorer``.

    Raises:
        AgentCoreConfigurationError: if the ``bedrock-agentcore`` package
            cannot be imported, or if constructing the client fails for any
            reason (e.g. no AWS credentials/region resolvable). This is
            deliberately not caught anywhere else in this module -- see
            ``AgentCoreConfigurationError``'s docstring -- so it propagates
            out of ``agentcore_scorer`` and aborts the whole run rather
            than being converted into a ``MetricFailure`` (Req 4.7).
    """
    global _agentcore_client
    if _agentcore_client is not None:
        return _agentcore_client

    try:
        from bedrock_agentcore.evaluation import EvaluationClient
    except ImportError as exc:
        raise AgentCoreConfigurationError(
            "Cannot construct the AgentCore evaluation client: the "
            "'bedrock-agentcore' package is not installed. Install it "
            "(see requirements.txt) before running an Evaluator_Run with "
            "the default agentcore_scorer."
        ) from exc

    import model_config

    try:
        _agentcore_client = EvaluationClient(region_name=model_config.AWS_REGION)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any
        # construction failure (missing/invalid AWS credentials, no
        # resolvable region, etc.) is a configuration problem, not a
        # per-pair scoring problem, so it is wrapped and re-raised rather
        # than swallowed.
        raise AgentCoreConfigurationError(
            f"Cannot construct the AgentCore evaluation client "
            f"(region={model_config.AWS_REGION!r}): {exc}"
        ) from exc

    return _agentcore_client


# Default evaluator used by agentcore_scorer. "Builtin.Correctness" is the
# most general-purpose built-in evaluator available for scoring a single
# agent response against expected/ground-truth content, which is the
# closest existing fit for an arbitrary CandidateHealthMetric rubric (see
# agentcore_scorer's docstring for the full adaptation rationale).
DEFAULT_EVALUATOR_ID = "Builtin.Correctness"

# AgentCore's Evaluate API score ranges "typically" run 0-1 but are
# evaluator-dependent (AWS docs: "Range depends on the evaluator (typically
# 0-1)"). This demo's MetricResult.score is documented as normalized
# 0.0-1.0 (see models.py), so agentcore_scorer clamps defensively in case a
# future/alternate evaluator ever returns a value outside that nominal
# range (e.g. a categorical-only evaluator surfacing no numeric value at
# all, handled separately below).
_SCORE_MIN = 0.0
_SCORE_MAX = 1.0


def _normalize_score(value: Optional[float], label: Optional[str]) -> float:
    """Normalize an AgentCore evaluation result's ``value``/``label`` into
    this demo's documented 0.0-1.0 ``MetricResult.score`` scale.

    AgentCore's built-in evaluators "typically" already return a 0-1
    ``value`` (per AWS's own documentation: "Range depends on the
    evaluator (typically 0-1)"), so the common case is a straight
    pass-through, clamped defensively to [0.0, 1.0] in case a differently
    configured/custom evaluator ever returns a value outside that nominal
    range.

    If ``value`` is absent (the API marks it optional -- some evaluators
    are categorical-label-only), this falls back to a small fixed
    label->score lookup for the handful of categorical labels AgentCore's
    own docs give as examples (``"Pass"``/``"Fail"``,
    ``"Excellent"``/``"Good"``/``"Poor"``). An unrecognized label with no
    numeric value raises, since silently guessing a score for an unknown
    categorical scale would misrepresent what the evaluator actually said
    (and such a raise is caught by ``run_evaluation``'s per-pair
    ``try/except`` and surfaced as a ``MetricFailure``, not swallowed).

    Args:
        value: the ``EvaluationResultContent.value`` field, or ``None``.
        label: the ``EvaluationResultContent.label`` field, or ``None``.

    Returns:
        A float in ``[0.0, 1.0]``.

    Raises:
        ValueError: if both ``value`` and ``label`` are absent, or if
            ``label`` is present but not one of the small set of
            recognized categorical labels and ``value`` is absent.
    """
    if value is not None:
        return max(_SCORE_MIN, min(_SCORE_MAX, float(value)))

    _CATEGORICAL_SCORES = {
        "pass": 1.0,
        "fail": 0.0,
        "excellent": 1.0,
        "good": 0.75,
        "fair": 0.5,
        "poor": 0.0,
    }
    if label is not None and label.strip().lower() in _CATEGORICAL_SCORES:
        return _CATEGORICAL_SCORES[label.strip().lower()]

    raise ValueError(
        f"AgentCore evaluation result has neither a usable numeric "
        f"'value' nor a recognized categorical 'label' to normalize into "
        f"a 0.0-1.0 score (value={value!r}, label={label!r})"
    )


def _build_synthetic_llm_span(
    record: ReviewRecord, metric: CandidateHealthMetric
) -> List[dict]:
    """Build one minimal, synthetic OpenTelemetry-shaped span wrapping a
    ``ReviewRecord``'s brief + response text, standing in for a real
    captured agent trace, using AgentCore Evaluations' documented *generic
    framework support* instrumentation convention.

    ADAPTATION DECISION (documented here since it is the crux of this
    function, and of ``agentcore_scorer`` as a whole):

    AgentCore's ``Evaluate`` API's ``evaluationInput.sessionSpans`` expects
    actual OpenTelemetry/ADOT spans captured from a real agent invocation.
    This demo's ``ReviewRecord`` is plain review text sourced from the
    sibling module's ``review/review_cases.jsonl`` -- there is no captured
    OTel trace for it at all, because the sibling module never instrumented
    its agents with OpenTelemetry. There is therefore no faithful way to
    turn "a brief + a response string" into "a multi-span agent execution
    trace": no tool-call spans exist, no multi-turn structure exists, only
    one brief and one piece of generated text exist.

    Given that mismatch, the most faithful reasonable adapter is to
    construct a single synthetic span representing the one agent turn that
    *does* exist -- the brief in, the response out -- shaped exactly as
    AgentCore's documented *generic framework support*
    (``supported-frameworks-generic.html``) says a custom/self-instrumented
    agent must shape its spans to be recognized:

      1. A ``scope.name`` beginning with ``opentelemetry.instrumentation.*``
         (the service classifies a span by its scope name; this prefix
         selects the OpenTelemetry GenAI reading path). This adapter uses an
         honestly-named synthetic scope,
         ``opentelemetry.instrumentation.agent-health-eval-synthetic``,
         rather than impersonating a real framework's instrumentation
         library -- the span genuinely *is* our own custom instrumentation
         of a synthetic one-turn trace.
      2. The identifying attribute ``gen_ai.operation.name`` = ``invoke_agent``
         so the service classifies this as an invoke-agent (top-level agent
         turn) span. The doc: "Set an identifying attribute on every span
         ... The service skips a span that carries no recognized
         identifying attribute."
      3. The content attributes the OpenTelemetry reading path extracts for
         an invoke-agent span, with content kept on the span itself (the
         "unified telemetry" delivery mode, where content "stays on the span
         as attributes"): ``gen_ai.task.input`` (the user prompt / brief)
         and ``gen_ai.task.output`` (the agent response).
      4. As a belt-and-suspenders fallback, the two scope-independent
         attributes the doc says "the service reads ... from any scope" and
         that "cover the agent prompt and response only":
         ``agentcore.invocation.user_prompt`` and
         ``agentcore.invocation.agent_response``. These guarantee the
         top-level agent turn is readable even if the primary
         ``gen_ai.task.*`` attributes are not picked up.

    LIMITATION: a built-in evaluator like ``Builtin.Correctness`` or
    ``Builtin.Helpfulness`` scores this single synthetic invoke-agent span
    as if it were a real agent turn. It has no visibility into tool calls
    (there are none to show it), no inference-span message history (there is
    none), and no knowledge that the "trace" was fabricated for this adapter
    rather than captured live. The resulting score and explanation should be
    read as "how good is this response text for this brief, taken in
    isolation" -- not as a judgment of the full agent trajectory the way
    AgentCore evaluation is designed to assess. This is the core, accepted
    limitation of adapting a plain-text review record into AgentCore's
    span-based evaluation surface; see also ``agentcore_scorer``'s
    docstring.

    Args:
        record: the review record whose brief text becomes the span's
            user-prompt attribute and whose response text (for
            ``record.agent_name``, falling back to the first available
            response if that key is absent) becomes the span's
            agent-response attribute.
        metric: unused directly by the span itself (the metric's
            rubric/definition is carried via ``evaluationReferenceInputs``
            instead -- see ``agentcore_scorer``) but accepted for a
            consistent call signature and used to disambiguate the
            synthetic ``spanId``.

    Returns:
        A single-element list holding one synthetic ADOT span document,
        suitable for passing directly as ``evaluationInput.sessionSpans``.
        (Returned as a list so the caller can treat ``sessionSpans`` as a
        list uniformly and so a future revision could add correlated
        inference/tool spans without changing the call site.)
    """
    response_text = record.responses.get(record.agent_name)
    if response_text is None:
        # Multi-agent records (see persistence.MULTI_AGENT_SENTINEL) key
        # agent_name as "multi" with no single matching responses entry;
        # fall back to the first available response rather than scoring
        # an empty string, so the synthetic span still carries real
        # content whenever any response text exists at all.
        response_text = next(iter(record.responses.values()), "")

    trace_id = f"synthetic-trace-{record.record_id}"
    span_id = f"synthetic-span-{record.record_id}-{metric.name}"

    # The synthetic span has no real invocation duration (there was no
    # live agent call to time), so it is stamped as a zero-duration span
    # ending "now" -- a timestamp is required by the service, but its
    # exact value has no evaluative meaning here.
    now_ns = int(time.time() * 1_000_000_000)

    # Shape per AgentCore Evaluations' documented generic framework support
    # (supported-frameworks-generic.html): an OpenTelemetry-convention span
    # recognized by (a) a scope name under the opentelemetry.instrumentation.*
    # prefix, (b) the gen_ai.operation.name identifying attribute, and (c)
    # the gen_ai.task.input/output content attributes for an invoke-agent
    # span, with agentcore.invocation.* as the documented any-scope fallback
    # for the top-level agent prompt/response. Content is carried on the span
    # itself (unified-telemetry mode), so no separate correlated log-record
    # document is needed.
    span_doc = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": "invoke_agent",
        "kind": "SERVER",
        "startTimeUnixNano": now_ns,
        "endTimeUnixNano": now_ns,
        "durationNano": 0,
        "status": {"code": "OK"},
        "scope": {"name": "opentelemetry.instrumentation.agent-health-eval-synthetic"},
        "resource": {"attributes": {}},
        "attributes": {
            # Identifying attribute: classifies this as an invoke-agent
            # (top-level agent turn) span.
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": record.agent_name,
            # Primary content attributes for an invoke-agent span on the
            # OpenTelemetry reading path.
            "gen_ai.task.input": record.brief_text,
            "gen_ai.task.output": response_text,
            # Documented any-scope fallback for the top-level agent
            # prompt/response.
            "agentcore.invocation.user_prompt": record.brief_text,
            "agentcore.invocation.agent_response": response_text,
            "session.id": trace_id,
        },
    }

    return [span_doc]


def agentcore_scorer(
    record: ReviewRecord,
    metric: CandidateHealthMetric,
    reference_label: Optional[str],
) -> Tuple[float, str]:
    """Score one ``(record, metric)`` pair using a live call to the
    ``bedrock-agentcore`` SDK's ``EvaluationClient.evaluate`` (the data-plane
    ``Evaluate`` operation), satisfying the ``Scorer`` contract documented
    at the top of this module.

    WHY THIS IS AN ADAPTER, NOT A DIRECT MAPPING:

    The installed ``bedrock-agentcore`` SDK's real evaluation surface (
    confirmed against the SDK actually installed in this environment, not
    just its docs) is session/trace/span-oriented:
    ``EvaluationClient``/the ``Evaluate`` data-plane operation takes
    ``evaluationInput.sessionSpans`` (a list of OpenTelemetry/ADOT span
    documents), an optional ``evaluationTarget`` (trace/span IDs),
    ``evaluationReferenceInputs`` (ground truth, scoped by span context),
    and an ``evaluatorId`` that is either a fixed built-in name
    (``"Builtin.Helpfulness"``, ``"Builtin.Correctness"``, etc.) or a
    *pre-registered* custom evaluator ID -- never an arbitrary rubric
    string passed inline per call.

    This demo's ``CandidateHealthMetric`` is the opposite shape: a
    free-text ``name``/``definition``/``rubric``/``traceability`` proposed
    per-agent by an LLM, with no corresponding OTel trace ever captured for
    the sibling module's agents (see ``_build_synthetic_llm_span``'s
    docstring for the full detail on why no 1:1 mapping exists).

    RESEARCH INTO A CLEANER PATH (so this doesn't have to be re-litigated
    later): the installed SDK does expose
    ``bedrock_agentcore.evaluation.custom_code_based_evaluators`` --
    ``@custom_code_based_evaluator()`` lets you write a Python function
    that receives an ``EvaluatorInput`` (still span-based) and returns an
    ``EvaluatorOutput`` (a score/label/explanation), which *could* embed
    an arbitrary rubric in the evaluator's own code. However, this is not
    a per-call, inline mechanism: the decorated function must be deployed
    as its own AWS Lambda, then registered once via the control-plane
    ``create_evaluator`` API to obtain a custom ``evaluatorId``, before
    ``evaluate()`` can reference it -- i.e. it is an out-of-band
    infrastructure-provisioning step (one Lambda + one registered
    evaluator per distinct rubric), not something ``agentcore_scorer`` can
    do synchronously inside a per-(record, metric) scorer call. Building
    that provisioning pipeline is out of scope for this task (it would be
    its own feature: dynamically deploying/registering a Lambda-backed
    evaluator per ``CandidateHealthMetric.rubric``), so this function does
    not use it. No other inline/freeform-rubric evaluation entry point
    exists in the installed SDK.

    THE ADAPTER THIS FUNCTION USES, given that mismatch:
      1. Builds a minimal synthetic OpenTelemetry/ADOT invoke-agent span
         wrapping the record's brief + response text via
         ``_build_synthetic_llm_span``, shaped per AgentCore's documented
         *generic framework support* convention (see that function's
         docstring for the detailed limitation this introduces, and for the
         exact scope-name / identifying-attribute / content-attribute shape
         the service requires to recognize a self-instrumented span).
      2. Uses ``DEFAULT_EVALUATOR_ID`` (``"Builtin.Correctness"``) as the
         ``evaluatorId`` -- the most general-purpose built-in evaluator
         for judging whether a response is correct/appropriate, since no
         built-in evaluator can be parameterized with an arbitrary
         ``CandidateHealthMetric`` rubric. Discovered empirically via a
         live call during this task's verification (not documented in the
         written API reference): ``Builtin.Correctness`` operates at
         TRACE level, and the live service rejects an
         ``evaluationReferenceInputs[].assertions`` entry on a
         TRACE-scoped context ("Fields {'assertions'} are not valid for
         TRACE-level context. Valid fields: {'expectedResponse'}"). So the
         metric's ``name``/``definition``/``rubric`` text -- the closest
         available carrier for a custom rubric, since no built-in
         evaluator field accepts an arbitrary rubric directly -- is folded
         into the one field this evaluator's level does accept:
         ``evaluationReferenceInputs[].expectedResponse.text``. When
         ``reference_label`` (an Expected_State_Label, when one exists for
         this record) is present, its text is included ahead of the
         metric's own rubric text, since a Domain_Expert-authored
         Expected_State_Label is a more concrete ground truth than an
         LLM-proposed rubric description.
      3. Normalizes the result's ``value``/``label`` into this demo's
         documented 0.0-1.0 ``MetricResult.score`` scale via
         ``_normalize_score``.

    LIMITATION (restated at the call site, since it is the single most
    important thing for a reader of an evaluation run's results to know):
    the resulting score reflects the built-in evaluator's own fixed
    correctness rubric applied to one synthetic single-span "trace"
    wrapping the response text, NOT a genuine evaluation against
    ``metric.rubric``. ``metric.rubric``/``metric.definition`` are
    surfaced to the evaluator only as supplementary natural-language
    assertions, which a built-in evaluator may or may not weight heavily.
    Rationale text returned to callers is prefixed accordingly so this
    limitation travels with every persisted ``MetricResult``.

    Args:
        record: the review record being scored.
        metric: the candidate health metric whose rubric/definition is
            passed through as supplementary reference-input context (see
            above); the metric's own fixed rubric is NOT evaluated
            directly, per the limitation above.
        reference_label: the latest ``ExpectedStateLabel.label_text`` for
            this record, or ``None`` if no label has been recorded. Passed
            as the ``expectedResponse`` reference input when present.

    Returns:
        A ``(score, rationale)`` tuple: ``score`` normalized to
        ``[0.0, 1.0]``, ``rationale`` prefixed with this adapter's
        limitation note and then the evaluator's own ``explanation``
        (verbatim) if one was returned.

    Raises:
        AgentCoreConfigurationError: if the AgentCore evaluation client
            cannot be constructed at all (missing ``bedrock-agentcore``
            package, or no resolvable AWS credentials/region). This is
            NOT caught here -- it propagates out of this function and,
            per Requirement 4.7, is meant to abort the whole
            ``run_evaluation`` run rather than being recorded as a
            per-pair ``MetricFailure``. (``run_evaluation``'s per-pair
            ``try/except Exception`` will still technically catch it if
            called from inside that loop, since Python cannot
            distinguish "abort the run" exceptions from
            "isolate this pair" exceptions by type alone without the
            caller cooperating -- callers that want strict Req-4.7
            abort-the-run behavior should construct/validate the
            AgentCore client up front, e.g. by calling
            ``_get_agentcore_client()`` once before invoking
            ``run_evaluation``, so a configuration failure surfaces before
            any pair is scored. See this module's ``if __name__ ==
            "__main__"`` live check for the pattern.)
        Exception: any error the underlying ``EvaluationClient.evaluate``
            call itself raises (e.g. a service-side validation error,
            throttling, or a malformed response) propagates unchanged, to
            be converted into a ``MetricFailure`` by ``run_evaluation``'s
            per-pair isolation.
    """
    client = _get_agentcore_client()

    span_documents = _build_synthetic_llm_span(record, metric)
    session_id = span_documents[0]["traceId"]

    # ADAPTATION NOTE (discovered empirically via a live call during this
    # task's verification, not documented in the written API reference):
    # DEFAULT_EVALUATOR_ID ("Builtin.Correctness") is a TRACE-level
    # evaluator, and the live service rejects an `assertions` entry on a
    # TRACE-scoped evaluationReferenceInputs context with "Fields
    # {'assertions'} are not valid for TRACE-level context. Valid fields:
    # {'expectedResponse'}". So the metric's rubric/definition text -- the
    # closest available carrier for a custom rubric per this function's
    # docstring -- is folded directly into the `expectedResponse` text
    # (the one field this evaluator's level actually accepts), rather than
    # passed as a separate assertions list. When a reference_label is also
    # present, it is included as the stated expected outcome ahead of the
    # metric's own rubric text, since a Domain_Expert-authored
    # Expected_State_Label is a more concrete ground truth than an
    # LLM-proposed rubric description.
    expected_response_parts = []
    if reference_label:
        expected_response_parts.append(f"Expected state: {reference_label}")
    expected_response_parts.append(f"Health metric being judged: {metric.name}")
    if metric.definition:
        expected_response_parts.append(f"Definition: {metric.definition}")
    if metric.rubric:
        expected_response_parts.append(f"Rubric: {metric.rubric}")
    expected_response_text = "\n".join(expected_response_parts)

    request: dict = {
        "evaluatorId": DEFAULT_EVALUATOR_ID,
        "evaluationInput": {"sessionSpans": span_documents},
        "evaluationReferenceInputs": [
            {
                "context": {"spanContext": {"sessionId": session_id, "traceId": session_id}},
                "expectedResponse": {"text": expected_response_text},
            }
        ],
    }

    response = client.evaluate(**request)
    evaluation_results = response.get("evaluationResults", [])
    if not evaluation_results:
        raise ValueError(
            f"AgentCore evaluate() returned no evaluationResults for "
            f"record_id={record.record_id!r} metric_name={metric.name!r} "
            f"(evaluatorId={DEFAULT_EVALUATOR_ID!r})"
        )

    result = evaluation_results[0]
    if result.get("errorCode"):
        raise RuntimeError(
            f"AgentCore evaluator {DEFAULT_EVALUATOR_ID!r} returned an "
            f"error for record_id={record.record_id!r} "
            f"metric_name={metric.name!r}: "
            f"{result.get('errorCode')}: {result.get('errorMessage')}"
        )

    score = _normalize_score(result.get("value"), result.get("label"))

    limitation_note = (
        f"[agentcore_scorer adapter note: scored via built-in evaluator "
        f"{DEFAULT_EVALUATOR_ID!r} against a synthetic single-span trace "
        f"of this response, with metric '{metric.name}'s rubric/"
        f"definition supplied only as supplementary free-text assertions "
        f"-- not a direct evaluation of that rubric.]"
    )
    explanation = result.get("explanation")
    rationale = f"{limitation_note} {explanation}" if explanation else limitation_note

    return score, rationale


def run_evaluation(
    metrics: List[CandidateHealthMetric],
    records: List[ReviewRecord],
    scorer: Scorer = agentcore_scorer,
) -> Tuple[List[MetricResult], List[MetricFailure]]:
    """Run an Evaluator_Run: score every ``(record, metric)`` combination.

    Iterates the full Cartesian product of ``records x metrics`` (Req 4.1,
    4.2, 4.3). For each pair, looks up the latest ``ExpectedStateLabel`` for
    ``record.record_id`` via ``feedback_store.latest_label_for_record`` and
    passes its ``label_text`` (or ``None`` if no label exists) to ``scorer``
    as ``reference_label`` (Req 4.6). Each call is wrapped in its own
    ``try/except Exception``, so a failing pair is isolated and never aborts
    the remaining pairs (Req 4.4). When the full product has been scored,
    both the results and failures lists are persisted (Req 4.5).

    Args:
        metrics: the ``CandidateHealthMetric``s to score every record
            against.
        records: the ``ReviewRecord``s to score.
        scorer: a callable ``scorer(record, metric, reference_label)``
            returning a ``(score, rationale)`` pair (or an object/dict
            shape accepted by ``_coerce_score_outcome``) on success, or
            raising any ``Exception`` on failure. Defaults to
            ``agentcore_scorer``, the live AgentCore-calling adapter
            defined above, per design.md. Callers may still inject a fake/
            test scorer by passing this argument explicitly (as this
            module's own tests and sanity check below do), so importing
            this module does not, by itself, require AWS credentials --
            only actually *calling* ``run_evaluation`` with the default
            scorer does.

    Returns:
        A ``(results, failures)`` tuple: ``results`` holds one
        ``MetricResult`` per non-raising ``(record, metric)`` pair,
        ``failures`` holds one ``MetricFailure`` per raising pair. Across
        both lists, ``len(results) + len(failures) ==
        len(records) * len(metrics)``, with every pair represented exactly
        once.
    """
    # Load the Feedback_Store once per run rather than once per pair, since
    # it is append-only and will not change mid-run; this also avoids
    # re-reading the file len(records) * len(metrics) times.
    feedback_entries = feedback_store.load_feedback()

    results: List[MetricResult] = []
    failures: List[MetricFailure] = []

    for record in records:
        label = feedback_store.latest_label_for_record(record.record_id, feedback_entries)
        reference_label = label.label_text if label is not None else None

        for metric in metrics:
            try:
                raw_outcome = scorer(record, metric, reference_label)
                score, rationale = _coerce_score_outcome(raw_outcome)
                results.append(
                    MetricResult(
                        record_id=record.record_id,
                        metric_name=metric.name,
                        agent_name=record.agent_name,
                        score=score,
                        rationale=rationale,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - intentional: isolate any scorer failure
                failures.append(
                    MetricFailure(
                        record_id=record.record_id,
                        metric_name=metric.name,
                        error=str(exc),
                    )
                )

    for result in results:
        append_jsonl(METRIC_RESULTS_PATH, result.to_dict())
    for failure in failures:
        append_jsonl(METRIC_FAILURES_PATH, failure.to_dict())

    return results, failures


if __name__ == "__main__":
    # Sanity check, run with: python evaluator_runner.py
    #
    # Uses a fake, non-AWS scorer and distinct test-only output filenames
    # (rather than the real METRIC_RESULTS_PATH/METRIC_FAILURES_PATH) so this
    # check never pollutes real Evaluator_Run output, and cleans those files
    # up at the end regardless of outcome.
    import os

    from persistence import read_jsonl, resolve_output_path

    _SANITY_RESULTS_PATH = "results/_sanity_check_metric_results.jsonl"
    _SANITY_FAILURES_PATH = "results/_sanity_check_metric_results.failures.jsonl"

    fake_records = [
        ReviewRecord(record_id="case-1", agent_name="market_trends", brief_text="brief 1", responses={}),
        ReviewRecord(record_id="case-2", agent_name="customer_insights", brief_text="brief 2", responses={}),
    ]
    fake_metrics = [
        CandidateHealthMetric(
            name="groundedness", definition="def", rubric="rubric", traceability="trace",
            agent_name="market_trends",
        ),
        CandidateHealthMetric(
            name="completeness", definition="def", rubric="rubric", traceability="trace",
            agent_name="market_trends",
        ),
    ]

    # Fail exactly one of the 4 (record, metric) combinations.
    _FAILING_PAIR = ("case-2", "completeness")

    def _fake_scorer(record: ReviewRecord, metric: CandidateHealthMetric, reference_label):
        if (record.record_id, metric.name) == _FAILING_PAIR:
            raise RuntimeError("simulated scorer failure")
        return (0.9, f"scored {record.record_id}/{metric.name} ref={reference_label!r}")

    results_file = resolve_output_path(*_SANITY_RESULTS_PATH.split("/"))
    failures_file = resolve_output_path(*_SANITY_FAILURES_PATH.split("/"))

    # Monkeypatch the module-level output paths for the duration of this
    # sanity check only, so run_evaluation's own persistence calls land in
    # the sanity-check files instead of the real ones. Patching the current
    # module's own globals() (rather than re-importing by name, which would
    # create a second, distinct module object when this file is run as
    # __main__) ensures run_evaluation actually observes the patched value.
    _original_results_path = METRIC_RESULTS_PATH
    _original_failures_path = METRIC_FAILURES_PATH
    globals()["METRIC_RESULTS_PATH"] = _SANITY_RESULTS_PATH
    globals()["METRIC_FAILURES_PATH"] = _SANITY_FAILURES_PATH

    try:
        results, failures = run_evaluation(fake_metrics, fake_records, _fake_scorer)

        assert len(results) == 3, f"expected 3 results, got {len(results)}"
        assert len(failures) == 1, f"expected 1 failure, got {len(failures)}"
        print(f"[ok] run_evaluation produced 3 results and 1 failure "
              f"(4 total pairs = 2 records x 2 metrics)")

        failure = failures[0]
        assert failure.record_id == "case-2", failure.record_id
        assert failure.metric_name == "completeness", failure.metric_name
        print(f"[ok] failure matches the configured failing pair: "
              f"record_id={failure.record_id!r} metric_name={failure.metric_name!r}")

        # Confirm persisted files exist and round-trip correctly.
        persisted_results = read_jsonl(results_file)
        persisted_failures = read_jsonl(failures_file)
        assert len(persisted_results) == 3, persisted_results
        assert len(persisted_failures) == 1, persisted_failures
        assert [MetricResult.from_dict(r) for r in persisted_results] == results
        assert [MetricFailure.from_dict(f) for f in persisted_failures] == failures
        print(f"[ok] persisted files exist and round-trip correctly: "
              f"{results_file.name}, {failures_file.name}")

        print("\nAll evaluator_runner sanity checks passed.")
    finally:
        globals()["METRIC_RESULTS_PATH"] = _original_results_path
        globals()["METRIC_FAILURES_PATH"] = _original_failures_path
        for f in (results_file, failures_file):
            if f.exists():
                os.remove(f)
                print(f"[cleanup] removed {f}")
