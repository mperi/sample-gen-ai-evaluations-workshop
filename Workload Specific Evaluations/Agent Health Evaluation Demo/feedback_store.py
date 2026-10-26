"""Feedback_Store for Agent_Health_Evaluation_Demo.

The seam shared between Phase 2 (``review_app.py``, writing) and Phase 5
(reading alongside ``MetricResult``s), keeping the Feedback_Store's on-disk
format decoupled from both the Review_UI's Streamlit-specific code and the
Evaluator_Runner's AWS-specific code.

Both ``append_*`` functions append one JSON line (with an ``"entry_type"``
discriminator already added by the dataclass's own ``to_dict()``, see
``models.py``) to ``output/feedback/feedback_store.jsonl`` via
``persistence.append_jsonl`` — append-only, so a record can accumulate a
label history. ``load_feedback`` reads all lines back and dispatches each
row to ``ExpectedStateLabel.from_dict`` or ``TaskSuccessFeedback.from_dict``
based on that discriminator. ``latest_label_for_record`` returns the most
recently appended ``ExpectedStateLabel`` for a given record id, which
``review_app.py`` uses to pre-fill the Expected_State_Label input control
(Requirement 3.7).

Per design.md's Error Handling section: write failures (disk full,
permissions) are allowed to propagate/raise rather than being swallowed,
since silently dropping a Domain_Expert's submitted label would defeat
Requirement 6's purpose. Neither ``append_*`` function catches exceptions
from ``persistence.append_jsonl`` — the Review_UI is responsible for
surfacing the exception as a Streamlit error banner (Phase 2/7.2), not this
module.

Requirements: 3.5, 3.7, 6.2, 6.3, 6.5
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple, Union

from pathlib import Path

from models import ExpectedStateLabel, MetricResult, TaskSuccessFeedback
from persistence import append_jsonl, read_jsonl, resolve_output_path

# Default path (relative to this package's output/ directory, resolved by
# persistence.py's write/read helpers) for the Feedback_Store's single JSONL
# file. Both Expected_State_Labels and Task_Success_Feedback entries are
# appended to this same file, distinguished by their "entry_type".
FEEDBACK_STORE_PATH = "feedback/feedback_store.jsonl"

# Default pass threshold (0-1 scale) used by find_disagreements: a
# MetricResult.score at or above this value is treated as a "pass" outcome,
# below it as a "fail" outcome, per design.md's Feedback_Store section.
DEFAULT_DISAGREEMENT_THRESHOLD = 0.7

FeedbackEntry = Union[ExpectedStateLabel, TaskSuccessFeedback]


def append_expected_state_label(label: ExpectedStateLabel) -> None:
    """Append one ``ExpectedStateLabel`` to the Feedback_Store.

    ``label.to_dict()`` already includes the ``"entry_type":
    "expected_state_label"`` discriminator (see ``models.py``), so it is
    passed straight through to ``persistence.append_jsonl``. Any exception
    raised by the underlying write (disk full, permissions, etc.) is
    allowed to propagate — this function does not catch or swallow it, per
    design.md's Error Handling section for Feedback_Store.

    Args:
        label: the ``ExpectedStateLabel`` to persist.
    """
    append_jsonl(FEEDBACK_STORE_PATH, label.to_dict())


def append_task_success_feedback(feedback: TaskSuccessFeedback) -> None:
    """Append one ``TaskSuccessFeedback`` to the Feedback_Store.

    ``feedback.to_dict()`` already includes the ``"entry_type":
    "task_success_feedback"`` discriminator (see ``models.py``), so it is
    passed straight through to ``persistence.append_jsonl``. Any exception
    raised by the underlying write (disk full, permissions, etc.) is
    allowed to propagate — this function does not catch or swallow it, per
    design.md's Error Handling section for Feedback_Store.

    Args:
        feedback: the ``TaskSuccessFeedback`` to persist.
    """
    append_jsonl(FEEDBACK_STORE_PATH, feedback.to_dict())


def load_feedback(path: Optional[str] = None) -> List[FeedbackEntry]:
    """Load all Feedback_Store entries, reconstructed into their concrete
    dataclass types.

    Reads every line of ``path`` via ``persistence.read_jsonl`` and
    dispatches each row to ``ExpectedStateLabel.from_dict`` or
    ``TaskSuccessFeedback.from_dict`` based on the row's ``"entry_type"``
    discriminator, so the returned list may contain a mix of both types.
    A missing file yields an empty list (``read_jsonl``'s existing
    behavior).

    Args:
        path: path to the Feedback_Store's JSONL file. ``None`` (the
            default) means "the current value of the module-level
            ``FEEDBACK_STORE_PATH``", resolved at call time rather than at
            function-definition time, so reassigning/monkeypatching
            ``feedback_store.FEEDBACK_STORE_PATH`` redirects reads and
            writes together. A bare relative
            path (e.g. ``FEEDBACK_STORE_PATH``, or any other
            relative path without a leading ``output/``) is resolved
            under this package's own ``output/`` directory via
            ``persistence.resolve_output_path``, mirroring how
            ``append_expected_state_label``/``append_task_success_feedback``
            resolve their write target — so the default argument reads
            back exactly what those functions wrote. An absolute path is
            used as-is, so a hand-edited or relocated copy of the file can
            still be loaded directly.

    Returns:
        A list of ``ExpectedStateLabel`` / ``TaskSuccessFeedback``
        instances, in file (append) order.

    Raises:
        ValueError: if a row's ``"entry_type"`` is missing or not one of
            the two known discriminator values — this indicates a
            corrupted or foreign file rather than a normal empty/missing
            file (which ``read_jsonl`` already handles by returning
            ``[]``).
    """
    if path is None:
        path = FEEDBACK_STORE_PATH
    resolved_path = path if Path(path).is_absolute() else resolve_output_path(*Path(path).parts)
    rows = read_jsonl(resolved_path)
    entries: List[FeedbackEntry] = []
    for row in rows:
        entry_type = row.get("entry_type")
        if entry_type == "expected_state_label":
            entries.append(ExpectedStateLabel.from_dict(row))
        elif entry_type == "task_success_feedback":
            entries.append(TaskSuccessFeedback.from_dict(row))
        else:
            raise ValueError(
                f"Unknown or missing 'entry_type' discriminator in Feedback_Store "
                f"row: {entry_type!r}"
            )
    return entries


def latest_label_for_record(
    record_id: str, entries: Optional[List[FeedbackEntry]] = None
) -> Optional[ExpectedStateLabel]:
    """Return the most recently appended ``ExpectedStateLabel`` for
    ``record_id``, or ``None`` if there is no such label.

    Approach: the Feedback_Store is append-only (``append_jsonl`` only ever
    adds lines, never rewrites or reorders existing ones), so "most
    recently appended" is equivalent to "last matching entry in file
    order". This function filters to ``ExpectedStateLabel`` entries whose
    ``record_id`` matches and returns the LAST one in list order, rather
    than sorting by the ``timestamp`` field. That is simpler and more
    robust than a timestamp sort: it has no dependency on timestamp
    formatting/parsing, and it still returns the correct answer even if
    two labels for the same record happen to share an identical (or
    missing/malformed) timestamp, which a timestamp-based sort could get
    wrong or raise on.

    Args:
        record_id: the ``ReviewRecord.record_id`` to look up.
        entries: a pre-loaded list of Feedback_Store entries to search
            instead of loading from disk (useful for tests and for
            callers that already have the full list in hand, e.g.
            ``evaluator_runner.run_evaluation`` scoring many records in a
            loop). If ``None`` (the default), entries are loaded fresh via
            ``load_feedback()``.

    Returns:
        The latest matching ``ExpectedStateLabel``, or ``None`` if no
        ``ExpectedStateLabel`` in ``entries`` has this ``record_id``.
    """
    if entries is None:
        entries = load_feedback()

    latest: Optional[ExpectedStateLabel] = None
    for entry in entries:
        if isinstance(entry, ExpectedStateLabel) and entry.record_id == record_id:
            latest = entry
    return latest


def find_disagreements(
    entries: List[FeedbackEntry],
    results: List[MetricResult],
    threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
) -> List[Tuple[TaskSuccessFeedback, MetricResult]]:
    """Find ``(TaskSuccessFeedback, MetricResult)`` pairs that share a
    ``record_id`` but disagree about whether the task succeeded.

    A pair disagrees when:
      - ``feedback.passed is False`` and ``result.score >= threshold``
        (the Domain_Expert judged failure, but the metric score says it
        passed), or
      - ``feedback.passed is True`` and ``result.score < threshold``
        (the Domain_Expert judged success, but the metric score says it
        failed).

    ``entries`` may be the mixed list returned by ``load_feedback()``
    (``ExpectedStateLabel`` and ``TaskSuccessFeedback`` together); only the
    ``TaskSuccessFeedback`` entries participate in pairing. Pairing is done
    per shared ``record_id``: every ``TaskSuccessFeedback`` for a given
    ``record_id`` is checked against every ``MetricResult`` for that same
    ``record_id`` (so a record with multiple feedback entries and/or
    multiple scored metrics yields the full cross-product of pairs for
    that record), and only the disagreeing pairs are kept. A
    ``TaskSuccessFeedback`` or ``MetricResult`` whose ``record_id`` has no
    counterpart on the other side contributes no pairs.

    Args:
        entries: a (possibly mixed-type) list of Feedback_Store entries,
            e.g. as returned by ``load_feedback()``.
        results: the ``MetricResult``s to check against, e.g. as loaded
            from ``output/results/metric_results.jsonl``.
        threshold: the pass/fail score cutoff on MetricResult's 0.0-1.0
            scale; a score at or above this value counts as a passing
            outcome. Defaults to ``DEFAULT_DISAGREEMENT_THRESHOLD`` (0.7).

    Returns:
        A list of ``(TaskSuccessFeedback, MetricResult)`` tuples, one per
        disagreeing pair, in the order produced by iterating feedback
        entries (grouped by record id) against their matching results.
    """
    feedback_by_record: "defaultdict[str, List[TaskSuccessFeedback]]" = defaultdict(list)
    for entry in entries:
        if isinstance(entry, TaskSuccessFeedback):
            feedback_by_record[entry.record_id].append(entry)

    results_by_record: "defaultdict[str, List[MetricResult]]" = defaultdict(list)
    for result in results:
        results_by_record[result.record_id].append(result)

    disagreements: List[Tuple[TaskSuccessFeedback, MetricResult]] = []
    for record_id, feedback_list in feedback_by_record.items():
        for result in results_by_record.get(record_id, []):
            for feedback in feedback_list:
                passed_disagrees_with_fail_score = feedback.passed and result.score < threshold
                failed_disagrees_with_pass_score = (not feedback.passed) and result.score >= threshold
                if passed_disagrees_with_fail_score or failed_disagrees_with_pass_score:
                    disagreements.append((feedback, result))
    return disagreements


if __name__ == "__main__":
    # Sanity check, run with: python feedback_store.py
    #
    # Uses a distinct test-only filename under output/feedback/ (rather than
    # the real FEEDBACK_STORE_PATH) so this check never pollutes real
    # Feedback_Store data, and cleans that file up at the end regardless of
    # outcome.
    import os

    from persistence import resolve_output_path

    _SANITY_PATH = "feedback/_sanity_check_feedback_store.jsonl"

    def _append_label(record_id: str, agent_name: str, label_text: str, timestamp: str) -> None:
        append_jsonl(
            _SANITY_PATH,
            ExpectedStateLabel(
                record_id=record_id,
                agent_name=agent_name,
                label_text=label_text,
                timestamp=timestamp,
            ).to_dict(),
        )

    def _append_feedback(record_id: str, agent_name: str, passed: bool, rationale: str, timestamp: str) -> None:
        append_jsonl(
            _SANITY_PATH,
            TaskSuccessFeedback(
                record_id=record_id,
                agent_name=agent_name,
                passed=passed,
                rationale=rationale,
                timestamp=timestamp,
            ).to_dict(),
        )

    sanity_file = resolve_output_path(*_SANITY_PATH.split("/"))
    try:
        # Two labels for the same record_id (different label_text), plus one
        # TaskSuccessFeedback for a different record_id.
        _append_label("case-1", "market_trends", "First label.", "2024-01-01T00:00:00Z")
        _append_label("case-1", "market_trends", "Second (most recent) label.", "2024-01-01T00:01:00Z")
        _append_feedback("case-2", "customer_insights", True, "Looked correct.", "2024-01-01T00:02:00Z")

        entries = load_feedback(_SANITY_PATH)
        assert len(entries) == 3, f"expected 3 entries, got {len(entries)}"
        assert isinstance(entries[0], ExpectedStateLabel)
        assert isinstance(entries[1], ExpectedStateLabel)
        assert isinstance(entries[2], TaskSuccessFeedback)
        print(f"[ok] load_feedback() returned 3 mixed-type entries correctly typed: "
              f"{[type(e).__name__ for e in entries]}")

        latest = latest_label_for_record("case-1", entries)
        assert latest is not None, "expected a label for case-1"
        assert latest.label_text == "Second (most recent) label.", latest.label_text
        print(f"[ok] latest_label_for_record('case-1') returned the second label: "
              f"{latest.label_text!r}")

        none_result = latest_label_for_record("case-3", entries)
        assert none_result is None, none_result
        print("[ok] latest_label_for_record('case-3') correctly returned None "
              "(no label for that record)")

        # --- find_disagreements sanity check ---
        agreeing_feedback = TaskSuccessFeedback(
            record_id="rec-agree", agent_name="market_trends", passed=True,
            rationale="Looked right.", timestamp="2024-01-01T00:00:00Z",
        )
        agreeing_result = MetricResult(
            record_id="rec-agree", metric_name="groundedness", agent_name="market_trends",
            score=0.85, rationale="Well grounded.",
        )

        disagree_fail_feedback = TaskSuccessFeedback(
            record_id="rec-disagree-1", agent_name="market_trends", passed=False,
            rationale="Missed the premise.", timestamp="2024-01-01T00:01:00Z",
        )
        disagree_fail_result = MetricResult(
            record_id="rec-disagree-1", metric_name="groundedness", agent_name="market_trends",
            score=0.9, rationale="Scored high despite the miss.",
        )

        disagree_pass_feedback = TaskSuccessFeedback(
            record_id="rec-disagree-2", agent_name="customer_insights", passed=True,
            rationale="Nailed it.", timestamp="2024-01-01T00:02:00Z",
        )
        disagree_pass_result = MetricResult(
            record_id="rec-disagree-2", metric_name="groundedness", agent_name="customer_insights",
            score=0.2, rationale="Scored low despite success.",
        )

        unmatched_feedback = TaskSuccessFeedback(
            record_id="rec-no-match-feedback", agent_name="market_trends", passed=False,
            rationale="No corresponding metric result.", timestamp="2024-01-01T00:03:00Z",
        )
        unmatched_result = MetricResult(
            record_id="rec-no-match-result", metric_name="groundedness", agent_name="market_trends",
            score=0.1, rationale="No corresponding feedback entry.",
        )

        disagreement_entries: List[FeedbackEntry] = [
            agreeing_feedback, disagree_fail_feedback, disagree_pass_feedback, unmatched_feedback,
        ]
        disagreement_results = [
            agreeing_result, disagree_fail_result, disagree_pass_result, unmatched_result,
        ]

        disagreements = find_disagreements(disagreement_entries, disagreement_results)
        assert len(disagreements) == 2, f"expected 2 disagreements, got {len(disagreements)}"
        disagreement_record_ids = {feedback.record_id for feedback, _ in disagreements}
        assert disagreement_record_ids == {"rec-disagree-1", "rec-disagree-2"}, disagreement_record_ids
        for feedback, result in disagreements:
            assert feedback.record_id == result.record_id
        print(f"[ok] find_disagreements() returned exactly the 2 disagreeing pairs: "
              f"{sorted(disagreement_record_ids)}")

        print("\nAll feedback_store sanity checks passed.")
    finally:
        if sanity_file.exists():
            os.remove(sanity_file)
            print(f"[cleanup] removed {sanity_file}")
