"""Review_UI for Agent_Health_Evaluation_Demo (Phase 2 / Phase 5).

Run with: ``streamlit run review_app.py``

On start, loads ``ReviewRecord``s read-only via
``persistence.load_review_records()`` (which itself reads only from the
sibling module's ``review/review_cases.jsonl``, see ``persistence.py``).
A sidebar record selector lets the Domain_Expert navigate between records
within one running session (Requirement 3.6). For the selected record, the
app displays the brief/task text, one expander per agent's response text,
and the expected-behavior tag when present (Requirements 3.1, 3.2, 3.3).

Task 7.2 adds the Expected_State_Label input control: a text area
pre-filled with the latest existing label for the selected record (via
``feedback_store.latest_label_for_record``, Requirement 3.7), and a Submit
button that appends a new ``ExpectedStateLabel`` via
``feedback_store.append_expected_state_label`` (Requirement 3.4). Write
failures are caught and surfaced as a Streamlit error banner rather than
crashing the session (Requirement 3.5).

Task 7.3 adds the Task_Success_Feedback input control: a pass/fail radio
plus a rationale text area, and a Submit button that appends a new
``TaskSuccessFeedback`` via ``feedback_store.append_task_success_feedback``
(Requirements 6.1, 6.2). Write failures are likewise caught and surfaced as
a Streamlit error banner rather than crashing the session.

Task 7.4 adds the Phase 5 results/disagreement panel
(``_render_results_panel``): when ``output/results/metric_results.jsonl``
exists, it shows each ``MetricResult`` score for the selected record
(Requirement 5.5), and flags any Domain_Expert/metric disagreements for
that record found via ``feedback_store.find_disagreements`` for manual
inspection (Requirement 6.4). Before any Evaluator_Run has produced that
file, the panel shows a muted "no results yet" caption instead of
erroring. ``dashboard_publisher.py`` (task 10.1) had already landed with a
pure, AWS-free ``aggregate_by_record(results, record_id) -> Dict[str,
float]`` function by the time this task was implemented, so the per-metric
average shown here is computed by importing and calling that function
rather than re-deriving the same grouping/averaging logic locally.

The Review_UI never opens any sibling-module file in write mode; it only
calls ``persistence.load_review_records()``, which itself only reads.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 6.1, 6.2, 5.5, 6.4
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import streamlit as st

import feedback_store
from dashboard_publisher import aggregate_by_record
from models import ExpectedStateLabel, MetricResult, ReviewRecord, TaskSuccessFeedback
from persistence import OUTPUT_DIR, load_review_records, read_jsonl


@st.cache_data(show_spinner="Loading review records...")
def _load_records() -> List[ReviewRecord]:
    """Load all ``ReviewRecord``s read-only from the sibling module.

    Cached per Streamlit session/process so navigating between records (a
    sidebar selectbox change, which reruns the script) does not re-read
    and re-parse the sibling's ``review/review_cases.jsonl`` on every
    interaction. The underlying call is strictly read-only (see
    ``persistence.load_review_records``).
    """
    return load_review_records()


def _render_record(record: ReviewRecord) -> None:
    """Display one ``ReviewRecord``'s brief text, per-agent responses, and
    expected-behavior tag (Requirements 3.1, 3.2, 3.3)."""
    st.subheader("Brief / Task")
    if record.brief_text:
        st.write(record.brief_text)
    else:
        st.caption("(no brief text recorded for this record)")

    st.subheader("Agent Responses")
    if record.responses:
        for agent_name, response_text in record.responses.items():
            with st.expander(agent_name, expanded=False):
                st.write(response_text)
    else:
        st.caption("(no per-agent responses parsed for this record)")

    st.subheader("Expected Behavior")
    if record.expected_behavior:
        st.markdown(f"`{record.expected_behavior}`")
    else:
        st.caption("(no expected-behavior tag present for this record)")


def _build_expected_state_label(record: ReviewRecord, label_text: str) -> ExpectedStateLabel:
    """Construct an ``ExpectedStateLabel`` for ``record`` with the given
    ``label_text`` and the current UTC time as its ISO 8601 timestamp.

    Factored out of ``_render_expected_state_label_section`` so the
    construction logic can be exercised directly (e.g. from a plain script
    or test) without going through Streamlit.
    """
    return ExpectedStateLabel(
        record_id=record.record_id,
        agent_name=record.agent_name,
        label_text=label_text,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _render_expected_state_label_section(record: ReviewRecord) -> None:
    """Render the Expected_State_Label text area and Submit button for
    ``record`` (Requirements 3.4, 3.5, 3.7).

    The text area is pre-filled with the latest existing label for this
    record, looked up fresh on every call (one lookup per record render)
    so switching records never shows a stale label from a previously
    selected record. On Submit, a new ``ExpectedStateLabel`` is appended to
    the Feedback_Store; write failures are caught and surfaced as a
    Streamlit error banner instead of crashing the session.
    """
    st.subheader("Expected State Label")

    existing_label = feedback_store.latest_label_for_record(record.record_id)
    label_text = st.text_area(
        "What should the agent have done or known for this record?",
        value=existing_label.label_text if existing_label else "",
        key=f"expected_state_label_text_{record.record_id}",
    )

    if st.button("Submit Expected State Label", key=f"submit_expected_state_label_{record.record_id}"):
        label = _build_expected_state_label(record, label_text)
        try:
            feedback_store.append_expected_state_label(label)
        except Exception as exc:  # noqa: BLE001 - surface any write failure, don't crash the session
            st.error(f"Failed to save Expected State Label: {exc}")
        else:
            st.success("Expected State Label saved.")


def _build_task_success_feedback(record: ReviewRecord, passed: bool, rationale: str) -> TaskSuccessFeedback:
    """Construct a ``TaskSuccessFeedback`` for ``record`` with the given
    ``passed``/``rationale`` and the current UTC time as its ISO 8601
    timestamp.

    Factored out of ``_render_task_success_feedback_section`` so the
    construction logic can be exercised directly (e.g. from a plain script
    or test) without going through Streamlit, mirroring
    ``_build_expected_state_label``.
    """
    return TaskSuccessFeedback(
        record_id=record.record_id,
        agent_name=record.agent_name,
        passed=passed,
        rationale=rationale,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _render_task_success_feedback_section(record: ReviewRecord) -> None:
    """Render the Task_Success_Feedback pass/fail radio, rationale text
    area, and Submit button for ``record`` (Requirements 6.1, 6.2).

    On Submit, a new ``TaskSuccessFeedback`` is appended to the
    Feedback_Store; write failures are caught and surfaced as a Streamlit
    error banner instead of crashing the session, mirroring
    ``_render_expected_state_label_section``.
    """
    st.subheader("Task Success Feedback")

    outcome = st.radio(
        "Did the agent succeed at this task?",
        options=["Pass", "Fail"],
        index=0,
        key=f"task_success_outcome_{record.record_id}",
        horizontal=True,
    )
    rationale = st.text_area(
        "Rationale",
        value="",
        key=f"task_success_rationale_{record.record_id}",
    )

    if st.button("Submit Task Success Feedback", key=f"submit_task_success_feedback_{record.record_id}"):
        feedback = _build_task_success_feedback(record, outcome == "Pass", rationale)
        try:
            feedback_store.append_task_success_feedback(feedback)
        except Exception as exc:  # noqa: BLE001 - surface any write failure, don't crash the session
            st.error(f"Failed to save Task Success Feedback: {exc}")
        else:
            st.success("Task Success Feedback saved.")


_RESULTS_RELATIVE_PATH = ("results", "metric_results.jsonl")


def _render_results_panel(record: ReviewRecord) -> None:
    """Render the Phase 5 results/disagreement panel for ``record``
    (Requirements 5.5, 6.4).

    Checks whether ``output/results/metric_results.jsonl`` exists using a
    plain ``Path`` construction (mirroring how ``resolve_output_path``
    would resolve the same segments) rather than calling
    ``resolve_output_path`` itself, so this existence check never creates
    ``output/results/`` as a side effect before an Evaluator_Run has ever
    produced that directory. If the file does not exist, shows a muted
    caption and returns early — this must never raise just because no
    Evaluator_Run has happened yet.

    If the file exists, loads every ``MetricResult`` via
    ``persistence.read_jsonl`` + ``MetricResult.from_dict``, filters to the
    ones for this record for display purposes, and uses
    ``dashboard_publisher.aggregate_by_record`` to compute each metric's
    average score for this record. It then loads the Feedback_Store and
    calls ``feedback_store.find_disagreements`` against
    the *full* loaded results list (matching that function's signature),
    filters the returned pairs down to this record, and flags any that
    remain with a warning banner showing the Domain_Expert's feedback
    alongside the disagreeing metric score/rationale for manual
    inspection.
    """
    st.subheader("Evaluator Run Results")

    plain_results_path = _results_file_path()
    if not plain_results_path.exists():
        st.caption("No evaluator run results yet.")
        return

    all_results = [MetricResult.from_dict(row) for row in read_jsonl(plain_results_path)]
    record_results = [result for result in all_results if result.record_id == record.record_id]

    if not record_results:
        st.caption("No evaluator run results for this record yet.")
    else:
        per_metric_average = aggregate_by_record(all_results, record.record_id)
        for metric_name in sorted(per_metric_average):
            st.metric(label=metric_name, value=f"{per_metric_average[metric_name]:.2f}")

        with st.expander("Rationale per metric result"):
            for result in record_results:
                st.write(f"**{result.metric_name}** (score: {result.score:.2f}): {result.rationale}")

    entries = feedback_store.load_feedback()
    disagreements = feedback_store.find_disagreements(entries, all_results)
    record_disagreements = [
        (feedback, result)
        for feedback, result in disagreements
        if feedback.record_id == record.record_id
    ]

    if record_disagreements:
        st.warning(f"{len(record_disagreements)} disagreement(s) found for this record — review manually.")
        for feedback, result in record_disagreements:
            st.markdown(
                f"- Domain expert said **{'Pass' if feedback.passed else 'Fail'}** "
                f"(\"{feedback.rationale}\"), but metric **{result.metric_name}** "
                f"scored **{result.score:.2f}** (\"{result.rationale}\")."
            )


def _results_file_path():
    """Plain, non-creating path to ``output/results/metric_results.jsonl``.

    Deliberately does not call ``persistence.resolve_output_path`` (which
    creates parent directories as a side effect) so that merely rendering
    the results panel before any Evaluator_Run has happened does not
    create ``output/results/`` on disk.
    """
    return OUTPUT_DIR.joinpath(*_RESULTS_RELATIVE_PATH)


def main() -> None:
    st.set_page_config(page_title="Agent Health Evaluation — Review", layout="wide")
    st.title("Agent Health Evaluation — Review")

    records = _load_records()

    if not records:
        st.warning("No review records were loaded from the sibling module's review dataset.")
        return

    records_by_id = {record.record_id: record for record in records}
    record_ids = list(records_by_id.keys())

    with st.sidebar:
        st.header("Records")
        selected_record_id = st.selectbox(
            "Select a record to review",
            options=record_ids,
            index=0,
            key="selected_record_id",
        )

    selected_record = records_by_id[selected_record_id]

    st.caption(f"Record `{selected_record.record_id}` ({len(records)} total loaded)")
    _render_record(selected_record)
    _render_expected_state_label_section(selected_record)
    _render_task_success_feedback_section(selected_record)
    _render_results_panel(selected_record)


if __name__ == "__main__":
    main()
