"""Guards for tests/conftest.py's output-path isolation.

These checks fail loudly if a future change makes a persisted-data path
constant bind at import/definition time again, or if the conftest redirect
stops covering one of the modules that writes data.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

import evaluator_runner  # noqa: E402
import feedback_store  # noqa: E402
import metric_proposer  # noqa: E402
import run_pipeline  # noqa: E402
from models import ExpectedStateLabel  # noqa: E402
from persistence import OUTPUT_DIR  # noqa: E402

_SCRATCH_ROOT = (OUTPUT_DIR / "_test_scratch").resolve()


def _is_under_scratch(path: str) -> bool:
    try:
        Path(path).resolve().relative_to(_SCRATCH_ROOT)
        return True
    except ValueError:
        return False


def test_all_data_path_constants_point_at_scratch():
    for value in (
        feedback_store.FEEDBACK_STORE_PATH,
        evaluator_runner.METRIC_RESULTS_PATH,
        evaluator_runner.METRIC_FAILURES_PATH,
        metric_proposer.CANDIDATE_METRICS_PATH,
        metric_proposer.REJECTED_METRICS_PATH,
        run_pipeline.RUNS_DIR,
    ):
        assert _is_under_scratch(value), value


def test_default_feedback_read_and_write_share_the_redirected_path(_isolate_output_data_paths):
    feedback_store.append_expected_state_label(
        ExpectedStateLabel(
            record_id="isolation-check",
            agent_name="market_trends",
            label_text="scratch only",
            timestamp="2024-01-01T00:00:00Z",
        )
    )
    # No path argument: must resolve the (monkeypatched) constant at call time.
    entries = feedback_store.load_feedback()
    assert [e.record_id for e in entries] == ["isolation-check"]
    assert (_isolate_output_data_paths / "feedback" / "feedback_store.jsonl").exists()
