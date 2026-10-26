"""Suite-wide test isolation for Agent_Health_Evaluation_Demo.

Every module in this package that persists data writes to a module-level
path constant (``feedback_store.FEEDBACK_STORE_PATH``,
``evaluator_runner.METRIC_RESULTS_PATH``/``METRIC_FAILURES_PATH``,
``metric_proposer.CANDIDATE_METRICS_PATH``/``REJECTED_METRICS_PATH``,
``run_pipeline.RUNS_DIR``). Those
constants are read at call time, so redirecting them here is enough to make
every write -- and every default read, e.g. ``feedback_store.load_feedback()``
with no argument, which ``evaluator_runner.run_evaluation`` relies on -- land
in a throwaway location instead of the real ``output/`` data files.

The autouse fixture below points all of these constants at a unique scratch
directory under ``output/_test_scratch/<uuid>/`` for the duration of each
test, then removes it. The scratch directory lives inside ``output/`` on
purpose: ``persistence.resolve_output_path`` only permits writes under
``output/``, and the containment rule must keep holding during tests too.
This applies to the default suite and to ``LIVE_AWS_TESTS=1`` runs alike.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

import evaluator_runner  # noqa: E402
import feedback_store  # noqa: E402
import metric_proposer  # noqa: E402
import run_pipeline  # noqa: E402
from persistence import OUTPUT_DIR  # noqa: E402

TEST_SCRATCH_ROOT = OUTPUT_DIR / "_test_scratch"


@pytest.fixture(autouse=True)
def _isolate_output_data_paths(monkeypatch):
    """Redirect every persisted-data path constant to a per-test scratch dir.

    Yields the scratch directory ``Path`` (unused by most tests, but handy
    for any test that wants to inspect what was written).
    """
    scratch_dir = TEST_SCRATCH_ROOT / uuid.uuid4().hex
    scratch_dir.mkdir(parents=True, exist_ok=True)

    redirects = {
        (feedback_store, "FEEDBACK_STORE_PATH"): scratch_dir / "feedback" / "feedback_store.jsonl",
        (evaluator_runner, "METRIC_RESULTS_PATH"): scratch_dir / "results" / "metric_results.jsonl",
        (evaluator_runner, "METRIC_FAILURES_PATH"): scratch_dir / "results" / "metric_results.failures.jsonl",
        (metric_proposer, "CANDIDATE_METRICS_PATH"): scratch_dir / "metrics" / "candidate_metrics.jsonl",
        (metric_proposer, "REJECTED_METRICS_PATH"): scratch_dir / "metrics" / "candidate_metrics.rejected.jsonl",
        (run_pipeline, "RUNS_DIR"): scratch_dir / "runs",
    }
    for (module, attr), target in redirects.items():
        # Absolute string paths: persistence's write helpers still route them
        # through resolve_output_path, which verifies they sit under output/.
        monkeypatch.setattr(module, attr, str(target))

    try:
        yield scratch_dir
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        # Drop the scratch root too once no other test is using it.
        try:
            TEST_SCRATCH_ROOT.rmdir()
        except OSError:
            pass
