"""Model/config constants for Agent_Health_Evaluation_Demo.

This module re-exports ``PEER_CONFIGS`` and ``AGENT_MODEL_ID`` from the
sibling module's ``model_config.py`` (see
``Workload Specific Evaluations/Multiagent Shared Context Evaluation/model_config.py``)
by loading that file directly via ``importlib`` under a distinct internal
module name. This is a read-only Python import (``exec_module`` only
executes the sibling file to populate its namespace; the sibling's file is
never opened in write mode and its contents are never copied into this
file) — it avoids relying on ``sys.path`` + a bare ``from model_config
import ...`` because this file is itself named ``model_config.py``, which
would otherwise collide with the sibling's identically-named module under
certain working-directory/``sys.path`` orderings (see comment below).

The sibling directory is located relative to this file's own location
(``Path(__file__).resolve().parent``), not the current working directory,
so this module behaves the same regardless of where it is run from.

Requirements: 1.1, 7.1
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate and import the sibling module (read-only)
# ---------------------------------------------------------------------------

# This package's own directory, e.g.
# ".../Workload Specific Evaluations/Agent Health Evaluation Demo"
_PACKAGE_DIR = Path(__file__).resolve().parent

# The sibling module's directory, resolved relative to this file's location.
_SIBLING_DIR = (
    _PACKAGE_DIR.parent / "Multiagent Shared Context Evaluation"
).resolve()

_SIBLING_MODEL_CONFIG_PATH = _SIBLING_DIR / "model_config.py"

# Loaded by file path under a distinct module name (never the bare name
# "model_config") and cached in sys.modules under that distinct name.
#
# A plain ``sys.path.insert(0, str(_SIBLING_DIR)); from model_config import
# ...`` is ambiguous here: this file is itself named ``model_config.py`` and
# frequently gets imported (or executed) while *this* package's own
# directory is already first on ``sys.path`` (e.g. whenever a sibling
# script in this same package directory is run directly, or during `python
# -c` from this directory). In that situation Python has already bound the
# bare name "model_config" in ``sys.modules`` to *this* partially
# initialized file before the ``from model_config import ...`` line below
# would run, so the import resolves back to this file instead of the
# sibling's and raises a circular-import ``ImportError``. Loading by
# explicit file path under a distinct internal name sidesteps the name
# collision entirely, regardless of ``sys.path`` ordering or working
# directory.
_SIBLING_MODULE_NAME = "_agent_health_demo_sibling_model_config"
if _SIBLING_MODULE_NAME in sys.modules:
    _sibling_model_config = sys.modules[_SIBLING_MODULE_NAME]
else:
    _spec = importlib.util.spec_from_file_location(
        _SIBLING_MODULE_NAME, _SIBLING_MODEL_CONFIG_PATH)
    _sibling_model_config = importlib.util.module_from_spec(_spec)
    sys.modules[_SIBLING_MODULE_NAME] = _sibling_model_config
    _spec.loader.exec_module(_sibling_model_config)  # read-only: just executes the sibling file

# Imported, never copied: the sibling's model_config.py stays the single
# source of truth for these values.
PEER_CONFIGS = _sibling_model_config.PEER_CONFIGS
AGENT_MODEL_ID = _sibling_model_config.AGENT_MODEL_ID

# ---------------------------------------------------------------------------
# This demo's own constants
# ---------------------------------------------------------------------------

# Defaults to the sibling's AGENT_MODEL_ID so Phase 1's metric proposal LLM
# calls use the same model family as the peer agents by default, without
# hardcoding a possibly-stale duplicate of that value.
METRIC_PROPOSER_MODEL_ID = AGENT_MODEL_ID

CLOUDWATCH_NAMESPACE = "AgentHealthDemo"

# Same AWS_REGION / us-west-2 convention used by the sibling module (see its
# model_config.py-adjacent scripts, e.g. run_behavior_live.py).
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")


if __name__ == "__main__":
    assert PEER_CONFIGS, "PEER_CONFIGS should be non-empty"
    assert AGENT_MODEL_ID, "AGENT_MODEL_ID should be non-empty"
    assert METRIC_PROPOSER_MODEL_ID == AGENT_MODEL_ID, (
        "METRIC_PROPOSER_MODEL_ID should default to the sibling's "
        "AGENT_MODEL_ID"
    )
    assert CLOUDWATCH_NAMESPACE == "AgentHealthDemo"

    print("PEER_CONFIGS:", [name for name, _ in PEER_CONFIGS])
    print("AGENT_MODEL_ID:", AGENT_MODEL_ID)
    print("METRIC_PROPOSER_MODEL_ID:", METRIC_PROPOSER_MODEL_ID)
    print("CLOUDWATCH_NAMESPACE:", CLOUDWATCH_NAMESPACE)
    print("AWS_REGION:", AWS_REGION)
    print("Sanity check passed.")
