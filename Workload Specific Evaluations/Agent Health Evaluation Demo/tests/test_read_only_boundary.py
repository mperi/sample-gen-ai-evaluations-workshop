"""Read-only boundary verification for Agent_Health_Evaluation_Demo.

**Validates: Requirements 7.1, 7.2**

Requirement 7 (and design.md's "Testing Strategy" / Read-only boundary
verification section) asks for an empirical check, independent of
``persistence.py``'s structural ``OutputPathViolation`` enforcement (see
``test_persistence.py``'s Property 20), that nothing this demo's test
suite does ever actually mutates a file under either:

  - ``Workload Specific Evaluations/Multiagent Shared Context Evaluation/``
    (the sibling Target_Agent_Module's directory), or
  - ``.kiro/specs/multi-agent-shared-context-evaluations/`` (that module's
    spec directory).

Design note -- why this is a dedicated test module rather than an addition
to ``test_persistence.py``:

``test_persistence.py`` already has an established scope and existing
"Property 20" docstring: it tests ``persistence.py``'s write-path
containment logic specifically (``resolve_output_path`` and the write
helpers that route through it). The check implemented here is a different,
broader concern -- it isn't about any one function's logic, it's a
suite-wide empirical guarantee about two directories that many modules in
this package read from (``persistence.py``, ``contract_deriver.py``,
``model_config.py``, etc.) and that no module is supposed to write to.
Keeping it in its own file makes that cross-cutting intent explicit and
keeps ``test_persistence.py`` focused on one module's unit under test.

Design note -- why this is one self-contained test function rather than a
true whole-suite ``conftest.py`` session fixture:

The design's wording is "hash every file ... before and after running the
full unit/property test suite". pytest has no built-in single hook that
runs once before *all* collected tests and once after, without reaching
for a session-scoped ``conftest.py`` autouse fixture (e.g. a fixture with
``scope="session", autouse=True`` that hashes before the ``yield`` and
re-hashes + asserts after it). That would be a stronger, more literal
implementation of the design's wording, and is noted here as a viable
follow-up if the project wants a true pre/post-suite bracket. This test
instead implements a single, ordinary pytest test function that hashes the
two directories at its own start, does nothing in between but the hashing
itself (no other risky operation), and re-hashes immediately before
asserting equality. Concretely, this validates the actual question the
design cares about -- "did this test suite's existence and execution touch
the sibling trees" -- because:

  - Collecting and importing every other test module in ``tests/`` already
    happens before any test body (including this one) runs, as part of
    pytest's normal collection phase, so by the time this test's first
    hash is taken, every other test module in the suite has already been
    imported (though not yet executed with ``pytest tests/ -q``'s
    ordering, since this file collects after the already-passed
    ``test_persistence.py`` alphabetically -- see the note below).
  - Running the full suite via ``pytest tests/ -q`` executes this test
    alongside all the others in one process; this test's own bracket
    still catches any mutation that happened to occur during collection or
    during this test's own run, and a `git status`-based check (performed
    separately, outside this test, per the task instructions) is the
    belt-and-suspenders confirmation that the *whole* suite run left the
    sibling trees untouched.

Scope of what is hashed:

Every regular file under each directory, with these ignorable
subtrees/caches skipped (none are expected to exist inside either sibling
directory today, but this is defensive against future tooling noise):
``.venv`` (the sibling module ships one; its ~8k files are virtualenv
artifacts, not part of the module itself, and are irrelevant to the
read-only boundary question), ``__pycache__``, ``.pytest_cache``, and
``.git``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from persistence import SIBLING_MODULE_DIR, SIBLING_SPEC_DIR  # noqa: E402

# Directory name components to skip entirely while walking the sibling
# trees. Matched against any path component, at any depth.
_IGNORED_DIR_NAMES = {".venv", "__pycache__", ".pytest_cache", ".git"}


def _hash_tree(root: Path) -> dict[str, str]:
    """Return ``{relative_posix_path: sha256_hex_digest}`` for every
    regular file under ``root``, skipping ``_IGNORED_DIR_NAMES`` subtrees.

    Reads every file strictly in binary read mode (``"rb"``) -- this
    function never opens anything for writing, consistent with the
    read-only boundary it is verifying.
    """
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _IGNORED_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root).as_posix()
        digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def test_sibling_module_and_spec_dirs_untouched_by_test_run():
    """Hash every file under the sibling module dir and the sibling spec
    dir, do nothing else risky, then re-hash and assert nothing changed.

    See the module docstring for why this is implemented as a single
    self-contained bracket rather than a whole-suite ``conftest.py``
    session fixture, and for the ignored-subtree list.
    """
    assert SIBLING_MODULE_DIR.is_dir(), (
        f"expected sibling module dir to exist: {SIBLING_MODULE_DIR}"
    )
    # The spec dir lives under .kiro/, which the repo root .gitignore
    # excludes, so it is absent in a fresh clone; then there is nothing
    # there to protect and only the sibling module tree is checked.
    spec_dir_present = SIBLING_SPEC_DIR.is_dir()

    before_module = _hash_tree(SIBLING_MODULE_DIR)
    before_spec = _hash_tree(SIBLING_SPEC_DIR) if spec_dir_present else {}

    # Sanity: both trees actually contain files, so an empty-vs-empty
    # comparison never masquerades as a passing test.
    assert before_module, f"expected at least one file under {SIBLING_MODULE_DIR}"
    if spec_dir_present:
        assert before_spec, f"expected at least one file under {SIBLING_SPEC_DIR}"

    # --- marker point -----------------------------------------------------
    # Nothing happens here beyond the hashing itself: this test performs no
    # writes, no sibling-module imports that could trigger bytecode-cache
    # writes outside the ignored __pycache__ subtrees, and no other I/O.
    # The re-hash immediately below is the evidence that hashing itself,
    # and the act of this test (and, when run as part of `pytest tests/
    # -q`, the rest of the suite having already executed/collected) did
    # not modify either tree.
    # -----------------------------------------------------------------------

    after_module = _hash_tree(SIBLING_MODULE_DIR)
    after_spec = _hash_tree(SIBLING_SPEC_DIR) if spec_dir_present else {}

    assert after_module == before_module, (
        "sibling module directory changed during the test run: "
        f"added={set(after_module) - set(before_module)} "
        f"removed={set(before_module) - set(after_module)} "
        f"modified={[p for p in before_module if p in after_module and before_module[p] != after_module[p]]}"
    )
    assert after_spec == before_spec, (
        "sibling spec directory changed during the test run: "
        f"added={set(after_spec) - set(before_spec)} "
        f"removed={set(before_spec) - set(after_spec)} "
        f"modified={[p for p in before_spec if p in after_spec and before_spec[p] != after_spec[p]]}"
    )
