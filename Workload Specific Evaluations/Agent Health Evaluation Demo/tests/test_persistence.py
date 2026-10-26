"""Property-based tests for persistence.py's write-path containment.

**Property 20: Every persistence function's write target is contained
within this demo's own output directory**

Validates: Requirements 7.1, 7.2, 7.3, 7.4

These tests only exercise path arithmetic (``resolve_output_path``) and the
write helpers that route through it (``write_json``, ``append_jsonl``).
They never read or write any file actually belonging to the sibling module
or its spec directory -- adversarial inputs are constructed so that their
*resolved* path would land there, and the test asserts that this is
rejected before any I/O against that location occurs.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

import persistence  # noqa: E402
from persistence import (  # noqa: E402
    OUTPUT_DIR,
    SIBLING_MODULE_DIR,
    SIBLING_SPEC_DIR,
    OutputPathViolation,
    append_jsonl,
    resolve_output_path,
    write_json,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Plain, well-behaved path segments: a single filename-safe ASCII component
# with no separators and no ".." traversal tricks. Restricted to ASCII
# letters/digits so generated directories stay easy to recognize and clean
# up as test fixtures (benign paths legitimately create directories under
# output/, mirroring resolve_output_path's own mkdir behavior).
_safe_segment = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), max_codepoint=0x7A),
    min_size=1,
    max_size=12,
).filter(lambda s: s not in (".", "..") and s.strip() != "")

# A benign relative path: 1-3 safe segments joined under output/, guaranteed
# to stay inside OUTPUT_DIR.
benign_relative_path = st.lists(_safe_segment, min_size=1, max_size=3)

# Number of ".." segments used to try to climb out of output/. OUTPUT_DIR is
# PACKAGE_DIR/output, so one ".." reaches PACKAGE_DIR, two reaches its
# parent ("Workload Specific Evaluations/"), etc. We use enough levels to
# comfortably reach the sibling module directory and beyond.
_escape_depth = st.integers(min_value=1, max_value=6)


def _relative_escape_parts(depth: int, target_segments: tuple[str, ...]) -> tuple[str, ...]:
    """Build a ``..``-climbing relative path of exactly ``depth`` ups,
    followed by ``target_segments``, intended to escape ``output/``."""
    return (("..",) * depth) + target_segments


# Adversarial path-part tuples, each one a *relative* parts tuple fed to
# resolve_output_path(*parts). Includes:
#  - ".." climbs of varying depth with a benign tail
#  - ".." climbs explicitly aimed at the sibling module dir by name
#  - ".." climbs explicitly aimed at the sibling spec dir by name
adversarial_relative_parts = st.one_of(
    st.builds(
        lambda depth, tail: _relative_escape_parts(depth, tuple(tail)),
        _escape_depth,
        st.lists(_safe_segment, min_size=0, max_size=2),
    ),
    st.builds(
        lambda tail: ("..", "Multiagent Shared Context Evaluation", *tail),
        st.lists(_safe_segment, min_size=0, max_size=2),
    ),
    st.builds(
        lambda tail: (
            "..",
            "..",
            ".kiro",
            "specs",
            "multi-agent-shared-context-evaluations",
            *tail,
        ),
        st.lists(_safe_segment, min_size=0, max_size=2),
    ),
)

# Adversarial *absolute* paths: either directly inside the sibling module
# dir, directly inside the sibling spec dir, or some other absolute path
# entirely outside OUTPUT_DIR (e.g. a filesystem root-level path).
adversarial_absolute_parts = st.one_of(
    st.builds(
        lambda tail: (str(SIBLING_MODULE_DIR / Path(*tail)) if tail else str(SIBLING_MODULE_DIR),),
        st.lists(_safe_segment, min_size=0, max_size=2),
    ),
    st.builds(
        lambda tail: (str(SIBLING_SPEC_DIR / Path(*tail)) if tail else str(SIBLING_SPEC_DIR),),
        st.lists(_safe_segment, min_size=0, max_size=2),
    ),
    st.builds(
        lambda tail: ("/" + "/".join(("tmp", *tail)),),
        st.lists(_safe_segment, min_size=1, max_size=3),
    ),
)

adversarial_parts = st.one_of(adversarial_relative_parts, adversarial_absolute_parts)


def _is_descendant(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _clean_output_dir_fixtures():
    """Benign-path tests legitimately create directories/files under
    output/ as a side effect of resolve_output_path's own mkdir behavior.
    Snapshot output/'s top-level entries before the test and remove any
    new ones it creates afterward, so the test suite never leaves fixture
    clutter behind in a directory that is also used for real artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before = set(OUTPUT_DIR.iterdir())
    yield
    after = set(OUTPUT_DIR.iterdir())
    for extra in after - before:
        if extra.is_dir():
            shutil.rmtree(extra, ignore_errors=True)
        else:
            extra.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Property: benign paths under output/ always succeed and stay contained
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(parts=benign_relative_path)
def test_benign_paths_resolve_under_output_dir(parts):
    """Any path whose resolved target IS a descendant of output/ succeeds
    and resolve_output_path returns a path rooted under OUTPUT_DIR."""
    resolved = resolve_output_path(*parts)
    assert _is_descendant(resolved, OUTPUT_DIR)
    assert resolved.is_absolute()


# ---------------------------------------------------------------------------
# Property: adversarial paths that escape output/ always raise
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(parts=adversarial_parts)
def test_escaping_paths_always_raise(parts):
    """Any path whose resolved target is NOT a descendant of output/ --
    especially ones landing in the sibling module dir or sibling spec dir
    -- raises OutputPathViolation and never returns a path."""
    # Sanity: confirm this generated example truly would escape OUTPUT_DIR,
    # so the test is exercising the intended adversarial case and not a
    # coincidentally-benign one.
    candidate = (OUTPUT_DIR / Path(*parts)).resolve()
    assume_escapes = not _is_descendant(candidate, OUTPUT_DIR)
    if not assume_escapes:
        # A generated combination of safe segments happened to stay inside
        # output/ (e.g. all ".." segments were absorbed and the tail is
        # benign) -- this is not an adversarial case, skip it.
        return

    try:
        resolve_output_path(*parts)
    except OutputPathViolation:
        pass
    else:
        raise AssertionError(
            f"Expected OutputPathViolation for escaping parts={parts!r} "
            f"(resolved to {candidate})"
        )


# ---------------------------------------------------------------------------
# Property: write helpers never succeed in writing outside output/
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(parts=adversarial_parts, payload=st.dictionaries(_safe_segment, st.integers(), max_size=3))
def test_write_json_never_escapes_output_dir(parts, payload):
    """write_json routes through resolve_output_path, so it must never
    create a file outside output/ for an escaping path."""
    candidate = (OUTPUT_DIR / Path(*parts)).resolve()
    if _is_descendant(candidate, OUTPUT_DIR):
        return  # not an adversarial case for this generated example

    pre_existed = candidate.exists()

    try:
        write_json(Path(*parts), payload)
    except OutputPathViolation:
        pass
    else:
        raise AssertionError(f"write_json did not reject escaping path parts={parts!r}")

    # No file was created (or altered) at the escaping location as a side
    # effect of the rejected write attempt.
    assert candidate.exists() == pre_existed


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(parts=adversarial_parts, record=st.dictionaries(_safe_segment, st.integers(), max_size=3))
def test_append_jsonl_never_escapes_output_dir(parts, record):
    """append_jsonl routes through resolve_output_path, so it must never
    create or modify a file outside output/ for an escaping path."""
    candidate = (OUTPUT_DIR / Path(*parts)).resolve()
    if _is_descendant(candidate, OUTPUT_DIR):
        return  # not an adversarial case for this generated example

    pre_existed = candidate.exists()
    pre_contents = candidate.read_bytes() if pre_existed and candidate.is_file() else None

    try:
        append_jsonl(Path(*parts), record)
    except OutputPathViolation:
        pass
    else:
        raise AssertionError(f"append_jsonl did not reject escaping path parts={parts!r}")

    assert candidate.exists() == pre_existed
    if pre_existed and candidate.is_file():
        assert candidate.read_bytes() == pre_contents


# ---------------------------------------------------------------------------
# Targeted examples pinning down the sibling-dir and sibling-spec-dir cases
# explicitly named in the design/requirements, independent of the property
# tests above (belt-and-suspenders, not relying solely on generation).
# ---------------------------------------------------------------------------

def test_relative_escape_into_sibling_module_dir_raises():
    try:
        resolve_output_path("..", "Multiagent Shared Context Evaluation", "model_config.py")
    except OutputPathViolation:
        pass
    else:
        raise AssertionError("expected OutputPathViolation")


def test_absolute_sibling_module_path_raises():
    try:
        resolve_output_path(str(SIBLING_MODULE_DIR / "model_config.py"))
    except OutputPathViolation:
        pass
    else:
        raise AssertionError("expected OutputPathViolation")


def test_absolute_sibling_spec_path_raises():
    try:
        resolve_output_path(str(SIBLING_SPEC_DIR / "tasks.md"))
    except OutputPathViolation:
        pass
    else:
        raise AssertionError("expected OutputPathViolation")


def test_benign_nested_path_succeeds_and_is_contained():
    resolved = resolve_output_path("contracts", "example_agent.json")
    assert _is_descendant(resolved, OUTPUT_DIR)


def test_no_parts_raises():
    try:
        resolve_output_path()
    except OutputPathViolation:
        pass
    else:
        raise AssertionError("expected OutputPathViolation for empty parts")
