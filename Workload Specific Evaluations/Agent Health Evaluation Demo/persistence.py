"""Shared persistence seam for Agent_Health_Evaluation_Demo.

This module is the architectural enforcement point for Requirement 7 (the
read-only boundary on the Target_Agent_Module): every write performed by any
other module in this package MUST be routed through ``resolve_output_path``
(directly, or indirectly via ``write_json``/``append_jsonl``), which rejects
any path that is not a descendant of this package's own ``output/``
directory. In particular it is impossible for a write resolved through this
module to land under:

  - ``Workload Specific Evaluations/Multiagent Shared Context Evaluation/``
  - ``.kiro/specs/multi-agent-shared-context-evaluations/``

Reads of the sibling module are allowed (the sibling module is a read-only
data/config source for this demo, see design.md's Architecture section), but
only through helpers in this module that open files strictly in ``"r"``
mode (``read_json``, ``read_jsonl``, ``read_sibling_file``) — there is no
write variant exposed for sibling paths, by design.

Requirements: 1.5, 2.4, 3.2, 3.5, 4.5, 7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from models import ReviewRecord

# ---------------------------------------------------------------------------
# Directory roots
# ---------------------------------------------------------------------------

# This package's own directory, e.g.
# ".../Workload Specific Evaluations/Agent Health Evaluation Demo"
PACKAGE_DIR = Path(__file__).resolve().parent

# This package's own output directory. All writes performed by this demo
# must resolve to a descendant of this path.
OUTPUT_DIR = PACKAGE_DIR / "output"

# The sibling Target_Agent_Module's directory. Read-only. Never written to
# by anything in this package.
SIBLING_MODULE_DIR = (
    PACKAGE_DIR.parent / "Multiagent Shared Context Evaluation"
).resolve()

# The sibling module's spec directory. Read-only. Never written to by
# anything in this package. Resolved relative to this file, independent of
# the caller's current working directory.
SIBLING_SPEC_DIR = (
    PACKAGE_DIR.parent.parent
    / ".kiro"
    / "specs"
    / "multi-agent-shared-context-evaluations"
).resolve()
# Bundled fallback for the sibling's review dataset. The repo root
# .gitignore excludes the sibling's review/ folder, so a fresh clone has no
# review/review_cases.jsonl; load_review_records() then reads this small,
# git-tracked sample (first 3 rows of the real file, copied verbatim).
SAMPLE_DATA_DIR = PACKAGE_DIR / "sample_data"
SAMPLE_REVIEW_CASES_PATH = SAMPLE_DATA_DIR / "review_cases.jsonl"
# Set to "1" to force the bundled sample even when the sibling file exists.
USE_SAMPLE_DATA_ENV = "AGENT_HEALTH_USE_SAMPLE_DATA"


class OutputPathViolation(ValueError):
    """Raised when a resolved path would escape this package's output/ dir.

    This is the concrete enforcement mechanism for Requirement 7: any
    attempt (accidental or malicious) to resolve a write target outside
    ``output/`` — including an attempt to reach into the sibling module's
    directory or its spec directory via ``..`` segments or an absolute
    path — raises this exception instead of performing the write.
    """


def resolve_output_path(*parts: str) -> Path:
    """Resolve ``parts`` to an absolute path rooted under this package's
    own ``output/`` directory, creating ``output/`` (and the resolved
    path's parent directories) if they do not yet exist.

    Every other write helper in this module (and, transitively, every
    other module in this package) MUST route its write target through
    this function.

    Args:
        *parts: path segments to join under ``output/``, e.g.
            ``resolve_output_path("contracts", "market_trends.json")``.

    Returns:
        The resolved, absolute ``Path``, guaranteed to be a descendant of
        ``OUTPUT_DIR``. Parent directories are created automatically
        (``parents=True, exist_ok=True``) so callers never need to create
        ``output/`` or its subdirectories themselves before writing.

    Raises:
        OutputPathViolation: if the resolved absolute path is not a
            descendant of ``OUTPUT_DIR`` (e.g. a ``parts`` sequence
            containing ``".."`` segments that escape ``output/``, or an
            absolute path pointing elsewhere — including anywhere under
            the sibling module's directory or its spec directory).
    """
    if not parts:
        raise OutputPathViolation("resolve_output_path() requires at least one path segment")

    candidate = (OUTPUT_DIR / Path(*parts)).resolve()

    try:
        candidate.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise OutputPathViolation(
            f"Refusing to write outside this package's output/ directory: "
            f"{candidate} is not a descendant of {OUTPUT_DIR.resolve()}"
        ) from None

    # Defense in depth: even if OUTPUT_DIR itself were ever misconfigured,
    # never allow a write into the sibling module or its spec directory.
    for forbidden in (SIBLING_MODULE_DIR, SIBLING_SPEC_DIR):
        try:
            candidate.relative_to(forbidden)
        except ValueError:
            continue
        raise OutputPathViolation(
            f"Refusing to write into the read-only sibling path: {candidate} "
            f"is a descendant of {forbidden}"
        )

    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


# ---------------------------------------------------------------------------
# Generic JSON helpers (single-object files, e.g. output/contracts/*.json)
# ---------------------------------------------------------------------------

def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write ``data`` as a single JSON object to ``path``.

    ``path`` is routed through ``resolve_output_path`` (its parts are
    passed through unchanged), so this call raises ``OutputPathViolation``
    if ``path`` would resolve outside this package's ``output/``
    directory. The parent directory is created automatically if absent.

    Args:
        path: a path (absolute, relative, or a bare filename) describing
            where under ``output/`` to write. Only the path's own
            components matter for containment — this is not mixed with
            the current working directory.
        data: a JSON-serializable dict.
    """
    resolved = resolve_output_path(*_path_parts(path))
    with open(resolved, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` verbatim (UTF-8) to ``path`` and return the resolved path.

    Routed through ``resolve_output_path`` exactly like ``write_json``, so
    it raises ``OutputPathViolation`` for any target outside ``output/``.
    Used for human-readable artifacts such as ``run_pipeline.py``'s
    ``output/runs/<timestamp>_summary.md``.
    """
    resolved = resolve_output_path(*_path_parts(path))
    with open(resolved, "w", encoding="utf-8") as fh:
        fh.write(text)
    return resolved


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a single JSON object from ``path``.

    Unlike the write helpers, ``read_json`` does not enforce output-dir
    containment: reads are permitted against any path, including sibling
    module paths, per the design's read-only boundary (the boundary
    constrains writes, not reads). This function always opens the file in
    ``"r"`` mode only — it never writes.

    Args:
        path: any readable path to a JSON file.

    Returns:
        The parsed JSON object as a dict.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Generic JSONL helpers (line-delimited files, e.g. feedback_store.jsonl)
# ---------------------------------------------------------------------------

def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single JSON-serialized ``record`` as one line to ``path``.

    ``path`` is routed through ``resolve_output_path``, so this raises
    ``OutputPathViolation`` if ``path`` would resolve outside this
    package's ``output/`` directory. The file (and any missing parent
    directories) is created automatically on first write.

    Args:
        path: a path describing where under ``output/`` to append.
        record: a JSON-serializable dict representing one line/record.
    """
    resolved = resolve_output_path(*_path_parts(path))
    with open(resolved, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record))
        fh.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all records from a line-delimited JSON (JSONL) file.

    Like ``read_json``, this does not enforce output-dir containment and
    always opens the file in ``"r"`` mode only, so it is safe to use
    against sibling module paths as well as this package's own output
    files. Blank lines are skipped. A missing file yields an empty list.

    Args:
        path: any readable path to a JSONL file.

    Returns:
        A list of parsed JSON objects, one per non-blank line, in file
        order.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Read-only sibling-module helpers
# ---------------------------------------------------------------------------

def read_sibling_file(relative_path: str) -> str:
    """Read the full text contents of a file under the sibling
    Target_Agent_Module's directory, strictly read-only.

    This is the only sanctioned way for this package to read a file from
    ``Workload Specific Evaluations/Multiagent Shared Context Evaluation/``.
    The file is always opened in ``"r"`` mode; there is intentionally no
    write counterpart for sibling paths anywhere in this module.

    Args:
        relative_path: a path relative to the sibling module's directory,
            e.g. ``"review/review_cases.jsonl"`` or
            ``"datasets/research_briefs.jsonl"``.

    Returns:
        The file's full text contents.

    Raises:
        OutputPathViolation: if ``relative_path`` would resolve outside
            the sibling module's directory (e.g. via ``..`` traversal or
            an absolute path), since this helper is only meant for
            reading files that actually belong to that module.
        FileNotFoundError: if the resolved file does not exist.
    """
    resolved = _resolve_sibling_path(relative_path)
    with open(resolved, "r", encoding="utf-8") as fh:
        return fh.read()


def _resolve_sibling_path(relative_path: str) -> Path:
    """Resolve ``relative_path`` under the sibling module directory,
    raising ``OutputPathViolation`` if it would escape that directory.
    Reads ``SIBLING_MODULE_DIR`` at call time so tests can monkeypatch it."""
    resolved = (SIBLING_MODULE_DIR / relative_path).resolve()
    try:
        resolved.relative_to(SIBLING_MODULE_DIR)
    except ValueError:
        raise OutputPathViolation(
            f"Refusing to read outside the sibling module directory: "
            f"{resolved} is not a descendant of {SIBLING_MODULE_DIR}"
        ) from None
    return resolved


# ---------------------------------------------------------------------------
# Read-only ReviewRecord loader (sibling review_cases.jsonl -> ReviewRecord)
# ---------------------------------------------------------------------------

# The marker that splits a combined pipeline_output string into per-agent
# blocks. capture_for_review.py (in the sibling module) builds this string
# as: "\n\n".join(f"[{agent_name}]\n{response_text}" for ...), i.e. each
# agent's block starts with "[agent_name]" on its own line.
_AGENT_BLOCK_RE = re.compile(r"^\[([^\]\n]+)\]\n", re.MULTILINE)

# Sentinel used for ReviewRecord.agent_name when a record's pipeline_output
# contains responses from more than one agent. models.py documents
# agent_name as designed primarily for the single-agent case; see
# parse_pipeline_output()'s docstring for the full rationale.
MULTI_AGENT_SENTINEL = "multi"


def parse_pipeline_output(pipeline_output: str) -> dict[str, str]:
    """Split a combined ``pipeline_output`` string into ``{agent_name:
    response_text}``.

    The sibling module's ``capture_for_review.py`` builds ``pipeline_output``
    as ``"\\n\\n".join(f"[{agent_name}]\\n{response_text}" for ...)`` — i.e.
    each agent's block is introduced by a ``[agent_name]`` marker on its own
    line, followed by that agent's full response text up to the next marker
    (or end of string). This function reverses that join.

    Args:
        pipeline_output: the raw ``pipeline_output`` field value from a
            ``review_cases.jsonl`` row.

    Returns:
        A dict mapping each agent name found to its response text (leading/
        trailing blank lines stripped). Returns an empty dict for an empty
        or marker-less input.
    """
    if not pipeline_output:
        return {}

    matches = list(_AGENT_BLOCK_RE.finditer(pipeline_output))
    if not matches:
        return {}

    responses: dict[str, str] = {}
    for i, match in enumerate(matches):
        agent_name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(pipeline_output)
        responses[agent_name] = pipeline_output[start:end].strip()
    return responses


def _load_research_briefs_by_id(
    relative_path: str = "datasets/research_briefs.jsonl",
) -> dict[str, dict[str, Any]]:
    """Best-effort, read-only load of ``{brief_id: row}`` from the
    sibling's research briefs dataset, for enriching ``ReviewRecord``s.

    ``row`` is the raw parsed JSON object for that brief (so callers can
    pull out ``facts`` and/or the brief's own ``expected_behavior``).

    Returns an empty dict (never raises) if the file is absent or
    malformed, since this enrichment is optional per the design (Review_UI
    loader description) — a missing or unreadable briefs file must not
    fail the primary ``load_review_records`` load.
    """
    try:
        text = read_sibling_file(relative_path)
    except (FileNotFoundError, OutputPathViolation):
        return {}

    rows_by_id: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        brief_id = row.get("id")
        if brief_id:
            rows_by_id[brief_id] = row
    return rows_by_id


def load_review_records(
    relative_path: str = "review/review_cases.jsonl",
    briefs_relative_path: str = "datasets/research_briefs.jsonl",
) -> list[ReviewRecord]:
    """Load ``ReviewRecord``s read-only from the sibling module's review
    dataset (Requirement 3.2, 3.3; design.md Review_UI component).

    Each row of the sibling's ``review/review_cases.jsonl`` is parsed as
    follows:

      - ``record_id``          <- row's ``case_id``
      - ``brief_text``         <- row's ``brief_text``
      - ``expected_behavior``  <- row's ``expected_behavior``, or ``None``
                                   if absent/empty
      - ``responses``          <- parsed out of row's ``pipeline_output``
                                   via ``parse_pipeline_output`` (splits on
                                   ``[agent_name]`` block markers)

    ``ReviewRecord.agent_name`` (a single required field) is set as
    follows, since a ``review_cases.jsonl`` row commonly contains responses
    from *all three* peer agents rather than just one:

      - If ``responses`` has exactly one agent, that agent's name is used.
      - If ``responses`` has more than one agent, ``MULTI_AGENT_SENTINEL``
        (``"multi"``) is used, and the real per-agent breakdown remains
        available via ``responses``' keys. This mirrors ``models.py``'s own
        docstring, which notes ``ReviewRecord.agent_name`` was designed
        primarily for the single-agent case, with multi-agent records
        expected to key off ``responses`` instead.
      - If ``responses`` is empty (unparseable ``pipeline_output``), the
        row's own ``case_id`` is reused as a last-resort, non-empty
        placeholder so ``agent_name`` is never blank.

    As a best-effort bonus enrichment, if the sibling's
    ``datasets/research_briefs.jsonl`` is present and readable, each
    record's ``expected_behavior`` is backfilled from the matching brief's
    own ``expected_behavior`` field when the row's own
    ``expected_behavior`` is missing, using the join key ``case_id`` with a
    leading ``"live-"`` prefix stripped (the prefix ``capture_for_review.py``
    adds when building ``case_id`` from a brief's ``id``, e.g.
    ``case_id == f"live-{brief.id}"``). This join was verified against the
    current dataset: all 10 ``review_cases.jsonl`` rows match a
    ``research_briefs.jsonl`` id after stripping ``"live-"``, though in
    practice every current row already carries its own non-empty
    ``expected_behavior``, so this fallback path is not exercised by today's
    data (documented here rather than silently assumed).

    Note on structured ``facts``: ``research_briefs.jsonl`` rows also carry
    a structured ``facts`` dict (e.g. market size, key players), which the
    design document mentions as available for enrichment. ``ReviewRecord``
    (see ``models.py``) has no ``facts`` field, so this loader does not
    attach it to the returned dataclass instances — doing so would require
    a model change, which is out of scope for this task. The join
    machinery (``_load_research_briefs_by_id``) already resolves the full
    brief row, including ``facts``, by id, so a future enrichment that adds
    a ``facts`` field to ``ReviewRecord`` (or a wrapper dataclass) can reuse
    it directly. The join is skipped silently (never raises) if the briefs
    file is absent, empty, or no row's ``case_id`` matches any brief id
    after stripping the prefix — enrichment is best-effort and must never
    fail the primary load.

    Args:
        relative_path: path to the review cases file, relative to the
            sibling module's directory.
        briefs_relative_path: path to the research briefs file, relative to
            the sibling module's directory, used only for the best-effort
            enrichment described above.

    Returns:
        A list of ``ReviewRecord``s, one per non-blank line in the source
        file, in file order.

    Fallback: if the sibling file is absent (e.g. a fresh git clone, where
    the root ``.gitignore`` excludes the sibling's ``review/`` folder), or
    ``AGENT_HEALTH_USE_SAMPLE_DATA=1`` is set, the bundled
    ``sample_data/review_cases.jsonl`` is read instead (``"r"`` mode) and a
    one-line notice is printed. The sibling remains the primary source.

    Raises:
        FileNotFoundError: if neither the sibling file nor the bundled
            sample exists.
        OutputPathViolation: if ``relative_path`` would resolve outside the
            sibling module's directory.
    """
    sibling_path = _resolve_sibling_path(relative_path)  # traversal guard
    force_sample = os.environ.get(USE_SAMPLE_DATA_ENV) == "1"
    if force_sample or not sibling_path.is_file():
        reason = (
            f"{USE_SAMPLE_DATA_ENV}=1" if force_sample
            else f"{sibling_path} not found"
        )
        print(
            f"[persistence] {reason}; using bundled sample "
            f"{SAMPLE_REVIEW_CASES_PATH.relative_to(PACKAGE_DIR)}"
        )
        with open(SAMPLE_REVIEW_CASES_PATH, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = read_sibling_file(relative_path)
    briefs_by_id = _load_research_briefs_by_id(briefs_relative_path)
    return parse_review_cases_text(text, briefs_by_id)


def parse_review_cases_text(
    text: str,
    briefs_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[ReviewRecord]:
    """Parse ``review_cases.jsonl`` text into ``ReviewRecord``s.

    Shared row-parsing logic for ``load_review_records`` (sibling file or
    bundled sample alike); see that function's docstring for the field
    mapping and the ``agent_name`` / enrichment rules. ``briefs_by_id`` is
    the optional ``{brief_id: row}`` map from
    ``_load_research_briefs_by_id``.
    """
    briefs_by_id = briefs_by_id or {}
    records: list[ReviewRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)

        case_id = row["case_id"]
        responses = parse_pipeline_output(row.get("pipeline_output", ""))

        if len(responses) == 1:
            agent_name = next(iter(responses))
        elif len(responses) > 1:
            agent_name = MULTI_AGENT_SENTINEL
        else:
            agent_name = case_id

        expected_behavior = row.get("expected_behavior") or None

        # Best-effort enrichment: only fills in a missing expected_behavior;
        # never overwrites data already present in review_cases.jsonl, and
        # never raises if the join key isn't present for this row.
        if expected_behavior is None:
            brief_id = case_id[len("live-"):] if case_id.startswith("live-") else case_id
            brief_row = briefs_by_id.get(brief_id)
            if brief_row is not None:
                expected_behavior = brief_row.get("expected_behavior") or None

        records.append(
            ReviewRecord(
                record_id=case_id,
                agent_name=agent_name,
                brief_text=row.get("brief_text", ""),
                responses=responses,
                expected_behavior=expected_behavior,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _path_parts(path: str | Path) -> tuple[str, ...]:
    """Normalize a caller-supplied path into path segments suitable for
    ``resolve_output_path(*parts)``.

    Accepts a bare filename (``"foo.json"``), a relative path with
    subdirectories (``"contracts/foo.json"``), a ``Path``, or an absolute
    path. In every case only the path's own components are used — the
    current working directory is never consulted — so containment is
    determined solely by where the path points, not by where the process
    happens to be running from.
    """
    p = Path(path)
    if p.is_absolute():
        # Use path relative to OUTPUT_DIR's anchor-independent parts so an
        # absolute path is still checked for containment by
        # resolve_output_path rather than silently joined under output/.
        return (str(p),)
    return p.parts


if __name__ == "__main__":
    # Quick sanity check (see also tests/test_persistence.py once added).
    print(f"PACKAGE_DIR        = {PACKAGE_DIR}")
    print(f"OUTPUT_DIR          = {OUTPUT_DIR}")
    print(f"SIBLING_MODULE_DIR  = {SIBLING_MODULE_DIR}")
    print(f"SIBLING_SPEC_DIR    = {SIBLING_SPEC_DIR}")

    # (a) Writing within output/ succeeds.
    write_json("contracts/_sanity_check.json", {"ok": True, "n": 1})
    round_tripped = read_json(resolve_output_path("contracts", "_sanity_check.json"))
    assert round_tripped == {"ok": True, "n": 1}, round_tripped
    print("[ok] write_json/read_json round-trip under output/ succeeded")

    append_jsonl("feedback/_sanity_check.jsonl", {"entry_type": "x", "n": 1})
    append_jsonl("feedback/_sanity_check.jsonl", {"entry_type": "x", "n": 2})
    lines = read_jsonl(resolve_output_path("feedback", "_sanity_check.jsonl"))
    assert lines == [{"entry_type": "x", "n": 1}, {"entry_type": "x", "n": 2}], lines
    print("[ok] append_jsonl/read_jsonl round-trip under output/ succeeded")

    # (b) A path that escapes output/ raises.
    escape_attempts = [
        ("relative ..", ("..", "Multiagent Shared Context Evaluation", "model_config.py")),
        ("absolute sibling path", (str(SIBLING_MODULE_DIR / "model_config.py"),)),
        ("absolute sibling spec path", (str(SIBLING_SPEC_DIR / "tasks.md"),)),
    ]
    for label, parts in escape_attempts:
        try:
            resolve_output_path(*parts)
        except OutputPathViolation:
            print(f"[ok] escape attempt correctly rejected: {label}")
        else:
            raise AssertionError(f"escape attempt NOT rejected: {label}")

    # (c) read_sibling_file can read an existing sibling file without error.
    readme_text = read_sibling_file("README.md")
    assert len(readme_text) > 0
    print("[ok] read_sibling_file read sibling README.md "
          f"({len(readme_text)} chars) without error")

    # Attempting to escape the sibling dir via read_sibling_file also raises.
    try:
        read_sibling_file("../Agent Health Evaluation Demo/persistence.py")
    except OutputPathViolation:
        print("[ok] read_sibling_file correctly rejected a path escaping the sibling dir")
    else:
        raise AssertionError("read_sibling_file did NOT reject an escaping path")

    # (d) load_review_records parses the real sibling review_cases.jsonl.
    review_records = load_review_records()
    assert len(review_records) >= 1, "expected at least one ReviewRecord"
    assert any(len(r.responses) > 0 for r in review_records), (
        "expected at least one ReviewRecord with a non-empty responses dict"
    )
    print(f"[ok] load_review_records loaded {len(review_records)} ReviewRecord(s) "
          f"from the sibling's review_cases.jsonl without error")
    sample = review_records[0]
    print(f"      sample: record_id={sample.record_id!r} agent_name={sample.agent_name!r} "
          f"agents_in_responses={sorted(sample.responses.keys())!r} "
          f"expected_behavior={sample.expected_behavior!r}")

    print("\nAll sanity checks passed.")
