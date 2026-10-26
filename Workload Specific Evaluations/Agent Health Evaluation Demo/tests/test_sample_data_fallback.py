"""Tests for ``load_review_records``' bundled-sample fallback.

A fresh git clone has no sibling ``review/review_cases.jsonl`` (the repo root
``.gitignore`` excludes that folder), so ``load_review_records`` falls back
to this package's git-tracked ``sample_data/review_cases.jsonl``. These
tests point ``persistence.SIBLING_MODULE_DIR`` at temp directories (missing
or populated with a tiny fixture) and never touch the real sibling module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

import persistence  # noqa: E402
from persistence import MULTI_AGENT_SENTINEL, OutputPathViolation  # noqa: E402

SAMPLE_IDS = [
    "live-brief-01-fastcasual",
    "live-brief-02-fastcasual-expansion",
    "live-brief-03-evfleet",
]
PEER_AGENTS = {"market_trends", "customer_insights", "strategy_synth"}


@pytest.fixture(autouse=True)
def _clear_force_env(monkeypatch):
    monkeypatch.delenv(persistence.USE_SAMPLE_DATA_ENV, raising=False)


def _assert_is_bundled_sample(records):
    assert [r.record_id for r in records] == SAMPLE_IDS
    for r in records:
        assert set(r.responses) == PEER_AGENTS
        assert all(text.strip() for text in r.responses.values())
        assert r.agent_name == MULTI_AGENT_SENTINEL
        assert r.brief_text
        assert r.expected_behavior


def test_sample_file_is_present_and_small():
    path = persistence.SAMPLE_REVIEW_CASES_PATH
    assert path.is_file()
    assert path.stat().st_size < 200_000
    assert len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]) == 3


def test_falls_back_to_sample_when_sibling_missing(tmp_path, monkeypatch, capsys):
    missing = (tmp_path / "no-such-sibling").resolve()
    monkeypatch.setattr(persistence, "SIBLING_MODULE_DIR", missing)

    records = persistence.load_review_records()

    _assert_is_bundled_sample(records)
    assert "using bundled sample" in capsys.readouterr().out
    assert not missing.exists()  # the fallback never creates sibling paths


def test_uses_sibling_when_present(tmp_path, monkeypatch, capsys):
    sibling = tmp_path / "sibling"
    (sibling / "review").mkdir(parents=True)
    row = {
        "case_id": "live-fixture-only",
        "expected_behavior": "normal_grounded_analysis",
        "brief_text": "Fixture brief.",
        "pipeline_output": "[market_trends]\nFixture response.",
    }
    (sibling / "review" / "review_cases.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(persistence, "SIBLING_MODULE_DIR", sibling.resolve())

    records = persistence.load_review_records()

    assert [r.record_id for r in records] == ["live-fixture-only"]
    assert records[0].responses == {"market_trends": "Fixture response."}
    assert records[0].agent_name == "market_trends"
    assert "bundled sample" not in capsys.readouterr().out


def test_env_var_forces_sample_even_when_sibling_present(tmp_path, monkeypatch):
    sibling = tmp_path / "sibling"
    (sibling / "review").mkdir(parents=True)
    (sibling / "review" / "review_cases.jsonl").write_text(
        json.dumps({"case_id": "live-fixture-only", "pipeline_output": ""}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(persistence, "SIBLING_MODULE_DIR", sibling.resolve())
    monkeypatch.setenv(persistence.USE_SAMPLE_DATA_ENV, "1")

    _assert_is_bundled_sample(persistence.load_review_records())


def test_traversal_guard_still_applies_with_missing_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "SIBLING_MODULE_DIR", (tmp_path / "no-such-sibling").resolve())
    with pytest.raises(OutputPathViolation):
        persistence.load_review_records("../outside.jsonl")
