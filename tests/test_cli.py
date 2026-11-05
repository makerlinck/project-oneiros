"""CLI smoke test (SANDBOX.md §10)."""

from __future__ import annotations

from pathlib import Path

from life_sandbox.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_runs_one_tick(capsys):
    main(["--config", str(ROOT / "configs" / "world" / "default.toml"), "--ticks", "1"])
    out = capsys.readouterr().out
    assert "state hash:" in out
    assert "ticks: 1" in out
