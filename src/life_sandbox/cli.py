"""Command line: one world run (SANDBOX.md §10)."""

from __future__ import annotations

import argparse
import time

from life_sandbox.config import Config
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.step import run
from life_sandbox.world.init import initial_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="life-sandbox", description="Oneiros life sandbox")
    parser.add_argument("--config", default="configs/world/default.toml")
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    if args.seed is not None or args.ticks is not None:
        run_cfg = cfg.run
        cfg = cfg.replaced(
            run=run_cfg.__class__(
                seed=run_cfg.seed if args.seed is None else args.seed,
                ticks=run_cfg.ticks if args.ticks is None else args.ticks,
            )
        )
    cfg.validate()
    state = initial_state(cfg)
    ticks = int(cfg.run.ticks)
    started = time.perf_counter()
    result = run(state, cfg, ticks)
    elapsed = time.perf_counter() - started
    rate = ticks / elapsed if elapsed > 0 else float("inf")
    print("config hash: " + cfg.config_hash())
    print(
        "seed: "
        + str(cfg.run.seed)
        + "  ticks: "
        + str(ticks)
        + "  state hash: "
        + state_hash(result.state)
    )
    print("elapsed: " + format(elapsed, ".2f") + "s  (" + format(rate, ".1f") + " ticks/s)")
    if result.counts:
        print(
            "events: "
            + ", ".join(name + "=" + str(value) for name, value in sorted(result.counts.items()))
        )
    else:
        print("events: none")
    print(
        "ledger: inflow="
        + format(float(result.inflow.sum()), ".3e")
        + " outflow="
        + format(float(result.outflow.sum()), ".3e")
        + " dropped_events="
        + str(result.dropped_events)
    )


if __name__ == "__main__":
    main()
