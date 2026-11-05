"""Oneiros life sandbox (docs/SANDBOX.md).

One engine, all experiments are configuration: no arm names live in code,
only capability switches (SANDBOX.md §0, §12.1).
"""

__version__ = "0.1.0"


def main() -> None:
    """Console entry point (see pyproject.toml [project.scripts])."""
    from life_sandbox.cli import main as _main

    _main()
