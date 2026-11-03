"""Command-line entry point (`judge-check`). Subcommands are added as the package grows."""

from __future__ import annotations

import argparse

from judge_check import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="judge-check", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    return 0
