from judge_check import __version__
from judge_check.cli import main


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_cli_runs_without_args() -> None:
    assert main([]) == 0
