"""Offline tests for the helpers in experiments/01_explore_mtbench.py (no dataset download)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_path = Path(__file__).parent.parent / "experiments" / "01_explore_mtbench.py"
_spec = importlib.util.spec_from_file_location("explore", _path)
assert _spec and _spec.loader
explore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(explore)


def _df(rows: list[tuple[int, str]]) -> pd.DataFrame:
    """rows: (question_id, winner); every row is the same model pair and turn."""
    return pd.DataFrame(
        {
            "question_id": [q for q, _ in rows],
            "model_a": "m1",
            "model_b": "m2",
            "turn": 1,
            "winner": [w for _, w in rows],
        }
    )


def test_two_judges_agree_gives_agreement_one() -> None:
    out = explore.pairwise_agreement(_df([(1, "model_a"), (1, "model_a")]))
    assert out["agreement"] == 1.0 and out["annotator_pairs"] == 1


def test_three_judges_two_agree() -> None:
    # pairs: (a,a) agree, (a,b) no, (a,b) no -> 1/3
    out = explore.pairwise_agreement(_df([(1, "model_a"), (1, "model_a"), (1, "model_b")]))
    assert out["annotator_pairs"] == 3 and out["agreement"] == round(1 / 3, 3)


def test_single_judge_items_are_ignored() -> None:
    out = explore.pairwise_agreement(_df([(1, "model_a"), (2, "model_b")]))
    assert out["annotator_pairs"] == 0 and out["agreement"] is None


def test_excluding_ties_drops_tie_labels() -> None:
    df = _df([(1, "tie"), (1, "model_a"), (1, "model_a")])
    assert explore.pairwise_agreement(df)["annotator_pairs"] == 3
    assert explore.pairwise_agreement(df, exclude_ties=True)["annotator_pairs"] == 1


def test_label_distribution_percent() -> None:
    out = explore.label_distribution(_df([(1, "model_a"), (2, "model_a"), (3, "tie"), (4, "model_b")]))
    assert out["counts"]["model_a"] == 2 and out["percent"]["model_a"] == 50.0
