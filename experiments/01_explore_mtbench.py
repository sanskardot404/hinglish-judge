"""Experiment 01: explore lmsys/mt_bench_human_judgments (English, human split).

Questions answered:
  1. What does the label distribution look like (model_a / model_b / tie)?
  2. How often do two human annotators agree when they judged the same pair?
  3. What do a few rows look like?

Run:  python experiments/01_explore_mtbench.py
Output: results/01_mtbench_summary.json and results/01_mtbench_summary.md
Deterministic: example rows use a fixed seed; the dataset is a fixed public snapshot.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

DATASET = "lmsys/mt_bench_human_judgments"
SEED = 0
# One "item" = one pair of answers to one question in one turn. Several humans may judge it.
ITEM_KEY = ["question_id", "model_a", "model_b", "turn"]


def label_distribution(df: pd.DataFrame) -> dict[str, Any]:
    counts = df["winner"].value_counts()
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "percent": {k: round(100 * v / len(df), 1) for k, v in counts.items()},
    }


def pairwise_agreement(df: pd.DataFrame, exclude_ties: bool = False) -> dict[str, Any]:
    """Raw agreement between humans who judged the *same item*.

    For an item with k labels there are k*(k-1)/2 annotator pairs; a pair agrees if both
    gave the same label. We pool all pairs over all items. This is plain agreement, NOT
    kappa: kappa also corrects for agreement expected by chance (you write that one).
    `chance_agreement` is that chance level, sum(p_i^2) over label shares p_i, for context.
    """
    data = df[df["winner"] != "tie"] if exclude_ties else df
    agree = total = items = 0
    for _, labels in data.groupby(ITEM_KEY)["winner"]:
        k = len(labels)
        if k < 2:
            continue
        items += 1
        total += k * (k - 1) // 2
        agree += sum(c * (c - 1) // 2 for c in Counter(labels).values())
    shares = data["winner"].value_counts(normalize=True)
    return {
        "items_with_2plus_judges": items,
        "annotator_pairs": total,
        "agreement": round(agree / total, 3) if total else None,
        "chance_agreement": round(float((shares**2).sum()), 3),
    }


def first_user_prompt(conversation: Any) -> str:
    return str(conversation[0]["content"])


def example_rows(df: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    rows = []
    for _, r in df.sample(n=n, random_state=SEED).iterrows():
        rows.append(
            {
                "question_id": int(r["question_id"]),
                "turn": int(r["turn"]),
                "model_a": r["model_a"],
                "model_b": r["model_b"],
                "winner": r["winner"],
                "judge": r["judge"],
                "prompt": first_user_prompt(r["conversation_a"])[:200],
            }
        )
    return rows


def summarise(df: pd.DataFrame) -> dict[str, Any]:
    per_item = df.groupby(ITEM_KEY).size()
    return {
        "dataset": DATASET,
        "split": "human",
        "rows": len(df),
        "distinct_questions": int(df["question_id"].nunique()),
        "distinct_items": int(len(per_item)),
        "judges_per_item": {str(k): int(v) for k, v in per_item.value_counts().sort_index().items()},
        "annotators": {k: int(v) for k, v in df["judge"].value_counts().items()},
        "label_distribution": label_distribution(df),
        "label_distribution_by_turn": {
            int(t): label_distribution(g) for t, g in df.groupby("turn")
        },
        "agreement_with_ties": pairwise_agreement(df),
        "agreement_without_ties": pairwise_agreement(df, exclude_ties=True),
        "examples": example_rows(df),
    }


def to_markdown(s: dict[str, Any]) -> str:
    a, b = s["agreement_with_ties"], s["agreement_without_ties"]
    lines = [
        "# Experiment 01: MT-Bench human judgments (English)",
        "",
        f"Dataset `{s['dataset']}`, split `{s['split']}`: {s['rows']} rows, "
        f"{s['distinct_questions']} questions, {s['distinct_items']} distinct items "
        "(question x model pair x turn).",
        "",
        "## Label distribution",
        "",
        "| winner | count | % |",
        "|---|---|---|",
    ]
    for k, v in s["label_distribution"]["counts"].items():
        lines.append(f"| {k} | {v} | {s['label_distribution']['percent'][k]} |")
    lines += [
        "",
        "## Human-human agreement (raw, not kappa)",
        "",
        "| labels used | items with 2+ judges | annotator pairs | agreement | chance level |",
        "|---|---|---|---|---|",
        f"| with ties | {a['items_with_2plus_judges']} | {a['annotator_pairs']} | "
        f"{a['agreement']} | {a['chance_agreement']} |",
        f"| ties removed | {b['items_with_2plus_judges']} | {b['annotator_pairs']} | "
        f"{b['agreement']} | {b['chance_agreement']} |",
        "",
        "Chance level = agreement expected if annotators labelled independently with the same "
        "label shares. Agreement only means something relative to it.",
        "",
        "## Judges per item",
        "",
        f"{s['judges_per_item']}  (number of humans -> number of items)",
        "",
        "## Annotators (rows per judge id)",
        "",
        f"{s['annotators']}",
        "",
        "## Example rows (seed 0)",
        "",
    ]
    for e in s["examples"]:
        lines.append(
            f"- q{e['question_id']} turn {e['turn']}: {e['model_a']} vs {e['model_b']} -> "
            f"**{e['winner']}** (by {e['judge']}). Prompt: {e['prompt']!r}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    from datasets import load_dataset  # imported here so tests can import this file offline

    df = load_dataset(DATASET, split="human").to_pandas()
    summary = summarise(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "01_mtbench_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = to_markdown(summary)
    (args.out_dir / "01_mtbench_summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
