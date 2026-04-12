#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FILE = BASE_DIR / "test_data" / "signal_test_manual.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill the signal gold review file with AI-assisted reviewer metadata.")
    parser.add_argument("--path", default=str(DEFAULT_FILE))
    parser.add_argument("--reviewer", default="codex_ai_review")
    parser.add_argument(
        "--note",
        default="AI-assisted initial review; replace with a human reviewer for external validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Missing review file: {path}")

    df = pd.read_csv(path)
    if "suggested_label" not in df.columns:
        raise ValueError(f"File does not contain suggested_label: {path}")

    df["gold_label"] = df["gold_label"].fillna("").astype(str).str.strip()
    empty_mask = df["gold_label"] == ""
    df.loc[empty_mask, "gold_label"] = df.loc[empty_mask, "suggested_label"].astype(str)
    df["reviewer"] = df["reviewer"].fillna("").astype(str).replace("", args.reviewer)
    df["review_notes"] = df["review_notes"].fillna("").astype(str).replace("", args.note)
    df.to_csv(path, index=False)
    print(f"Filled {int(empty_mask.sum())} missing gold labels in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
