#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LABELED_FILE = BASE_DIR / "data" / "labeled" / "weak_labeled.csv"
OUTPUT_FILE = BASE_DIR / "test_data" / "signal_test_manual.csv"
SIGNAL_LABELS = [
    "NEUTRAL",
    "DEMAND_UNMET",
    "DEMAND_PRESENT",
    "SUPPLY_SIGNAL",
    "COMPETITION_HIGH",
    "COMPLAINT",
    "TREND",
]


def build_candidate_pool(source_path: Path, per_class: int) -> pd.DataFrame:
    df = pd.read_csv(source_path).dropna(subset=["text", "signal"]).copy()
    df = df[df["signal"].isin(SIGNAL_LABELS)].copy()
    df = df.sort_values(["signal", "confidence"], ascending=[True, False])
    grouped = []
    for label in SIGNAL_LABELS:
        label_df = df[df["signal"] == label].head(per_class).copy()
        grouped.append(label_df)
    pool = pd.concat(grouped, ignore_index=True)
    pool = pool.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return pool


def interactive_review(pool: pd.DataFrame, output_path: Path) -> None:
    rows: list[dict[str, str]] = []
    print("=" * 60)
    print("Manual Gold Signal Review")
    print("=" * 60)
    print("Tekan Enter untuk menerima label usulan.")
    print("Ketik salah satu label resmi untuk mengganti label.")
    print("Ketik 'skip' untuk melewati contoh.")
    print()

    for index, row in pool.iterrows():
        print(f"[{index + 1}/{len(pool)}]")
        print(f"Teks          : {row['text']}")
        print(f"Kota          : {row.get('city', '')}")
        print(f"Area hint     : {row.get('area_hint', '')}")
        print(f"Bisnis        : {row.get('business_hint', '')}")
        print(f"Label usulan  : {row['signal']}")
        answer = input("Gold label     : ").strip()
        if answer.lower() == "skip":
            print()
            continue
        if not answer:
            gold_label = str(row["signal"])
        else:
            gold_label = answer.upper()
        if gold_label not in SIGNAL_LABELS:
            print(f"Label '{gold_label}' tidak valid, contoh dilewati.")
            print()
            continue
        reviewer = input("Reviewer       : ").strip()
        notes = input("Catatan        : ").strip()
        rows.append(
            {
                "text": str(row["text"]),
                "city": str(row.get("city", "")),
                "area_hint": str(row.get("area_hint", "")),
                "business_hint": str(row.get("business_hint", "")),
                "gold_label": gold_label,
                "reviewer": reviewer,
                "review_notes": notes,
                "suggested_label": str(row["signal"]),
            }
        )
        print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Tersimpan {len(rows)} baris ke {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manually reviewed signal gold set.")
    parser.add_argument("--source", default=str(LABELED_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Only write a suggested template CSV without interactive review.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = Path(args.source)
    output_path = Path(args.output)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing labeled source file: {source_path}")

    pool = build_candidate_pool(source_path, per_class=args.per_class)
    if args.dry_run:
        draft = pool.rename(columns={"signal": "suggested_label"}).copy()
        for column in ["gold_label", "reviewer", "review_notes"]:
            draft[column] = ""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        draft[
            [
                "text",
                "city",
                "area_hint",
                "business_hint",
                "gold_label",
                "reviewer",
                "review_notes",
                "suggested_label",
            ]
        ].to_csv(output_path, index=False)
        print(f"Draft template written to {output_path}")
        return 0

    interactive_review(pool, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
