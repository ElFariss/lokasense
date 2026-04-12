#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
AGENT_DIR = BASE_DIR / "06_agent"
for path in [BASE_DIR, AGENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from airgap_retriever import build_airgap_corpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local airgap retrieval corpus.")
    parser.add_argument("--refresh", action="store_true", help="Force a rebuild from source artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus_df, corpus_path, manifest_path = build_airgap_corpus(refresh=args.refresh)
    print(f"Corpus rows  : {len(corpus_df)}")
    print(f"Corpus path  : {corpus_path}")
    print(f"Manifest path: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
