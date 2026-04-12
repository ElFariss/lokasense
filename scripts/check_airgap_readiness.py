#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch
from seqeval.metrics import classification_report as seq_classification_report
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
AGENT_DIR = BASE_DIR / "06_agent"
for path in [BASE_DIR, AGENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze as analyze_module
from airgap_retriever import AIRGAP_CORPUS, AIRGAP_MANIFEST, build_airgap_corpus

SIGNAL_LABELS = [
    "NEUTRAL",
    "DEMAND_UNMET",
    "DEMAND_PRESENT",
    "SUPPLY_SIGNAL",
    "COMPETITION_HIGH",
    "COMPLAINT",
    "TREND",
]
LOG_PATH = BASE_DIR / "logs" / "airgap_production_readiness.json"
GOLD_PATH = BASE_DIR / "test_data" / "signal_test_manual.csv"
NER_TEST_PATH = BASE_DIR / "test_data" / "ner_test.json"
SIGNAL_MODEL_DIR = BASE_DIR / "models" / "signal_base"
NER_MODEL_DIR = BASE_DIR / "models" / "ner_base"


@contextmanager
def no_network():
    original_create_connection = socket.create_connection
    original_socket = socket.socket

    def blocked_create_connection(*args, **kwargs):
        raise RuntimeError("network_blocked_for_airgap_check")

    class BlockedSocket(socket.socket):  # type: ignore[misc]
        def connect(self, *args, **kwargs):
            raise RuntimeError("network_blocked_for_airgap_check")

    socket.create_connection = blocked_create_connection  # type: ignore[assignment]
    socket.socket = BlockedSocket  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.socket = original_socket  # type: ignore[assignment]


def evaluate_signal_gold() -> dict[str, object]:
    if not GOLD_PATH.exists():
        return {"ready": False, "reason": "gold_file_missing", "samples": 0}

    gold_df = pd.read_csv(GOLD_PATH)
    required_columns = {"text", "gold_label", "reviewer"}
    if not required_columns.issubset(gold_df.columns):
        return {"ready": False, "reason": "gold_columns_missing", "samples": 0}

    gold_df = gold_df.dropna(subset=["text", "gold_label"]).copy()
    gold_df["gold_label"] = gold_df["gold_label"].astype(str).str.strip()
    gold_df = gold_df[gold_df["gold_label"].isin(SIGNAL_LABELS)].copy()
    if gold_df.empty:
        return {"ready": False, "reason": "gold_file_empty", "samples": 0}

    label_counts = gold_df["gold_label"].value_counts().to_dict()
    min_per_class = min(label_counts.get(label, 0) for label in SIGNAL_LABELS)
    reviewers = sorted(set(gold_df["reviewer"].fillna("").astype(str)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(SIGNAL_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(SIGNAL_MODEL_DIR)).to(device)
    model.eval()
    label2id = {label: index for index, label in enumerate(SIGNAL_LABELS)}

    texts = gold_df["text"].astype(str).tolist()
    preds: list[int] = []
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    true_ids = gold_df["gold_label"].map(label2id).tolist()
    report = classification_report(
        true_ids,
        preds,
        labels=list(range(len(SIGNAL_LABELS))),
        target_names=SIGNAL_LABELS,
        zero_division=0,
        output_dict=True,
    )
    return {
        "ready": True,
        "samples": int(len(gold_df)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "min_per_class": int(min_per_class),
        "reviewers": reviewers,
        "review_mode": "ai_assisted" if any("ai" in reviewer.lower() for reviewer in reviewers) else "manual",
        "manual_review_complete": not any("ai" in reviewer.lower() for reviewer in reviewers),
    }


def evaluate_ner() -> dict[str, object]:
    if not NER_TEST_PATH.exists():
        return {"ready": False, "reason": "ner_test_missing"}
    with open(NER_TEST_PATH, "r", encoding="utf-8") as handle:
        test_data = json.load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(NER_MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_DIR)).to(device)
    model.eval()

    true_labels = []
    pred_labels = []

    for item in test_data:
        tokens = item["tokens"]
        encoded = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt")
        word_ids = encoded.word_ids(batch_index=0)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        aligned = []
        previous_word_id = None
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word_id:
                previous_word_id = word_id
                continue
            aligned.append(model.config.id2label[pred_ids[token_index]])
            previous_word_id = word_id

        min_len = min(len(item["ner_tags"]), len(aligned))
        if min_len == 0:
            continue
        true_labels.append(item["ner_tags"][:min_len])
        pred_labels.append(aligned[:min_len])

    report = seq_classification_report(true_labels, pred_labels, output_dict=True)
    return {
        "ready": True,
        "micro_f1": float(report["micro avg"]["f1-score"]),
        "samples": int(len(true_labels)),
    }


def run_airgap_scenarios(timeout_sec: int) -> list[dict[str, object]]:
    scenarios = [
        ("saya ingin memulai bisnis kedai kopi di Malang", "outputs/airgap_malang.html"),
        ("mau buka laundry di daerah gubeng surabaya", "outputs/airgap_surabaya.html"),
        ("analisis peluang ayam geprek di Bandung", "outputs/airgap_bandung.html"),
    ]
    results: list[dict[str, object]] = []
    with no_network():
        for query, output in scenarios:
            result = analyze_module.analyze(
                query,
                output_path=output,
                timeout_sec=timeout_sec,
                source_mode="airgap",
            )
            results.append(
                {
                    "query": query,
                    "error": result.get("error", ""),
                    "rows_used": int(result.get("rows_used", 0)),
                    "elapsed_sec": float(result.get("elapsed_sec", 0.0)),
                    "map_path": result.get("map_path", ""),
                }
            )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check airgap production readiness.")
    parser.add_argument("--refresh-corpus", action="store_true", help="Rebuild the airgap corpus before running checks.")
    parser.add_argument("--timeout-sec", type=int, default=30, help="Per-query timeout for airgap analyze checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus_df, corpus_path, manifest_path = build_airgap_corpus(refresh=args.refresh_corpus)
    signal_gold = evaluate_signal_gold()
    ner_metrics = evaluate_ner()
    scenario_results = run_airgap_scenarios(timeout_sec=args.timeout_sec)

    checks = [
        {"check": "signal_model_present", "passed": SIGNAL_MODEL_DIR.exists(), "details": str(SIGNAL_MODEL_DIR)},
        {"check": "ner_model_present", "passed": NER_MODEL_DIR.exists(), "details": str(NER_MODEL_DIR)},
        {"check": "airgap_corpus_present", "passed": corpus_path.exists(), "details": str(corpus_path)},
        {"check": "airgap_corpus_rows", "passed": len(corpus_df) >= 150, "details": int(len(corpus_df))},
        {"check": "signal_gold_present", "passed": bool(signal_gold.get("ready")), "details": signal_gold},
        {"check": "signal_gold_macro_f1", "passed": float(signal_gold.get("macro_f1", 0.0)) >= 0.50, "details": signal_gold.get("macro_f1", 0.0)},
        {"check": "signal_gold_min_per_class", "passed": int(signal_gold.get("min_per_class", 0)) >= 8, "details": signal_gold.get("min_per_class", 0)},
        {
            "check": "signal_gold_manual_review_complete",
            "passed": bool(signal_gold.get("manual_review_complete")),
            "details": signal_gold.get("review_mode", "unknown"),
        },
        {"check": "ner_micro_f1", "passed": float(ner_metrics.get("micro_f1", 0.0)) >= 0.78, "details": ner_metrics.get("micro_f1", 0.0)},
        {
            "check": "airgap_queries_pass",
            "passed": all(not item["error"] for item in scenario_results),
            "details": scenario_results,
        },
        {
            "check": "airgap_latency_under_30s",
            "passed": all(float(item["elapsed_sec"]) <= args.timeout_sec for item in scenario_results),
            "details": [item["elapsed_sec"] for item in scenario_results],
        },
    ]

    failed_checks = [item["check"] for item in checks if not item["passed"]]
    operational_failed_checks = [
        item["check"]
        for item in checks
        if not item["passed"] and item["check"] != "signal_gold_manual_review_complete"
    ]
    operational_status = "ready" if not operational_failed_checks else "not_ready"
    status = "ready" if not failed_checks else "not_ready"
    payload = {
        "status": status,
        "operational_status": operational_status,
        "failed_checks": failed_checks,
        "operational_failed_checks": operational_failed_checks,
        "checks": checks,
        "airgap_corpus_path": str(corpus_path),
        "airgap_manifest_path": str(manifest_path),
        "signal_gold": signal_gold,
        "ner_metrics": ner_metrics,
        "scenario_results": scenario_results,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
