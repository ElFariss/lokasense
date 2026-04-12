#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
AGENT_DIR = BASE_DIR / "06_agent"
SPATIAL_DIR = BASE_DIR / "04_spatial_engine" / "modelling"
EXPLAIN_DIR = BASE_DIR / "05_explainability" / "modelling"
for path in [BASE_DIR, AGENT_DIR, SPATIAL_DIR, EXPLAIN_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.location_resolution import LocationResolver
from airgap_retriever import AirgapRetrievalResult, build_airgap_corpus, collect_airgap_data
from explainer import generate_explanation
from live_retriever import LiveRetrievalResult, collect_live_data
from query_parser import IntentResult, parse_query

import heatmap as spatial_heatmap
import scoring as spatial_scoring

SIGNAL_LABELS = [
    "NEUTRAL",
    "DEMAND_UNMET",
    "DEMAND_PRESENT",
    "SUPPLY_SIGNAL",
    "COMPETITION_HIGH",
    "COMPLAINT",
    "TREND",
]

RUNTIME_PROFILES: dict[str, dict[str, object]] = {
    "full": {
        "use_ner": True,
        "enable_lime": True,
        "prefer_quantized_signal": False,
        "prefer_quantized_ner": False,
        "single_scope_max_rows": 80,
        "multi_scope_max_rows": 120,
    },
    "edge": {
        "use_ner": False,
        "enable_lime": False,
        "prefer_quantized_signal": True,
        "prefer_quantized_ner": False,
        "single_scope_max_rows": 48,
        "multi_scope_max_rows": 72,
    },
    "mobile": {
        "use_ner": False,
        "enable_lime": False,
        "prefer_quantized_signal": True,
        "prefer_quantized_ner": False,
        "single_scope_max_rows": 32,
        "multi_scope_max_rows": 48,
    },
}


class SequenceRuntime:
    def __init__(self, onnx_dir: Path, pytorch_dir: Path, *, prefer_quantized: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.source = "pytorch"
        self.session = None

        quantized_path = onnx_dir / "model_quantized.onnx"
        if prefer_quantized and quantized_path.exists():
            try:
                import onnxruntime as ort

                self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
                self.session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
                self.input_names = [item.name for item in self.session.get_inputs()]
                self.source = "onnx_quantized"
                return
            except Exception:
                self.source = "pytorch"
        if onnx_dir.exists():
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification

                self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
                self.model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))
                self.source = "onnx"
                return
            except Exception:
                self.source = "pytorch"
        if pytorch_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(pytorch_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(pytorch_dir)).to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Signal model not found at {onnx_dir} or {pytorch_dir}")

    def predict_proba(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.source == "onnx_quantized":
            encoded = self.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="np")
            input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
            attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
            token_type_ids = np.asarray(encoded.get("token_type_ids", np.zeros_like(input_ids)), dtype=np.int64)
            ort_inputs: dict[str, np.ndarray] = {}
            for name in self.input_names:
                if name == "input_ids":
                    ort_inputs[name] = input_ids
                elif name == "attention_mask":
                    ort_inputs[name] = attention_mask
                elif name == "token_type_ids":
                    ort_inputs[name] = token_type_ids
            logits = np.asarray(self.session.run(None, ort_inputs)[0], dtype=np.float32)
            logits = logits - logits.max(axis=-1, keepdims=True)
            exp = np.exp(logits)
            return exp / exp.sum(axis=-1, keepdims=True)
        encoded = self.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
        if self.source == "pytorch":
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def predict_labels(self, texts: list[str]) -> tuple[list[str], list[float]]:
        probs = self.predict_proba(texts)
        labels: list[str] = []
        confidences: list[float] = []
        for prob in probs:
            top_idx = int(np.argmax(prob))
            labels.append(SIGNAL_LABELS[top_idx])
            confidences.append(float(prob[top_idx]))
        return labels, confidences


class TokenRuntime:
    def __init__(self, onnx_dir: Path, pytorch_dir: Path, *, prefer_quantized: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.source = "pytorch"
        self.session = None

        quantized_path = onnx_dir / "model_quantized.onnx"
        if prefer_quantized and quantized_path.exists():
            try:
                import onnxruntime as ort

                self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
                self.session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
                self.input_names = [item.name for item in self.session.get_inputs()]
                if pytorch_dir.exists():
                    config_model = AutoModelForTokenClassification.from_pretrained(str(pytorch_dir))
                    self.id2label = {int(key): value for key, value in config_model.config.id2label.items()}
                    del config_model
                else:
                    self.id2label = {}
                self.source = "onnx_quantized"
                return
            except Exception:
                self.source = "pytorch"
        if onnx_dir.exists():
            try:
                from optimum.onnxruntime import ORTModelForTokenClassification

                self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
                self.model = ORTModelForTokenClassification.from_pretrained(str(onnx_dir))
                self.source = "onnx"
                return
            except Exception:
                self.source = "pytorch"
        if pytorch_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(pytorch_dir))
            self.model = AutoModelForTokenClassification.from_pretrained(str(pytorch_dir)).to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"NER model not found at {onnx_dir} or {pytorch_dir}")

    def predict_entities(self, text: str) -> list[dict[str, str]]:
        tokens = re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)
        if not tokens:
            return []
        tensor_type = "np" if self.source == "onnx_quantized" else "pt"
        encoded = self.tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128, return_tensors=tensor_type)
        word_ids = encoded.word_ids(batch_index=0)
        if self.source == "onnx_quantized":
            input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
            attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
            token_type_ids = np.asarray(encoded.get("token_type_ids", np.zeros_like(input_ids)), dtype=np.int64)
            ort_inputs: dict[str, np.ndarray] = {}
            for name in self.input_names:
                if name == "input_ids":
                    ort_inputs[name] = input_ids
                elif name == "attention_mask":
                    ort_inputs[name] = attention_mask
                elif name == "token_type_ids":
                    ort_inputs[name] = token_type_ids
            logits = np.asarray(self.session.run(None, ort_inputs)[0], dtype=np.float32)
            pred_ids = logits.argmax(axis=-1)[0].tolist()
        else:
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()

        aligned: list[tuple[str, str]] = []
        previous_word_id = None
        id2label = self.id2label if self.source == "onnx_quantized" else self.model.config.id2label
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word_id:
                previous_word_id = word_id
                continue
            aligned.append((tokens[word_id], id2label[pred_ids[token_index]]))
            previous_word_id = word_id

        entities: list[dict[str, str]] = []
        current_tokens: list[str] = []
        current_label: str | None = None
        for token, tag in aligned:
            if tag.startswith("B-"):
                if current_tokens:
                    entities.append({"entity": " ".join(current_tokens), "label": str(current_label or "")})
                current_tokens = [token]
                current_label = tag[2:]
            elif tag.startswith("I-") and current_tokens and current_label == tag[2:]:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append({"entity": " ".join(current_tokens), "label": str(current_label or "")})
                current_tokens = []
                current_label = None
        if current_tokens:
            entities.append({"entity": " ".join(current_tokens), "label": str(current_label or "")})
        return entities


def classify_signal_rows(raw_df: pd.DataFrame, runtime: SequenceRuntime) -> pd.DataFrame:
    classified = raw_df.copy()
    labels, confidences = runtime.predict_labels(classified["text"].fillna("").astype(str).tolist())
    classified["final_signal"] = labels
    classified["signal_confidence"] = [round(conf, 4) for conf in confidences]
    return classified


def resolve_live_rows(
    labeled_df: pd.DataFrame,
    intent: IntentResult,
    ner_runtime: TokenRuntime | None,
    resolver: LocationResolver,
) -> pd.DataFrame:
    resolved_rows: list[dict[str, object]] = []
    allowed_areas = set(intent.kecamatan_scope)
    single_scope_area = next(iter(allowed_areas)) if len(allowed_areas) == 1 else ""

    for _, row in labeled_df.iterrows():
        item = row.to_dict()
        area_hint = str(item.get("area_hint", "")).strip()
        chosen_area = area_hint if area_hint in allowed_areas else ""
        chosen_lat = None
        chosen_lng = None
        resolution_source = ""

        if chosen_area:
            area_resolution = resolver.resolve_area(intent.city, chosen_area)
            chosen_lat = area_resolution["lat"]
            chosen_lng = area_resolution["lng"]
            resolution_source = "seed_area_hint"

        resolved_entities: list[dict[str, object]] = []
        if ner_runtime is not None:
            entities = ner_runtime.predict_entities(str(item.get("text", "")))
            resolved_entities = resolver.resolve_entities(entities, city_hint=intent.city, area_hint=area_hint)
            best_entity = next(
                (
                    entity
                    for entity in resolved_entities
                    if entity.get("resolved_area") and (not allowed_areas or entity.get("resolved_area") in allowed_areas)
                ),
                None,
            )
            if best_entity:
                chosen_area = str(best_entity.get("resolved_area") or chosen_area)
                chosen_lat = best_entity.get("lat")
                chosen_lng = best_entity.get("lng")
                resolution_source = str(best_entity.get("resolution_source") or "ner_entity")

        if not chosen_area:
            text_resolution = resolver.resolve_text(
                str(item.get("text", "")),
                city_hint=intent.city,
                area_hint=area_hint,
                allowed_areas=list(allowed_areas),
            )
            if text_resolution.resolved_area:
                chosen_area = text_resolution.resolved_area
                chosen_lat = text_resolution.lat
                chosen_lng = text_resolution.lng
                resolution_source = text_resolution.resolution_source

        if not chosen_area and single_scope_area:
            area_resolution = resolver.resolve_area(intent.city, single_scope_area)
            chosen_area = single_scope_area
            chosen_lat = area_resolution["lat"]
            chosen_lng = area_resolution["lng"]
            resolution_source = "query_scope_fallback"

        item["city"] = intent.city
        item["resolved_area"] = chosen_area
        item["area_hint"] = chosen_area or area_hint
        item["resolved_lat"] = chosen_lat
        item["resolved_lng"] = chosen_lng
        item["resolution_source"] = resolution_source or "unresolved"
        item["resolved_entities"] = json.dumps(resolved_entities, ensure_ascii=False)
        resolved_rows.append(item)

    resolved_df = pd.DataFrame(resolved_rows)
    if allowed_areas:
        resolved_df = resolved_df[resolved_df["area_hint"].isin(allowed_areas)].copy()
    return resolved_df


def run_lime_for_top_areas(
    signal_runtime: SequenceRuntime,
    labeled_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    *,
    max_areas: int = 3,
    num_samples: int = 500,
) -> dict[str, list[dict[str, object]]]:
    try:
        from explain import explain_single
    except Exception:
        return {}

    lime_data: dict[str, list[dict[str, object]]] = {}
    if scores_df.empty or labeled_df.empty:
        return lime_data

    top_areas = scores_df.sort_values("opportunity_score", ascending=False)["kecamatan"].head(max_areas).tolist()
    for area in top_areas:
        area_df = labeled_df[labeled_df["area_hint"] == area].copy()
        if area_df.empty:
            continue
        if "signal_confidence" in area_df.columns:
            area_df = area_df.sort_values("signal_confidence", ascending=False)
        sample_text = str(area_df.iloc[0]["text"])
        try:
            explanation = explain_single(signal_runtime, sample_text, num_features=3, num_samples=num_samples)
            lime_data[str(area)] = explanation.get("top_3_features", [])
        except Exception:
            continue
    return lime_data


def build_run_dir(mode: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / "outputs" / f"{mode}_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def analyze(
    query: str,
    *,
    output_path: str = "outputs/lokasense_heatmap.html",
    top_n: int = 5,
    timeout_sec: int = 90,
    min_live_rows: int = 30,
    platforms: list[str] | None = None,
    headless: bool = True,
    source_mode: str = "airgap",
    runtime_profile: str = "full",
    refresh_airgap_corpus: bool = False,
    collector=collect_live_data,
) -> dict[str, object]:
    start = time.time()
    intent = parse_query(query)
    resolver = LocationResolver()
    if source_mode not in {"airgap", "live"}:
        raise ValueError(f"Unsupported source_mode: {source_mode}")
    if runtime_profile not in RUNTIME_PROFILES:
        raise ValueError(f"Unsupported runtime_profile: {runtime_profile}")
    profile = RUNTIME_PROFILES[runtime_profile]
    run_dir = build_run_dir(source_mode)
    live_fetch_budget_sec = max(15, int(timeout_sec) - 10)
    required_scored_areas = 1 if len(intent.kecamatan_scope) == 1 else 3
    effective_min_live_rows = max(10, min_live_rows // 3) if len(intent.kecamatan_scope) == 1 else min_live_rows
    max_runtime_rows = int(
        profile["single_scope_max_rows"] if len(intent.kecamatan_scope) == 1 else profile["multi_scope_max_rows"]
    )

    print("=" * 60)
    print("LokaSense Analysis")
    print("=" * 60)
    print(f"Query   : {query}")
    print(f"Kota    : {intent.city}")
    print(f"Bisnis  : {intent.business_type}")
    print(f"Scope   : {', '.join(intent.kecamatan_scope)}")
    print(f"Mode    : {source_mode}")
    print(f"Profile : {runtime_profile}")
    if intent.fallback_used:
        print("Parser memakai fallback default untuk sebagian intent.")

    try:
        if source_mode == "live":
            retrieval = collector(
                intent,
                run_dir,
                timeout_sec=live_fetch_budget_sec,
                platforms=platforms,
                min_live_rows=effective_min_live_rows,
                headless=headless,
            )
        else:
            retrieval = collect_airgap_data(
                intent,
                min_rows=effective_min_live_rows,
                refresh_corpus=refresh_airgap_corpus,
            )
    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        return {
            "intent": intent,
            "rows_fetched": 0,
            "rows_used": 0,
            "scores_df": pd.DataFrame(),
            "explanation": "",
            "map_path": None,
            "elapsed_sec": elapsed,
            "error": str(exc),
        }

    raw_df = retrieval.frame.copy()
    if len(raw_df) > max_runtime_rows:
        raw_df = raw_df.head(max_runtime_rows).copy()
    print(f"Rows fetched : {retrieval.rows_fetched}")
    print(f"Rows usable  : {len(raw_df)}")

    signal_runtime = SequenceRuntime(
        BASE_DIR / "signal_onnx",
        BASE_DIR / "models" / "signal_base",
        prefer_quantized=bool(profile["prefer_quantized_signal"]),
    )
    labeled_df = classify_signal_rows(raw_df, signal_runtime)
    if not bool(profile["enable_lime"]):
        del signal_runtime
        gc.collect()

    ner_runtime: TokenRuntime | None = None
    if bool(profile["use_ner"]):
        ner_runtime = TokenRuntime(
            BASE_DIR / "ner_onnx",
            BASE_DIR / "models" / "ner_base",
            prefer_quantized=bool(profile["prefer_quantized_ner"]),
        )

    labeled_df = resolve_live_rows(labeled_df, intent, ner_runtime, resolver)
    if ner_runtime is not None:
        del ner_runtime
        gc.collect()
    if "business_hint" in labeled_df.columns:
        labeled_df["retrieved_business_hint"] = labeled_df["business_hint"].fillna("").astype(str)
    labeled_df["business_hint"] = intent.business_type

    poi_file = BASE_DIR / "data" / "poi" / "overpass_poi.csv"
    poi_df = pd.read_csv(poi_file) if poi_file.exists() else None
    scores_df = spatial_scoring.compute_opportunity_scores(labeled_df, poi_df=poi_df, resolver=resolver)
    if scores_df.empty or scores_df["kecamatan"].nunique() < required_scored_areas:
        elapsed = round(time.time() - start, 2)
        result = {
            "intent": intent,
            "rows_fetched": retrieval.rows_fetched,
            "rows_used": int(len(labeled_df)),
            "scores_df": scores_df,
            "explanation": "",
            "map_path": None,
            "elapsed_sec": elapsed,
            "error": "insufficient_live_data_after_scoring",
        }
        with open(run_dir / "analysis_result.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "intent": asdict(intent),
                    "rows_fetched": retrieval.rows_fetched,
                    "rows_used": int(len(labeled_df)),
                    "elapsed_sec": elapsed,
                    "required_scored_areas": required_scored_areas,
                    "effective_min_live_rows": effective_min_live_rows,
                    "source_mode": source_mode,
                    "runtime_profile": runtime_profile,
                    "error": result["error"],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        return result

    elapsed_before_lime = time.time() - start
    remaining_budget = max(0.0, float(timeout_sec) - elapsed_before_lime)
    if not bool(profile["enable_lime"]):
        lime_max_areas = 0
        lime_samples = 0
    elif remaining_budget >= 35:
        lime_max_areas = 3
        lime_samples = 500
    elif remaining_budget >= 20:
        lime_max_areas = 2
        lime_samples = 300
    elif remaining_budget >= 10:
        lime_max_areas = 1
        lime_samples = 200
    else:
        lime_max_areas = 0
        lime_samples = 0
    if source_mode == "airgap":
        lime_max_areas = min(lime_max_areas, 2)
        lime_samples = min(lime_samples, 250)
        if len(raw_df) >= 100 and remaining_budget < 20:
            lime_max_areas = min(lime_max_areas, 1)
            lime_samples = min(lime_samples, 150)

    lime_data = (
        run_lime_for_top_areas(
            signal_runtime,
            labeled_df,
            scores_df,
            max_areas=lime_max_areas,
            num_samples=lime_samples,
        )
        if lime_max_areas > 0
        else {}
    )
    marker_map = spatial_heatmap.create_marker_map(scores_df, lime_data=lime_data)
    output_file = BASE_DIR / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    marker_map.save(str(output_file))

    explanation = generate_explanation(
        intent,
        scores_df,
        lime_data=lime_data,
        top_n=top_n,
        source_mode=source_mode,
    )
    labeled_output = run_dir / "labeled_live_rows.csv"
    labeled_df.to_csv(labeled_output, index=False)
    scores_df.to_csv(run_dir / "opportunity_scores.csv", index=False)

    elapsed = round(time.time() - start, 2)
    if elapsed > timeout_sec:
        timeout_error = f"timeout_exceeded: elapsed={elapsed}s > budget={timeout_sec}s"
        timeout_payload = {
            "intent": asdict(intent),
            "rows_fetched": retrieval.rows_fetched,
            "rows_used": int(len(labeled_df)),
            "map_path": str(output_file),
            "elapsed_sec": elapsed,
            "live_fetch_budget_sec": live_fetch_budget_sec,
            "required_scored_areas": required_scored_areas,
            "effective_min_live_rows": effective_min_live_rows,
            "source_mode": source_mode,
            "runtime_profile": runtime_profile,
            "error": timeout_error,
        }
        with open(run_dir / "analysis_result.json", "w", encoding="utf-8") as handle:
            json.dump(timeout_payload, handle, indent=2, ensure_ascii=False)
        return {
            "intent": intent,
            "rows_fetched": retrieval.rows_fetched,
            "rows_used": int(len(labeled_df)),
            "scores_df": scores_df,
            "explanation": explanation,
            "map_path": str(output_file),
            "elapsed_sec": elapsed,
            "error": timeout_error,
        }
    summary = {
        "intent": asdict(intent),
        "rows_fetched": retrieval.rows_fetched,
        "rows_used": int(len(labeled_df)),
        "map_path": str(output_file),
        "elapsed_sec": elapsed,
        "error": "",
        "live_fetch_budget_sec": live_fetch_budget_sec,
        "required_scored_areas": required_scored_areas,
        "effective_min_live_rows": effective_min_live_rows,
        "source_mode": source_mode,
        "runtime_profile": runtime_profile,
    }
    with open(run_dir / "analysis_result.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Map      : {output_file}")
    print(f"Elapsed  : {elapsed:.2f}s")
    print()
    print(explanation)

    return {
        "intent": intent,
        "rows_fetched": retrieval.rows_fetched,
        "rows_used": int(len(labeled_df)),
        "scores_df": scores_df,
        "explanation": explanation,
        "map_path": str(output_file),
        "elapsed_sec": elapsed,
        "error": "",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LokaSense market analysis")
    parser.add_argument("query", help="Natural-language Indonesian business query")
    parser.add_argument("--output", default="outputs/lokasense_heatmap.html")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--min-live-rows", type=int, default=30)
    parser.add_argument("--platforms", default="tiktok,instagram,x")
    parser.add_argument("--profile", choices=sorted(RUNTIME_PROFILES), default="full", help="Runtime footprint profile.")
    parser.add_argument("--show-browser", action="store_true", help="Run the scraper in non-headless mode.")
    parser.add_argument("--live", action="store_true", help="Use live public scraping instead of the default airgap local corpus.")
    parser.add_argument("--refresh-airgap-corpus", action="store_true", help="Rebuild the local airgap corpus before running offline analysis.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    platforms = [item.strip() for item in args.platforms.split(",") if item.strip()]
    result = analyze(
        args.query,
        output_path=args.output,
        top_n=args.top_n,
        timeout_sec=args.timeout_sec,
        min_live_rows=args.min_live_rows,
        platforms=platforms,
        headless=not args.show_browser,
        source_mode="live" if args.live else "airgap",
        runtime_profile=args.profile,
        refresh_airgap_corpus=args.refresh_airgap_corpus,
    )
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
