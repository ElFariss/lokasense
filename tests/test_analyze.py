from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in [REPO_ROOT, REPO_ROOT / "06_agent"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze
from live_retriever import LiveRetrievalResult


class StubSignalRuntime:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict_labels(self, texts: list[str]):
        labels = []
        confidences = []
        cycle = ["DEMAND_UNMET", "DEMAND_PRESENT", "COMPLAINT", "TREND"]
        for index, _ in enumerate(texts):
            labels.append(cycle[index % len(cycle)])
            confidences.append(0.82)
        return labels, confidences

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        probs = np.zeros((len(texts), len(analyze.SIGNAL_LABELS)), dtype=float)
        probs[:, 1] = 0.7
        probs[:, 2] = 0.2
        probs[:, 5] = 0.1
        return probs


class StubTokenRuntime:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict_entities(self, text: str):
        lowered = text.lower()
        if "lowokwaru" in lowered:
            return [{"entity": "Lowokwaru", "label": "LOC"}]
        if "klojen" in lowered:
            return [{"entity": "Klojen", "label": "LOC"}]
        if "blimbing" in lowered:
            return [{"entity": "Blimbing", "label": "LOC"}]
        return []


def fake_live_retrieval(intent, output_dir, **kwargs):
    rows = []
    for area in ["Lowokwaru", "Klojen", "Blimbing"]:
        for index in range(12):
            rows.append(
                {
                    "text": f"butuh kedai kopi di {area} malang untuk anak kos {index}",
                    "raw_text": f"Butuh kedai kopi di {area} Malang untuk anak kos {index}",
                    "source": "test_stub",
                    "platform": "tiktok",
                    "url": f"https://example.com/{area.lower()}/{index}",
                    "content_id": f"{area.lower()}-{index}",
                    "video_id": f"{area.lower()}-{index}",
                    "author": "tester",
                    "timestamp": "2026-04-10T00:00:00Z",
                    "area_hint": area,
                    "city": intent.city,
                    "business_hint": intent.business_type,
                    "likes": "10",
                    "comments": "1",
                    "shares": "0",
                    "views": "100",
                    "hashtags": "",
                    "query": intent.raw_query,
                    "query_intent": "user_query",
                    "scrape_mode": "test",
                    "collected_at": "2026-04-12T00:00:00Z",
                }
            )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "live_social_rows.csv", index=False)
    return LiveRetrievalResult(
        frame=frame,
        rows_fetched=len(frame),
        rows_used=len(frame),
        run_dir=output_dir,
        manifest_path=output_dir / "live_manifest.json",
        combined_path=output_dir / "live_social_rows.csv",
        errors=[],
    )


class AnalyzeTests(unittest.TestCase):
    @mock.patch.object(analyze, "run_lime_for_top_areas", return_value={"Lowokwaru": [{"token": "butuh", "weight": 0.41}]})
    @mock.patch.object(analyze, "TokenRuntime", StubTokenRuntime)
    @mock.patch.object(analyze, "SequenceRuntime", StubSignalRuntime)
    def test_analyze_success_with_stubbed_live_data(self, _mock_lime) -> None:
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT)) as tmp_dir:
            output_path = Path(tmp_dir) / "heatmap.html"
            result = analyze.analyze(
                "saya ingin memulai bisnis kedai kopi di Malang",
                output_path=str(output_path),
                collector=fake_live_retrieval,
                timeout_sec=5,
                source_mode="live",
            )
            self.assertEqual(result["error"], "")
            self.assertTrue(output_path.exists())
            self.assertGreaterEqual(result["rows_used"], 30)
            self.assertGreaterEqual(len(result["scores_df"]), 3)

    def test_analyze_reports_live_data_failure(self) -> None:
        def failing_collector(*args, **kwargs):
            raise ValueError("insufficient_live_data")

        result = analyze.analyze(
            "saya ingin memulai bisnis kedai kopi di Malang",
            collector=failing_collector,
            timeout_sec=5,
            source_mode="live",
        )
        self.assertIn("insufficient_live_data", result["error"])

    @mock.patch.object(analyze, "run_lime_for_top_areas", return_value={})
    @mock.patch.object(analyze, "TokenRuntime", StubTokenRuntime)
    @mock.patch.object(analyze, "SequenceRuntime", StubSignalRuntime)
    def test_single_district_scope_allows_one_scored_area(self, _mock_lime) -> None:
        def single_area_collector(intent, output_dir, **kwargs):
            rows = []
            for index in range(12):
                rows.append(
                    {
                        "text": f"butuh laundry di Gubeng Surabaya cepat dan rapi {index}",
                        "raw_text": f"Butuh laundry di Gubeng Surabaya cepat dan rapi {index}",
                        "source": "test_stub",
                        "platform": "tiktok",
                        "url": f"https://example.com/gubeng/{index}",
                        "content_id": f"gubeng-{index}",
                        "video_id": f"gubeng-{index}",
                        "author": "tester",
                        "timestamp": "2026-04-10T00:00:00Z",
                        "area_hint": "Gubeng",
                        "city": intent.city,
                        "business_hint": intent.business_type,
                        "likes": "10",
                        "comments": "1",
                        "shares": "0",
                        "views": "100",
                        "hashtags": "",
                        "query": intent.raw_query,
                        "query_intent": "user_query",
                        "scrape_mode": "test",
                        "collected_at": "2026-04-12T00:00:00Z",
                    }
                )
            frame = pd.DataFrame(rows)
            output_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_dir / "live_social_rows.csv", index=False)
            return LiveRetrievalResult(
                frame=frame,
                rows_fetched=len(frame),
                rows_used=len(frame),
                run_dir=output_dir,
                manifest_path=output_dir / "live_manifest.json",
                combined_path=output_dir / "live_social_rows.csv",
                errors=[],
            )

        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT)) as tmp_dir:
            output_path = Path(tmp_dir) / "heatmap_single.html"
            result = analyze.analyze(
                "mau buka laundry di daerah gubeng surabaya",
                output_path=str(output_path),
                collector=single_area_collector,
                timeout_sec=5,
                source_mode="live",
            )
            self.assertEqual(result["error"], "")
            self.assertTrue(output_path.exists())
            self.assertEqual(result["scores_df"]["kecamatan"].nunique(), 1)

    @mock.patch.object(analyze, "SequenceRuntime", StubSignalRuntime)
    def test_mobile_profile_runs_without_ner(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT)) as tmp_dir:
            output_path = Path(tmp_dir) / "heatmap_mobile.html"
            result = analyze.analyze(
                "saya ingin memulai bisnis kedai kopi di Malang",
                output_path=str(output_path),
                collector=fake_live_retrieval,
                timeout_sec=5,
                source_mode="live",
                runtime_profile="mobile",
            )
            self.assertEqual(result["error"], "")
            self.assertTrue(output_path.exists())
            self.assertGreaterEqual(result["scores_df"]["kecamatan"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
