from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / "06_agent"
for path in [REPO_ROOT, AGENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import airgap_retriever
from query_parser import parse_query


class AirgapRetrieverTests(unittest.TestCase):
    def test_collect_airgap_data_from_local_corpus(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT)) as tmp_dir:
            tmp_root = Path(tmp_dir)
            source_file = tmp_root / "weak_labeled.csv"
            rows = []
            for area in ["Lowokwaru", "Klojen", "Blimbing"]:
                for index in range(12):
                    rows.append(
                        {
                            "text": f"butuh kedai kopi di {area} malang untuk anak kos {index}",
                            "signal": "DEMAND_UNMET",
                            "source": "test",
                            "platform": "tiktok",
                            "url": f"https://example.com/{area}/{index}",
                            "timestamp": "2026-04-10T00:00:00Z",
                            "city": "Malang",
                            "area_hint": area,
                            "business_hint": "kedai kopi",
                            "query": "kedai kopi",
                            "query_intent": "demand",
                        }
                    )
            pd.DataFrame(rows).to_csv(source_file, index=False)

            with mock.patch.object(airgap_retriever, "SOURCE_FILES", [source_file]), \
                 mock.patch.object(airgap_retriever, "SOURCE_GLOBS", []), \
                 mock.patch.object(airgap_retriever, "AIRGAP_DIR", tmp_root / "airgap"), \
                 mock.patch.object(airgap_retriever, "AIRGAP_CORPUS", tmp_root / "airgap" / "airgap_corpus.csv"), \
                 mock.patch.object(airgap_retriever, "AIRGAP_MANIFEST", tmp_root / "airgap" / "airgap_manifest.json"):
                intent = parse_query("saya ingin memulai bisnis kedai kopi di Malang")
                result = airgap_retriever.collect_airgap_data(intent, min_rows=30, refresh_corpus=True)
                self.assertGreaterEqual(result.rows_used, 30)
                self.assertTrue(result.corpus_path.exists())
                self.assertEqual(sorted(result.frame["area_hint"].unique().tolist()), ["Blimbing", "Klojen", "Lowokwaru"])


if __name__ == "__main__":
    unittest.main()
