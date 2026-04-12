from __future__ import annotations

import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / "06_agent"
for path in [REPO_ROOT, AGENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from query_parser import parse_query


class QueryParserTests(unittest.TestCase):
    def test_parses_city_and_business(self) -> None:
        intent = parse_query("saya ingin memulai bisnis kedai kopi di Malang")
        self.assertEqual(intent.city, "Malang")
        self.assertEqual(intent.business_type, "kedai kopi")
        self.assertEqual(len(intent.kecamatan_scope), 5)

    def test_parses_specific_district(self) -> None:
        intent = parse_query("mau buka laundry di Lowokwaru")
        self.assertEqual(intent.city, "Malang")
        self.assertEqual(intent.kecamatan_scope, ["Lowokwaru"])
        self.assertEqual(intent.business_type, "laundry")

    def test_uses_fallbacks(self) -> None:
        intent = parse_query("rekomendasi usaha dong")
        self.assertEqual(intent.city, "Malang")
        self.assertEqual(intent.business_type, "kuliner")
        self.assertTrue(intent.fallback_used)


if __name__ == "__main__":
    unittest.main()
