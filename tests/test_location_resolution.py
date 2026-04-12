from __future__ import annotations

import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.location_resolution import LocationResolver


class LocationResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = LocationResolver()

    def test_resolves_proximity_text(self) -> None:
        item = self.resolver.resolve_text(
            "butuh tempat nongkrong deket lowokwaru buat anak kos",
            city_hint="Malang",
            allowed_areas=["Lowokwaru", "Klojen"],
        )
        self.assertEqual(item.resolved_area, "Lowokwaru")
        self.assertEqual(item.resolved_city, "Malang")

    def test_resolves_exact_area_mention(self) -> None:
        item = self.resolver.resolve_text(
            "laundry di gubeng lagi rame",
            city_hint="Surabaya",
            allowed_areas=["Gubeng", "Tegalsari"],
        )
        self.assertEqual(item.resolved_area, "Gubeng")
        self.assertEqual(item.resolved_city, "Surabaya")


if __name__ == "__main__":
    unittest.main()
