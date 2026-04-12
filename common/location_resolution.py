from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.market_catalog import CITIES_KECAMATAN
from common.text_normalization import normalize_for_dedupe

BASE_DIR = Path(__file__).resolve().parent.parent
GMAPS_FILE = BASE_DIR / "data" / "social_media" / "gmaps_reviews.csv"
POI_FILE = BASE_DIR / "data" / "poi" / "overpass_poi.csv"


@dataclass
class ResolvedLocation:
    entity: str
    label: str
    resolved_city: str
    resolved_area: str
    lat: float | None
    lng: float | None
    resolution_source: str


class LocationResolver:
    def __init__(self) -> None:
        self.city_index = {normalize_for_dedupe(city): city for city in CITIES_KECAMATAN}
        self.area_index: dict[str, list[dict[str, str]]] = {}
        for city, areas in CITIES_KECAMATAN.items():
            for area in areas:
                key = normalize_for_dedupe(area)
                self.area_index.setdefault(key, []).append({"city": city, "area": area})

        self.city_centroids: dict[str, tuple[float, float]] = {}
        self.area_centroids: dict[tuple[str, str], tuple[float, float]] = {}
        self.poi_index: dict[tuple[str, str], tuple[float, float, str]] = {}
        self._load_centroids()

    @staticmethod
    def _safe_mean(df: pd.DataFrame, lat_col: str, lng_col: str) -> tuple[float, float] | None:
        if df.empty:
            return None
        lat = pd.to_numeric(df[lat_col], errors="coerce").dropna()
        lng = pd.to_numeric(df[lng_col], errors="coerce").dropna()
        if lat.empty or lng.empty:
            return None
        return float(lat.mean()), float(lng.mean())

    def _load_centroids(self) -> None:
        city_frames: list[pd.DataFrame] = []
        area_frames: list[pd.DataFrame] = []

        if GMAPS_FILE.exists():
            gmaps_df = pd.read_csv(GMAPS_FILE)
            if {"city", "lat", "lng"}.issubset(gmaps_df.columns):
                city_frames.append(gmaps_df[["city", "lat", "lng"]].copy())
            if {"city", "area_hint", "lat", "lng"}.issubset(gmaps_df.columns):
                area_frames.append(gmaps_df[["city", "area_hint", "lat", "lng"]].copy())
            if {"city", "place_name", "lat", "lng"}.issubset(gmaps_df.columns):
                poi_df = gmaps_df.dropna(subset=["city", "place_name", "lat", "lng"]).copy()
                for _, row in poi_df.iterrows():
                    key = (str(row["city"]).strip(), normalize_for_dedupe(str(row["place_name"])))
                    if key[1]:
                        self.poi_index[key] = (float(row["lat"]), float(row["lng"]), "gmaps_place")

        if POI_FILE.exists():
            poi_df = pd.read_csv(POI_FILE)
            if {"city", "lat", "lon"}.issubset(poi_df.columns):
                city_frames.append(poi_df[["city", "lat", "lon"]].rename(columns={"lon": "lng"}))
            if {"city", "name", "lat", "lon"}.issubset(poi_df.columns):
                named_poi_df = poi_df.dropna(subset=["city", "name", "lat", "lon"]).copy()
                for _, row in named_poi_df.iterrows():
                    key = (str(row["city"]).strip(), normalize_for_dedupe(str(row["name"])))
                    if key[1] and key not in self.poi_index:
                        self.poi_index[key] = (float(row["lat"]), float(row["lon"]), "overpass_poi")

        if city_frames:
            combined_city_df = pd.concat(city_frames, ignore_index=True)
            for city, city_df in combined_city_df.groupby("city"):
                centroid = self._safe_mean(city_df, "lat", "lng")
                if centroid:
                    self.city_centroids[str(city).strip()] = centroid

        if area_frames:
            combined_area_df = pd.concat(area_frames, ignore_index=True)
            combined_area_df = combined_area_df.dropna(subset=["city", "area_hint"])
            for (city, area), area_df in combined_area_df.groupby(["city", "area_hint"]):
                centroid = self._safe_mean(area_df, "lat", "lng")
                if centroid:
                    self.area_centroids[(str(city).strip(), str(area).strip())] = centroid

    def resolve_area(self, city: str, area: str) -> dict[str, object]:
        city = str(city or "").strip()
        area = str(area or "").strip()
        lat, lng = None, None
        resolution_source = "unresolved"

        if city and area and (city, area) in self.area_centroids:
            lat, lng = self.area_centroids[(city, area)]
            resolution_source = "area_centroid"
        elif city and city in self.city_centroids:
            lat, lng = self.city_centroids[city]
            resolution_source = "city_centroid"

        return {
            "resolved_city": city,
            "resolved_area": area,
            "lat": lat,
            "lng": lng,
            "resolution_source": resolution_source,
        }

    def resolve_entity(self, entity_text: str, label: str = "", city_hint: str = "", area_hint: str = "") -> ResolvedLocation:
        entity_text = str(entity_text or "").strip()
        normalized = normalize_for_dedupe(entity_text)
        city_hint = str(city_hint or "").strip()
        area_hint = str(area_hint or "").strip()
        if not normalized:
            return ResolvedLocation(entity_text, label, city_hint, area_hint, None, None, "unresolved")

        if city_hint:
            poi_key = (city_hint, normalized)
            if poi_key in self.poi_index:
                lat, lng, source = self.poi_index[poi_key]
                return ResolvedLocation(entity_text, label, city_hint, area_hint, lat, lng, source)

        if normalized in self.city_index:
            resolved_city = self.city_index[normalized]
            lat_lng = self.city_centroids.get(resolved_city)
            return ResolvedLocation(
                entity_text,
                label,
                resolved_city,
                "",
                lat_lng[0] if lat_lng else None,
                lat_lng[1] if lat_lng else None,
                "city_catalog",
            )

        area_matches = self.area_index.get(normalized, [])
        if area_matches:
            area_match = next((match for match in area_matches if not city_hint or match["city"] == city_hint), area_matches[0])
            centroid = self.area_centroids.get((area_match["city"], area_match["area"])) or self.city_centroids.get(area_match["city"])
            return ResolvedLocation(
                entity_text,
                label,
                area_match["city"],
                area_match["area"],
                centroid[0] if centroid else None,
                centroid[1] if centroid else None,
                "area_catalog",
            )

        unique_poi_matches = [value for key, value in self.poi_index.items() if key[1] == normalized]
        if len(unique_poi_matches) == 1:
            lat, lng, source = unique_poi_matches[0]
            return ResolvedLocation(entity_text, label, city_hint, area_hint, lat, lng, source)

        area_resolution = self.resolve_area(city_hint, area_hint)
        return ResolvedLocation(
            entity_text,
            label,
            str(area_resolution["resolved_city"]),
            str(area_resolution["resolved_area"]),
            area_resolution["lat"],
            area_resolution["lng"],
            str(area_resolution["resolution_source"]),
        )

    def resolve_entities(self, entities: list[dict[str, object]], city_hint: str = "", area_hint: str = "") -> list[dict[str, object]]:
        resolved: list[dict[str, object]] = []
        for entity in entities:
            item = self.resolve_entity(
                str(entity.get("entity", "")),
                label=str(entity.get("label", "")),
                city_hint=city_hint,
                area_hint=area_hint,
            )
            resolved.append(
                {
                    "entity": item.entity,
                    "label": item.label,
                    "resolved_city": item.resolved_city,
                    "resolved_area": item.resolved_area,
                    "lat": item.lat,
                    "lng": item.lng,
                    "resolution_source": item.resolution_source,
                }
            )
        return resolved
