from __future__ import annotations

from dataclasses import dataclass

from common.market_catalog import BUSINESS_HINTS, CITIES_KECAMATAN
from common.text_normalization import normalize_text

DEFAULT_CITY = "Malang"
DEFAULT_BUSINESS = "kuliner"

BUSINESS_ALIAS_MAP = {
    "coffee": "kedai kopi",
    "kopi": "kedai kopi",
    "kedai kopi": "kedai kopi",
    "cafe": "cafe",
    "kafe": "cafe",
    "warung": "warung makan",
    "rumah makan": "warung makan",
    "restoran": "warung makan",
    "ayam": "ayam geprek",
    "geprek": "ayam geprek",
    "fried chicken": "ayam geprek",
    "laundri": "laundry",
    "laundry": "laundry",
    "minuman": "es teh",
    "teh": "es teh",
    "mie ayam": "mie",
    "baso": "bakso",
}


@dataclass(slots=True)
class IntentResult:
    city: str
    kecamatan_scope: list[str]
    business_type: str
    raw_query: str
    fallback_used: bool


def parse_query(query: str) -> IntentResult:
    normalized = normalize_text(query).lower()
    fallback_used = False

    detected_city = ""
    for city in CITIES_KECAMATAN:
        if city.lower() in normalized:
            detected_city = city
            break

    detected_districts: list[str] = []
    district_city = ""
    for city, districts in CITIES_KECAMATAN.items():
        for district in districts:
            if district.lower() in normalized:
                if not district_city:
                    district_city = city
                if city == district_city and district not in detected_districts:
                    detected_districts.append(district)

    if not detected_city and district_city:
        detected_city = district_city

    if not detected_city:
        detected_city = DEFAULT_CITY
        fallback_used = True

    detected_business = ""
    for hint in sorted(BUSINESS_HINTS, key=len, reverse=True):
        if hint.lower() in normalized:
            detected_business = hint
            break

    if not detected_business:
        for alias, canonical in sorted(BUSINESS_ALIAS_MAP.items(), key=lambda item: len(item[0]), reverse=True):
            if alias in normalized:
                detected_business = canonical
                break

    if not detected_business:
        detected_business = DEFAULT_BUSINESS
        fallback_used = True

    if detected_districts:
        scope = detected_districts
    else:
        scope = list(CITIES_KECAMATAN[detected_city])

    return IntentResult(
        city=detected_city,
        kecamatan_scope=scope,
        business_type=detected_business,
        raw_query=query,
        fallback_used=fallback_used,
    )
