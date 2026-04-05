from __future__ import annotations

import hashlib
from itertools import islice

CITIES_KECAMATAN: dict[str, list[str]] = {
    "Malang": ["Lowokwaru", "Klojen", "Blimbing", "Sukun", "Kedungkandang"],
    "Surabaya": ["Gubeng", "Tegalsari", "Genteng", "Wonokromo", "Rungkut"],
    "Yogyakarta": ["Gondokusuman", "Umbulharjo", "Kotagede", "Mergangsan", "Danurejan"],
    "Bandung": ["Coblong", "Bandung Wetan", "Sumur Bandung", "Cicendo", "Lengkong"],
    "Semarang": ["Semarang Tengah", "Semarang Selatan", "Candisari", "Gajahmungkur", "Banyumanik"],
}

BUSINESS_HINTS: list[str] = [
    "ayam geprek",
    "cafe",
    "kedai kopi",
    "kopi",
    "kuliner",
    "kedai makan",
    "mie",
    "laundry",
    "jajanan kaki lima",
    "bakso",
    "seblak",
    "warung makan",
    "es teh",
    "nasi padang",
    "ayam bakar",
    "soto",
]

QUERY_TEMPLATES: list[str] = [
    "{business} {district} {city}",
    "{business} dekat kos {district} {city}",
    "{business} sekitar kampus {district} {city}",
    "{business} wajib coba {district} {city}",
    "{business} viral {district} {city}",
    "{business} buka baru {district} {city}",
    "review {business} {district} {city}",
    "kuliner malam {district} {city}",
    "tempat makan murah {district} {city}",
    "jajanan {district} {city}",
    "{district} {city} review usaha",
    "belum ada {business} di {district} {city}",
    "butuh {business} di {district} {city}",
    "kapan buka {business} di {district} {city}",
    "tolong buka {business} di {district} {city}",
    "{business} kebanyakan di {district} {city}",
    "{business} saingan banyak di {district} {city}",
    "{business} dimana mana {district} {city}",
    "{business} mahal di {district} {city}",
    "{business} gak enak {district} {city}",
    "{business} mengecewakan {district} {city}",
    "{business} lagi hits di {district} {city}",
    "{business} rame terus {district} {city}",
    "ada banyak {business} di {district} {city}",
    "{business} buka cabang di {district} {city}",
]


def iter_query_seeds(max_queries: int | None = None) -> list[dict[str, str]]:
    seeds: list[dict[str, str]] = []
    for template in QUERY_TEMPLATES:
        business_values = BUSINESS_HINTS if "{business}" in template else [""]
        for business in business_values:
            for city, districts in CITIES_KECAMATAN.items():
                for district in districts:
                    seeds.append(
                        {
                            "query": template.format(business=business, district=district, city=city).strip(),
                            "city": city,
                            "area_hint": district,
                            "business_hint": business,
                        }
                    )
    seeds.sort(key=lambda seed: hashlib.sha256(seed["query"].encode("utf-8")).hexdigest())
    if max_queries is not None:
        return list(islice(seeds, max_queries))
    return seeds
