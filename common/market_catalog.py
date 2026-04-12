from __future__ import annotations

import hashlib

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

QUERY_TEMPLATE_SPECS: list[dict[str, object]] = [
    {"template": "review jujur {business} {district} {city}", "intent": "review", "priority": 3},
    {"template": "{business} mahal banget {district} {city}", "intent": "complaint", "priority": 5},
    {"template": "{business} gak worth it {district} {city}", "intent": "complaint", "priority": 5},
    {"template": "{business} pelayanan lama {district} {city}", "intent": "complaint", "priority": 5},
    {"template": "{business} kurang enak {district} {city}", "intent": "complaint", "priority": 5},
    {"template": "{business} zonk {district} {city}", "intent": "complaint", "priority": 5},
    {"template": "{business} gak rekomen {district} {city}", "intent": "complaint", "priority": 5},
    {"template": "{business} porsi kecil {district} {city}", "intent": "complaint", "priority": 4},
    {"template": "{business} mengecewakan {district} {city}", "intent": "complaint", "priority": 4},
    {"template": "{business} ga sesuai harga {district} {city}", "intent": "complaint", "priority": 4},
    {"template": "{business} {district} {city}", "intent": "discovery", "priority": 3},
    {"template": "{business} dekat kos {district} {city}", "intent": "demand", "priority": 3},
    {"template": "{business} sekitar kampus {district} {city}", "intent": "demand", "priority": 3},
    {"template": "{business} wajib coba {district} {city}", "intent": "demand_present", "priority": 3},
    {"template": "{business} viral {district} {city}", "intent": "trend", "priority": 3},
    {"template": "{business} buka baru {district} {city}", "intent": "trend", "priority": 3},
    {"template": "review {business} {district} {city}", "intent": "review", "priority": 3},
    {"template": "kuliner malam {district} {city}", "intent": "discovery", "priority": 2},
    {"template": "tempat makan murah {district} {city}", "intent": "demand_present", "priority": 2},
    {"template": "jajanan {district} {city}", "intent": "discovery", "priority": 2},
    {"template": "{district} {city} review usaha", "intent": "review", "priority": 2},
    {"template": "belum ada {business} di {district} {city}", "intent": "demand", "priority": 3},
    {"template": "butuh {business} di {district} {city}", "intent": "demand", "priority": 3},
    {"template": "kapan buka {business} di {district} {city}", "intent": "demand", "priority": 3},
    {"template": "tolong buka {business} di {district} {city}", "intent": "demand", "priority": 3},
    {"template": "{business} kebanyakan di {district} {city}", "intent": "competition", "priority": 3},
    {"template": "{business} saingan banyak di {district} {city}", "intent": "competition", "priority": 3},
    {"template": "{business} dimana mana {district} {city}", "intent": "competition", "priority": 3},
    {"template": "{business} lagi hits di {district} {city}", "intent": "trend", "priority": 3},
    {"template": "{business} rame terus {district} {city}", "intent": "trend", "priority": 3},
    {"template": "ada banyak {business} di {district} {city}", "intent": "supply", "priority": 3},
    {"template": "{business} buka cabang di {district} {city}", "intent": "supply", "priority": 3},
]


def _stable_seed_order(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _build_template_pool(template_index: int, spec: dict[str, object]) -> list[dict[str, str]]:
    template = str(spec["template"])
    intent = str(spec["intent"])
    priority = int(spec["priority"])
    business_values = BUSINESS_HINTS if "{business}" in template else [""]
    seeds: list[dict[str, str]] = []
    for business in business_values:
        for city, districts in CITIES_KECAMATAN.items():
            for district in districts:
                query = template.format(business=business, district=district, city=city).strip()
                seeds.append(
                    {
                        "query": query,
                        "city": city,
                        "area_hint": district,
                        "business_hint": business,
                        "query_intent": intent,
                        "query_template": template,
                        "query_priority": str(priority),
                        "template_index": str(template_index),
                    }
                )
    seeds.sort(key=lambda seed: _stable_seed_order(seed["query"]))
    return seeds


def iter_query_seeds(max_queries: int | None = None) -> list[dict[str, str]]:
    pools = [
        _build_template_pool(template_index, spec)
        for template_index, spec in enumerate(QUERY_TEMPLATE_SPECS)
    ]
    priorities = sorted({int(spec["priority"]) for spec in QUERY_TEMPLATE_SPECS}, reverse=True)

    grouped_indices: dict[int, list[int]] = {priority: [] for priority in priorities}
    for template_index, spec in enumerate(QUERY_TEMPLATE_SPECS):
        grouped_indices[int(spec["priority"])].append(template_index)

    ordered: list[dict[str, str]] = []
    while any(pool for pool in pools):
        made_progress = False
        for priority in priorities:
            for template_index in grouped_indices[priority]:
                pool = pools[template_index]
                if not pool:
                    continue
                ordered.append(pool.pop(0))
                made_progress = True
                if max_queries is not None and len(ordered) >= max_queries:
                    return ordered
        if not made_progress:
            break
    return ordered
