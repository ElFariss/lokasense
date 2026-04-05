from __future__ import annotations

TARGET_ENTITY_MAP: dict[str, str] = {
    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "PLACE": "LOC",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "COMPANY": "ORG",
    "PER": "PER",
    "PERSON": "PER",
}

DROP_ENTITY_TYPES: set[str] = {
    "DATE",
    "TIME",
    "QUANTITY",
    "CARDINAL",
    "ORDINAL",
    "MONEY",
    "PERCENT",
}


def normalize_ner_tag(tag: str) -> str:
    if not isinstance(tag, str):
        return "O"
    cleaned = tag.strip().upper()
    if not cleaned or cleaned == "O":
        return "O"
    if "-" not in cleaned:
        return "O"

    prefix, entity = cleaned.split("-", 1)
    if prefix not in {"B", "I"}:
        return "O"
    if entity in DROP_ENTITY_TYPES:
        return "O"

    mapped = TARGET_ENTITY_MAP.get(entity)
    if not mapped:
        return "O"
    return f"{prefix}-{mapped}"


def normalize_ner_tags(tags: list[str]) -> list[str]:
    return [normalize_ner_tag(tag) for tag in tags]


def has_entity(tags: list[str]) -> bool:
    return any(tag != "O" for tag in tags)
