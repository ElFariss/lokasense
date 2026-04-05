from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SLANG_FILE = BASE_DIR / "data" / "slang" / "slang_dict.json"

_slang_dict: dict[str, str] = {}

INDONESIAN_HINT_WORDS: set[str] = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "karena",
    "banget", "bgt", "murah", "enak", "mahal", "ga", "gak", "nggak", "udah", "sudah",
    "belum", "deket", "dekat", "sekitar", "anak", "kos", "tempat", "makan", "jualan",
    "wajib", "coba", "ramai", "rame", "buka", "baru", "kota", "jalan", "jl", "kec",
    "harga", "pakai", "sama", "cari", "buat", "lagi", "disini", "di sini", "mantap",
    "murmer", "murah", "cocok", "banget", "nih", "aja", "aku", "saya", "kami", "kalian",
    "malang", "surabaya", "bandung", "yogyakarta", "jogja", "semarang", "lowokwaru",
    "klojen", "gubeng", "umbulharjo", "coblong", "candisari",
}

ENGLISH_HINT_WORDS: set[str] = {
    "the", "and", "is", "are", "was", "were", "be", "to", "of", "for", "with", "very",
    "good", "bad", "food", "place", "service", "price", "expensive", "cheap", "really",
    "taste", "cold", "hot", "this", "that", "would", "should", "great", "nice", "okay",
    "recommend", "owner", "employee", "forgot", "small", "large", "group", "road", "order",
    "rice", "chicken", "meat", "review", "favorite", "best", "worst", "disappointed",
}

INDONESIAN_SLANG_HINTS: set[str] = {
    "banget", "bgt", "murmer", "mantul", "mantap", "kak", "ga", "gak", "nggak", "udah",
    "deket", "anak", "kos", "murah", "enak", "wkwk", "wkwkwk", "nih", "aja", "dong",
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u27BF"
    "]+",
    flags=re.UNICODE,
)


def load_slang_dict() -> dict[str, str]:
    """Load the shared Indonesian slang normalization dictionary."""
    global _slang_dict
    if _slang_dict or not SLANG_FILE.exists():
        return _slang_dict

    with open(SLANG_FILE, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    _slang_dict.update(payload.get("normalization", {}))
    _slang_dict.update(payload.get("food_slang", {}))
    return _slang_dict


def extract_hashtags(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    seen: set[str] = set()
    hashtags: list[str] = []
    for tag in re.findall(r"#([\w_]+)", text.lower()):
        if tag and tag not in seen:
            seen.add(tag)
            hashtags.append(tag)
    return hashtags


def strip_emoji(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = EMOJI_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Normalize a noisy social-media string into classifier-friendly text.
    """
    if not isinstance(text, str):
        return ""

    text = strip_emoji(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"[^a-z0-9\s.,!?/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slang(text: str) -> str:
    slang_dict = load_slang_dict()
    if not slang_dict:
        return text

    normalized_words: list[str] = []
    for word in text.split():
        cleaned = re.sub(r"[.,!?]$", "", word)
        suffix = word[len(cleaned):]
        replacement = slang_dict.get(cleaned, cleaned)
        if replacement:
            normalized_words.append(f"{replacement}{suffix}")
    return " ".join(normalized_words)


def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = normalize_slang(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_dedupe(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def language_scores(text: str) -> dict[str, int]:
    tokens = re.findall(r"[a-z]+", normalize_text(text))
    token_set = set(tokens)
    return {
        "id": len(token_set & INDONESIAN_HINT_WORDS),
        "en": len(token_set & ENGLISH_HINT_WORDS),
        "slang": len(token_set & INDONESIAN_SLANG_HINTS),
        "tokens": len(tokens),
    }


def is_probably_indonesian(text: str, *, strict: bool = False) -> bool:
    scores = language_scores(text)
    id_score = scores["id"] + scores["slang"]
    en_score = scores["en"]

    if scores["tokens"] < 3:
        return False
    if id_score >= 3 and id_score >= en_score:
        return True
    if strict:
        return id_score >= 2 and id_score > en_score
    if id_score >= 2 and en_score <= id_score + 1:
        return True
    if id_score >= 1 and scores["slang"] >= 1 and en_score <= 2:
        return True
    return False


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    if not isinstance(text, str):
        return tokens
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens
