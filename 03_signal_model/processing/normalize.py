from common.text_normalization import (
    clean_text,
    load_slang_dict,
    normalize_slang,
    normalize_text,
)

__all__ = [
    "clean_text",
    "load_slang_dict",
    "normalize_slang",
    "normalize_text",
]


if __name__ == "__main__":
    test_texts = [
        "Tempatnya bagus bgt tp pelayanannya jelek #kecewa @owner",
        "Wkwkwk ayam gepreknya mantul bnyk porsinya jg 10k aja dong",
        "Di malang blum ada mixue yg deket kampus krn msh dibangun",
    ]

    print("Normalizer Test:")
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Cleaned : {normalize_text(text)}\n")
