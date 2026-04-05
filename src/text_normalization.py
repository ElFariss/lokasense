import re
import json
from pathlib import Path

# Load slang dictionary
BASE_DIR = Path(__file__).parent.parent
SLANG_FILE = BASE_DIR / "data" / "slang" / "slang_dict.json"

_slang_dict = {}

def load_slang_dict():
    """Load the slang dictionary into memory."""
    global _slang_dict
    if not _slang_dict and SLANG_FILE.exists():
        with open(SLANG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            _slang_dict.update(data.get("normalization", {}))
            _slang_dict.update(data.get("food_slang", {}))
    return _slang_dict

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Lowercase
    - Remove URLs, @mentions, #hashtags, and HTML tags
    - Replace multiple spaces with single space
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-alphanumeric (keep basic punctuation)
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    # Replace multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_slang(text: str) -> str:
    """Normalize slang words using the loaded dictionary."""
    slang_dict = load_slang_dict()
    if not slang_dict:
        return text
        
    words = text.split()
    normalized_words = []
    
    for word in words:
        # Strip trailing punctuation for the lookup
        clean_word = re.sub(r'[.,!?]$', '', word)
        
        if clean_word in slang_dict:
            # Replace the word but try to preserve the punctuation if there was any
            punct = word[len(clean_word):]
            normalized = slang_dict[clean_word]
            if normalized: # some slang maps to empty string (e.g. wkwk)
                normalized_words.append(normalized + punct)
        else:
            normalized_words.append(word)
            
    return " ".join(normalized_words)

def normalize_text(text: str) -> str:
    """Full normalization pipeline: clean -> de-slang."""
    text = clean_text(text)
    text = normalize_slang(text)
    # Final cleanup of double spaces that might be created by slang removal
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Tempatnya bagus bgt tp pelayanannya jelek #kecewa @owner",
        "Wkwkwk ayam gepreknya mantul bnyk porsinya jg 10k aja dong",
        "Di malang blum ada mixue yg deket kampus krn msh dibangun",
    ]
    
    print("Normalizer Test:")
    for t in test_texts:
        print(f"Original: {t}")
        print(f"Cleaned : {normalize_text(t)}\n")
