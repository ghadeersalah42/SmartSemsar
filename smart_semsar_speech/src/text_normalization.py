"""
Egyptian Arabic Text Normalization Module
Specially designed for Egyptian Conversational & Dialectal Speech Recognition
"""

import re
from typing import List, Optional

# Mapping of Arabic-Indic numerals to Western digits
ARABIC_INDIC_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

# Diacritics Unicode range (Tashkeel, Tanween, Shadda, Sukun)
DIACRITICS_REGEX = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

# Punctuation regex (Arabic & Latin punctuation, brackets, quotes)
PUNCTUATION_REGEX = re.compile(r"[،؛؟!\.,:;\"'״׳\(\)\[\]\{\}—\-_/\\]+")


def remove_diacritics(text: str) -> str:
    """Removes all Arabic diacritics (Harakat, Tanween, Shadda, Sukun)."""
    return DIACRITICS_REGEX.sub("", text)


def normalize_digits(text: str) -> str:
    """Converts Eastern Arabic (Hindi) digits ٠-٩ to standard 0-9."""
    for arabic_digit, western_digit in ARABIC_INDIC_DIGITS.items():
        text = text.replace(arabic_digit, western_digit)
    return text


def normalize_arabic_text(
    text: str,
    normalize_taa_marbuta: bool = True,
    remove_punct: bool = True,
    to_lower: bool = True,
) -> str:
    """
    Normalizes Egyptian Arabic text for fair, robust ASR evaluation (WER/CER).
    
    Operations:
    1. Removes Tatweel (Kashida 'ـ')
    2. Normalizes Alef variations (أ, إ, آ, ٱ -> ا)
    3. Normalizes Yeh variants (ى -> ي)
    4. Normalizes Taa Marbuta (ة -> ه) when normalize_taa_marbuta is True
    5. Removes diacritics
    6. Removes punctuation
    7. Normalizes digits
    8. Normalizes multiple whitespaces into a single space
    """
    if not text:
        return ""
    
    text = str(text)
    
    # Remove Tatweel
    text = text.replace("ـ", "")
    
    # Normalize Alef variants
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ٱ", "ا")
    
    # Normalize Yeh variants (Alef Maqsura to Yeh)
    text = text.replace("ى", "ي")
    
    # Normalize Taa Marbuta to Haa (widely interchangeable in Egyptian Arabic writing)
    if normalize_taa_marbuta:
        text = text.replace("ة", "ه")
        
    # Remove diacritics
    text = remove_diacritics(text)
    
    # Normalize digits
    text = normalize_digits(text)
    
    # Remove punctuation
    if remove_punct:
        text = PUNCTUATION_REGEX.sub(" ", text)
        
    # Lowercase Latin characters if any
    if to_lower:
        text = text.lower()
        
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def tokenize_arabic_words(text: str) -> List[str]:
    """Tokenizes normalized text into a list of words."""
    normalized = normalize_arabic_text(text)
    return [w for w in normalized.split() if w]


def tokenize_arabic_chars(text: str, include_spaces: bool = True) -> List[str]:
    """Tokenizes normalized text into a list of characters."""
    normalized = normalize_arabic_text(text)
    if not include_spaces:
        return [c for c in normalized if c != " "]
    return list(normalized)
