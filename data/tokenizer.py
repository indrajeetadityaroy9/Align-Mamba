"""Tokenization utilities for German and English."""

import re
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def tokenize_german(text: str) -> list[str]:
    """Tokenize German text using simple rules."""
    text = text.lower().strip()
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


def tokenize_english(text: str) -> list[str]:
    """Tokenize English text using NLTK."""
    text = text.lower().strip()
    return nltk.word_tokenize(text)


def tokenize(text: str, lang: str) -> list[str]:
    """Tokenize text based on language."""
    if lang == 'german':
        return tokenize_german(text)
    elif lang == 'english':
        return tokenize_english(text)
    else:
        return text.lower().split()
