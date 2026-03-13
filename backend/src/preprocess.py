"""
=============================================================================
preprocess.py — Text cleaning for English + Nepali/Hindi Roman script
=============================================================================

WHY CUSTOM PREPROCESSING?
  Standard NLP tools (NLTK, spaCy) don't understand romanized Nepali/Hindi.
  Code-mixed text like "vayo ni yaar, good video" needs special handling:
    • Keep romanized words as-is (model learns them)
    • Normalize repeated characters ("hahahahaha" → "haha")
    • Handle Devanagari script (convert or drop)
    • Remove noise (URLs, mentions, hashtags, spam patterns)
"""

import re
import string
import unicodedata
import pickle
from pathlib import Path
from typing import List, Optional


# ── Common romanized Nepali stopwords ──────────────────────────────────────
NEPALI_ROMAN_STOPWORDS = {
    'ko', 'ka', 'ki', 'ma', 'le', 'lai', 'bata', 'sanga', 'haru',
    'yo', 'yो', 'tyo', 'ra', 'ani', 'pani', 'nai', 'ta', 'ni',
    'ho', 'cha', 'chha', 'chhe', 'thiyo', 'bhayo', 'garyo', 'garne',
    'garnuhos', 'please', 'garna', 'garnu', 'xa', 'xaina', 'chaina',
    'k', 'kina', 'kasari', 'kahile', 'kaha', 'kun', 'ke',
    'aaja', 'hijo', 'bholi', 'aile', 'sab', 'sabai', 'ali',
    'dherai', 'thora', 'ekdam', 'ekdamai',
}

# Common spam patterns in YouTube comments
SPAM_PATTERNS = [
    r'sub\s*4\s*sub',
    r'sub\s*back',
    r'check\s*out\s*my\s*channel',
    r'visit\s*my\s*channel',
    r'please\s*subscribe',
    r'first\s*comment',
    r'\b(?:f+i+r+s+t+|1st)\s+comment\b',
]


class TextCleaner:
    """
    Comprehensive text cleaner for multilingual YouTube comments.
    Handles: English, Nepali Roman, Hindi Roman, code-mixed text.
    """

    def __init__(
        self,
        remove_stopwords: bool = False,   # Usually False for deep learning
        normalize_repeats: bool = True,
        keep_emojis_as_text: bool = True,
        remove_spam: bool = True,
        min_length: int = 3,
    ):
        self.remove_stopwords   = remove_stopwords
        self.normalize_repeats  = normalize_repeats
        self.keep_emojis_as_text = keep_emojis_as_text
        self.remove_spam        = remove_spam
        self.min_length         = min_length

        # Precompile all regexes for speed
        self._url_re      = re.compile(r'https?://\S+|www\.\S+')
        self._mention_re  = re.compile(r'@\w+')
        self._hashtag_re  = re.compile(r'#(\w+)')
        self._html_re     = re.compile(r'<[^>]+>')
        self._devanagari_re = re.compile(r'[\u0900-\u097F]+')
        self._repeat_re   = re.compile(r'(.)\1{2,}')         # aaa → aa (keeps doubles)
        self._repeat_word_re = re.compile(r'\b(\w+)(?:\s+\1){2,}\b')  # word word word → word
        self._whitespace_re = re.compile(r'\s+')
        self._punct_re    = re.compile(r'[^\w\s]')
        self._spam_re     = re.compile('|'.join(SPAM_PATTERNS), re.IGNORECASE)
        self._laugh_re    = re.compile(r'\b(?:ha){3,}\b|\b(?:he){3,}\b|\b(?:hi){3,}\b|\b(?:lol+)\b', re.IGNORECASE)
        self._number_re   = re.compile(r'\b\d+\b')

        # Emoji to sentiment hint mapping (common YouTube emojis)
        self._emoji_map = {
            '❤️': ' love ', '😍': ' love ', '🥰': ' love ',
            '😂': ' funny ', '🤣': ' funny ',
            '😭': ' sad ', '💔': ' sad ',
            '😡': ' angry ', '🤬': ' angry ',
            '👍': ' good ', '🔥': ' amazing ',
            '💯': ' perfect ', '⭐': ' good ',
            '😒': ' disappointed ', '😤': ' frustrated ',
            '🙏': ' respect ', '❌': ' bad ', '✅': ' good ',
        }

    def clean(self, text: str) -> str:
        """Clean a single comment."""
        if not isinstance(text, str) or not text.strip():
            return ''

        # 1. Normalize unicode
        text = unicodedata.normalize('NFKC', text)

        # 2. HTML entities
        text = self._html_re.sub(' ', text)

        # 3. Emoji handling
        if self.keep_emojis_as_text:
            for emoji, replacement in self._emoji_map.items():
                text = text.replace(emoji, replacement)

        # 4. Remove URLs
        text = self._url_re.sub(' ', text)

        # 5. Remove mentions
        text = self._mention_re.sub(' ', text)

        # 6. Hashtags — keep the word, remove the #
        text = self._hashtag_re.sub(r' \1 ', text)

        # 7. Remove/transliterate Devanagari
        #    (keep it — the model won't understand it well anyway, better remove)
        text = self._devanagari_re.sub(' ', text)

        # 8. Lowercase
        text = text.lower()

        # 9. Spam detection — mark as neutral noise
        if self.remove_spam and self._spam_re.search(text):
            return ''

        # 10. Normalize laughs
        text = self._laugh_re.sub(' laugh ', text)

        # 11. Remove numbers (generally not useful for sentiment)
        text = self._number_re.sub(' ', text)

        # 12. Remove punctuation (keep spaces)
        text = re.sub(r'[^\w\s]', ' ', text)

        # 13. Normalize repeated characters: "sooooo" → "soo"
        if self.normalize_repeats:
            text = self._repeat_re.sub(r'\1\1', text)

        # 14. Remove repeated words: "good good good" → "good"
        text = self._repeat_word_re.sub(r'\1', text)

        # 15. Remove stopwords (optional — usually disabled for DL)
        if self.remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in NEPALI_ROMAN_STOPWORDS]
            text = ' '.join(words)

        # 16. Collapse whitespace
        text = self._whitespace_re.sub(' ', text).strip()

        # 17. Length filter
        if len(text) < self.min_length:
            return ''

        return text

    def batch_clean(
        self,
        texts        : List[str],
        show_progress: bool = False,
        n_workers    : int  = 4,
    ) -> List[str]:
        """
        Clean texts in parallel using threads.
        Regex operations release Python's GIL so threads genuinely overlap.
        Falls back to single-threaded for batches under 500 (overhead not worth it).
        """
        if len(texts) < 500 or n_workers <= 1:
            if show_progress:
                try:
                    from tqdm import tqdm
                    return [self.clean(t) for t in tqdm(texts, desc='Cleaning')]
                except ImportError:
                    pass
            return [self.clean(t) for t in texts]

        from concurrent.futures import ThreadPoolExecutor

        chunk_size = max(1, len(texts) // n_workers)
        chunks     = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        results    = [None] * len(chunks)

        def clean_chunk(args):
            idx, chunk = args
            return idx, [self.clean(t) for t in chunk]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for idx, cleaned_chunk in executor.map(clean_chunk, enumerate(chunks)):
                results[idx] = cleaned_chunk

        flattened = []
        for chunk in results:
            flattened.extend(chunk)
        return flattened

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'TextCleaner':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── TF-IDF Feature Extraction ──────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
import numpy as np
import scipy.sparse as sp


class TFIDFExtractor:
    """
    Combined word + character TF-IDF for text classification.

    WHY CHARACTER N-GRAMS?
      Romanized Nepali: "ramro", "raaam", "ramrooo" all mean "good/nice"
      Character n-grams handle spelling variants better than word-level features.
      This is critical for code-mixed text where spelling is inconsistent.
    """

    def __init__(self, config=None):
        if config is None:
            from config import TFIDFConfig
            config = TFIDFConfig

        self.word_vectorizer = TfidfVectorizer(
            max_features   = config.WORD_MAX_FEATURES,
            ngram_range    = config.WORD_NGRAM_RANGE,
            min_df         = config.WORD_MIN_DF,
            max_df         = config.WORD_MAX_DF,
            sublinear_tf   = config.WORD_SUBLINEAR_TF,
            analyzer       = 'word',
            token_pattern  = r'\b\w+\b',
        )

        self.char_vectorizer = TfidfVectorizer(
            max_features   = config.CHAR_MAX_FEATURES,
            ngram_range    = config.CHAR_NGRAM_RANGE,
            min_df         = config.CHAR_MIN_DF,
            max_df         = config.CHAR_MAX_DF,
            sublinear_tf   = config.CHAR_SUBLINEAR_TF,
            analyzer       = 'char_wb',
        )

        self.word_weight = config.WORD_WEIGHT
        self.char_weight = config.CHAR_WEIGHT
        self.fitted = False

    def fit(self, texts: List[str]) -> 'TFIDFExtractor':
        print('[TFIDF] Fitting word vectorizer...')
        self.word_vectorizer.fit(texts)
        print(f'[TFIDF] Word vocab size: {len(self.word_vectorizer.vocabulary_)}')

        print('[TFIDF] Fitting char vectorizer...')
        self.char_vectorizer.fit(texts)
        print(f'[TFIDF] Char vocab size: {len(self.char_vectorizer.vocabulary_)}')

        self.fitted = True
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        if not self.fitted:
            raise RuntimeError('TFIDFExtractor must be fitted before transform.')

        word_features = self.word_vectorizer.transform(texts)
        char_features = self.char_vectorizer.transform(texts)

        # Weighted hstack
        combined = sp.hstack([
            word_features * self.word_weight,
            char_features * self.char_weight,
        ])
        return combined

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        self.fit(texts)
        return self.transform(texts)

    def save(self, word_path: str, char_path: str):
        with open(word_path, 'wb') as f:
            pickle.dump(self.word_vectorizer, f)
        with open(char_path, 'wb') as f:
            pickle.dump(self.char_vectorizer, f)
        print(f'[TFIDF] Saved vectorizers → {word_path}, {char_path}')

    @classmethod
    def load(cls, word_path: str, char_path: str) -> 'TFIDFExtractor':
        obj = cls.__new__(cls)
        with open(word_path, 'rb') as f:
            obj.word_vectorizer = pickle.load(f)
        with open(char_path, 'rb') as f:
            obj.char_vectorizer = pickle.load(f)
        obj.word_weight = 0.7
        obj.char_weight = 0.3
        obj.fitted = True
        return obj


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cleaner = TextCleaner()
    test_comments = [
        "This video is absolutely amazing! Loved it so much 🔥",
        "vayo ni yaar ekdam ramro video thiyo",
        "kasto bakwas video ho yaar, time waste bhayo",
        "Good content keep it up bro 👍",
        "sub 4 sub please visit my channel",
        "राम्रो भिडियो",                   # Devanagari (will be stripped)
        "😂😂😂 hahahahaha so funny yaar",
    ]
    cleaned = cleaner.batch_clean(test_comments)
    for orig, clean in zip(test_comments, cleaned):
        print(f'  Original : {orig}')
        print(f'  Cleaned  : {clean}')
        print()
