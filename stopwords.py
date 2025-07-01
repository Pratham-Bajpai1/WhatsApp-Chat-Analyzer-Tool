from typing import Set, Optional
import os

class Stopwords:
    """
    Loads and caches stopwords from file or set.
    Supports fallback and customization.
    """
    _cache = {}

    def __init__(self, stopword_file: Optional[str] = None, default_set: Optional[Set[str]] = None):
        """
        Args:
            stopword_file: Path to stopword file (one word per line)
            default_set: Optional fallback set if file missing
        """
        self.stopword_file = stopword_file
        self.default_set = default_set or set()
        self._stopwords = None

    def load(self) -> Set[str]:
        """Loads stopwords from file or uses default set, caches in memory."""
        if self._stopwords is not None:
            return self._stopwords
        if self.stopword_file and os.path.exists(self.stopword_file):
            try:
                with open(self.stopword_file, 'r', encoding='utf-8') as f:
                    self._stopwords = set(w.strip() for w in f if w.strip())
                    Stopwords._cache[self.stopword_file] = self._stopwords
                    return self._stopwords
            except Exception as e:
                print(f"Failed to load stopwords from {self.stopword_file}: {e}")
        # Fallback
        self._stopwords = self.default_set
        return self._stopwords

    def add(self, word: str):
        """Add a word to the stopword set."""
        self.load().add(word)

    def remove(self, word: str):
        """Remove a word from the stopword set if present."""
        self.load().discard(word)

    def is_stopword(self, word: str) -> bool:
        """Check if a word is in the stopword set."""
        return word in self.load()