"""Constants shared across the package and the deploy script, defined once.

Importing this module is cheap: scikit-learn is imported only when a vectoriser is built.
"""

from __future__ import annotations

# The model the few-shot classifier is built for and the deploy script serves.
MODEL_ID = "google/flan-ul2"
# FLAN-UL2 takes 2048 input tokens. The 16 extra tokens in the serving limit hold the generated 5-digit code.
MAX_INPUT_TOKENS = 2048
MAX_TOTAL_TOKENS = 2064
RESERVED_TOKENS = MAX_TOTAL_TOKENS - MAX_INPUT_TOKENS


def make_vectorizer():
    """The TF-IDF vectoriser used everywhere: split grouping, baselines and example retrieval."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
