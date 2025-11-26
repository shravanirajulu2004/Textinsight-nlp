import re
import string
from typing import List, Tuple

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Make sure required NLTK data is available
def init_nltk():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def basic_clean(text: str) -> str:
    """Lowercase, remove extra spaces and punctuation."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def split_documents(raw_text: str) -> List[str]:
    """
    Split input text into documents.

    Strategy:
    - If there are blank lines -> use blank lines as document separators.
    - Else -> treat each non-empty line as a document.
    """
    # Split by double newlines first
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]

    if len(parts) > 1:
        return parts

    # Fallback: split by single lines
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    return lines


def build_tfidf(docs: List[str]) -> Tuple[TfidfVectorizer, any]:
    """
    Build a TF-IDF vectorizer and transform documents.
    Returns (vectorizer, tfidf_matrix).
    """
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix


def cluster_documents(tfidf_matrix, n_clusters: int) -> KMeans:
    """
    Run KMeans clustering on the tfidf_matrix.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )
    model.fit(tfidf_matrix)
    return model


def extract_top_keywords_per_cluster(
    model: KMeans,
    vectorizer: TfidfVectorizer,
    top_n: int = 8
) -> dict:
    """
    For each cluster, extract top_n keywords based on cluster centers.
    Returns dict: {cluster_id: [keywords]}
    """
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for cluster_id, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[::-1][:top_n]
        cluster_keywords[cluster_id] = [terms[i] for i in top_indices]

    return cluster_keywords


def compute_sentiment_scores(docs: List[str]) -> List[float]:
    """
    Compute sentiment scores using NLTK VADER (compound score).
    Range: [-1, 1], where >0 is positive, <0 is negative.
    """
    init_nltk()
    sia = SentimentIntensityAnalyzer()
    scores = []
    for text in docs:
        if not text.strip():
            scores.append(0.0)
        else:
            scores.append(sia.polarity_scores(text)["compound"])
    return scores
