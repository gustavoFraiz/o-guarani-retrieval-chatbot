"""Core retrieval pipeline extracted from the recovered O Guarani notebook."""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    text = re.sub(r"/", "", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def chunk_tokens(tokens, chunk_size=200, overlap_ratio=0.5):
    overlap = int(chunk_size * overlap_ratio)
    step = max(1, chunk_size - overlap)
    return [
        " ".join(tokens[i:i + chunk_size])
        for i in range(0, len(tokens), step)
        if tokens[i:i + chunk_size]
    ]


class TfidfRetriever:
    def __init__(self, chunks, processed_chunks=None):
        self.chunks = chunks
        self.processed_chunks = processed_chunks or chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.processed_chunks)

    def retrieve(self, processed_query, top_k=3, threshold=0.1):
        vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(vector, self.matrix).flatten()
        indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            (self.chunks[i], float(similarities[i]))
            for i in indices
            if similarities[i] > threshold
        ]
