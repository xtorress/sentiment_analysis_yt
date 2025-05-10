from src.preprocessing import DataPreprocessor
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np
import logging

class Vectorizer():
    """
    A flexible text vectorization class suporting TF-IDF, Glove and SBERT
    embeddings.
    """
    def __init__(self, method: str):
        method = method.lower()
        valid_methods = ["tfidf", "glove", "sbert"]
        if method is not valid_methods:
            raise ValueError(f"El método debe de ser una de: {valid_methods}")

        self.method = method
        self.vectorizer = None
        self.model = None

        if method == "tfidf":
            self.vectorizer = TfidfVectorizer()
        elif method == "glove":
            self.model = self._load_glove_model()
        elif method == "sbert":
            pass

    def fit(self, tokens):
        """Train vectorizer."""
        if self.method == "tfidf":
            return self.vectorizer.fit(tokens)

    def transform(self, tokens):
        if self.method == 'tfid':
            return self.vectorizer.transform(tokens).toarray()
        elif self.method == 'glove':
            return np.mean([self.model[token] for token in tokens if token in self.model], axis=0)

    def fit_transform(self, tokens):
        """Fit and transform data."""
        if self.method == "tfidf":
            return self.vectorizer.fit_transform()
        return self.transform(tokens)

    def _load_glove_model(self, paht_file='../data/glove.6B.100d.txt'):
        """Save glove's embeddings."""
        print("Cargando glove.")
        embeddings = {}
        with open(paht_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        print(f"Carga finalizada. Se guardo un total de {len(embeddings)}")
        return embeddings
