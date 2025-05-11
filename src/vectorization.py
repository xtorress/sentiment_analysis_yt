from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import logging
from enum import Enum

class VectMethod(Enum):
    TFIDF = "tfidf"
    GLOVE = "glove"
    SBERT = "sbert"

class Vectorizer:
    """
    A flexible text vectorization class suporting TF-IDF, Glove and SBERT
    embeddings.
    """
    def __init__(self, method: str):
        try:
            self.method = VectMethod(method.lower())
        except ValueError:
            raise ValueError(f"Método no soportado, debe ser: {[m for m in VectMethod]}")
        
        self.vectorizer = None
        self.model = None

        if self.method == VectMethod.TFIDF:
            self.vectorizer = TfidfVectorizer()
        elif self.method == VectMethod.GLOVE:
            self.model = self._load_glove_model()
        elif self.method == VectMethod.SBERT:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit(self, tokens):
        """Train vectorizer."""
        if self.method == VectMethod.TFIDF:
            return self.vectorizer.fit(tokens)

    def transform(self, tokens):
        if self.method == VectMethod.TFIDF:
            return self.vectorizer.transform(tokens)
        elif self.method == VectMethod.GLOVE:
            tokens = tokens.split()
            return np.mean([self.model[token] for token in tokens if token in self.model], axis=0)
        elif self.method == VectMethod.SBERT:
            return self.model.encode(tokens)

    def fit_transform(self, tokens):
        """Fit and transform data."""
        if self.method == VectMethod.TFIDF:
            return self.vectorizer.fit_transform(tokens)
        return self.transform(tokens)

    def _load_glove_model(self, file_path='../data/glove.6B.100d.txt'):
        """Save glove's embeddings."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"GloVe file not found at: {file_path}. \
                                    Please download it and place it in the 'data/' folder.")
        print("Cargando glove.")
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        print(f"Carga finalizada. Se guardo un total de {len(embeddings)}")
        return embeddings
