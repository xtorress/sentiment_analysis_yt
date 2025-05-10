import logging
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from typing import List
from nltk_setup import NLTKSetup

nltk_setup = NLTKSetup()
nltk_setup.download_resources()

class DataPreprocessor:
    def __init__(self):
        self.language = "english"
        self.stop_words = stopwords.words(self.language)

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"http\S+|www\S+|@\w+|#", "", text)
        text = text.strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    def preprocess(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        return " ".join(tokens)

