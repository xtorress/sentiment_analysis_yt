import pytest
from src.nltk_setup import NLTKSetup

from src.preprocessing import DataPreprocessor

nltk_setup1 = NLTKSetup()
nltk_setup1.download_resources()

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

def test_clean_text(preprocessor):
    text = "H3llo!!! everybody"
    cleaned = preprocessor.clean_text(text)
    assert "3" not in cleaned
    assert "!" not in cleaned
    assert "hllo everybody" == cleaned, f"Error: {cleaned}"

def test_tokenize(preprocessor):
    text = "hello world 42"
    tokens = preprocessor.tokenize(text)
    assert tokens == ["hello", "world", "42"]
 
def test_remove_stopwords(preprocessor):
    tokens = "this is an apple".split()
    preprocess = preprocessor.remove_stopwords(tokens)
    assert "this" not in preprocess
    assert "an" not in preprocess

def test_preprocess_integration(preprocessor):
    text = "Hello world this is a Test. !!_."
    preprocess = preprocessor.preprocess(text)
    assert "Hello" not in preprocess
    assert "." not in preprocess
    assert "hello world test" == preprocess 