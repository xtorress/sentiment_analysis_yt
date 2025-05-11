import pytest
import numpy as np
from src.vectorization import Vectorizer


# ---------- TF-IDF TESTS ----------

def test_tfidf_fit_transform():
    texts = ["this is a test", "another test"]
    vect = Vectorizer("tfidf")
    vect.fit(texts)
    result = vect.transform(texts)
    assert result.shape[0] == 2  # Dos documentos
    assert result.shape[1] > 0   # Alguna dimensión

def test_tfidf_fit_transform_combined():
    texts = ["sample one", "sample two"]
    vect = Vectorizer("tfidf")
    result = vect.fit_transform(texts)
    assert result.shape[0] == 2

# # ---------- GLOVE TESTS ----------

# @patch("your_module.Vectorizer._load_glove_model")
# def test_glove_vectorizer_mean_vector(mock_glove_loader):
#     mock_glove_loader.return_value = {
#         "hello": np.array([1.0, 2.0]),
#         "world": np.array([3.0, 4.0])
#     }

#     vect = Vectorizer("glove")
#     result = vect.transform("hello world")
#     expected = np.array([2.0, 3.0])
#     np.testing.assert_array_equal(result, expected)

# # ---------- SBERT TESTS ----------

# @patch("your_module.SentenceTransformer")
# def test_sbert_vectorizer(mock_model):
#     dummy_encoder = MagicMock()
#     dummy_encoder.encode.return_value = np.array([0.5] * 384)
#     mock_model.return_value = dummy_encoder

#     vect = Vectorizer("sbert")
#     result = vect.transform("This is SBERT test")
#     assert isinstance(result, np.ndarray)
#     assert result.shape[0] == 384