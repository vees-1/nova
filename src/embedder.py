"""
NOVA — Embedder
Converts product metadata and raw text queries into dense vectors.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, EMBEDDING_DIM


class Embedder:
    """
    Wraps SentenceTransformer for NOVA's embedding needs.
    Keeps the model loaded in memory for fast repeated inference.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim   = self.model.get_sentence_embedding_dimension()
        print(f"Embedder ready — dimension: {self.dim} ✅")

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        Embed a list of text strings.
        Returns numpy array of shape (len(texts), dim), dtype float32.
        """
        vectors = self.model.encode(
            texts,
            batch_size=256,
            show_progress_bar=len(texts) > 500,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    def embed_one(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single string. Returns 1D array of shape (dim,).
        """
        return self.embed([text], normalize=normalize)[0]

    @staticmethod
    def build_product_text(category: str, avg_price: float = None, weight_g: float = None, photos: int = 1) -> str:
        """
        Construct a structured text representation for a product.
        Mirrors the logic used in notebook 02.
        """
        category_str = str(category).replace("_", " ") if category else "unknown category"

        if avg_price is None:
            price_tier = "unknown price"
        elif avg_price < 50:
            price_tier = "budget"
        elif avg_price < 200:
            price_tier = "mid-range"
        else:
            price_tier = "premium"

        if weight_g is None:
            size_tier = "unknown size"
        elif weight_g < 500:
            size_tier = "small"
        elif weight_g < 5000:
            size_tier = "medium"
        else:
            size_tier = "large"

        photo_str = "multiple photos" if photos > 1 else "single photo"

        return f"{category_str} product, {price_tier} price range, {size_tier} item, {photo_str}"
