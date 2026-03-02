"""
NOVA — FAISS Index Wrapper
Loads the pre-built index and product assets, exposes a clean search interface.
"""

import numpy as np
import pandas as pd
import faiss
from pathlib import Path

from src.config import (
    INDEX_FILE,
    PRODUCT_IDS_FILE,
    EMBEDDINGS_FILE,
    METADATA_FILE,
    CENTROIDS_FILE,
)


class ProductIndex:
    """
    Wraps the FAISS index with product metadata for convenient retrieval.
    All search methods return pandas DataFrames.
    """

    def __init__(self):
        print("Loading NOVA product index...")

        self.index       = faiss.read_index(str(INDEX_FILE))
        self.product_ids = np.load(PRODUCT_IDS_FILE, allow_pickle=True)
        self.embeddings  = np.load(EMBEDDINGS_FILE).astype("float32")
        self.metadata    = pd.read_csv(METADATA_FILE)
        self.centroids   = pd.read_csv(CENTROIDS_FILE, index_col="category")

        # Fast lookup: product_id → integer index
        self.pid_to_idx  = {pid: i for i, pid in enumerate(self.product_ids)}

        print(f"Index loaded ✅")
        print(f"  Products:   {self.index.ntotal:,}")
        print(f"  Categories: {len(self.centroids)}")

    # Core search 

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        exclude_ids: set = None,
    ) -> pd.DataFrame:
        """
        Search the index with a query vector.
        Returns top_k results as a DataFrame, excluding any product_ids in exclude_ids.
        """
        exclude_ids = exclude_ids or set()
        query = query_vector.astype("float32").reshape(1, -1)
        query = query / np.linalg.norm(query)

        fetch_k = top_k + len(exclude_ids) + 10
        scores, indices = self.index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            pid = self.product_ids[idx]
            if pid in exclude_ids:
                continue
            row = self.metadata[self.metadata["product_id"] == pid]
            if len(row) == 0:
                continue
            results.append({
                "product_id":  pid,
                "category_en": row.iloc[0]["category_en"],
                "product_text": row.iloc[0]["product_text"],
                "avg_price":   row.iloc[0]["avg_price"],
                "similarity":  float(score),
            })
            if len(results) == top_k:
                break

        return pd.DataFrame(results)

    # Convenience methods 

    def search_by_product(self, product_id: str, top_k: int = 10) -> pd.DataFrame:
        """Find products similar to a given product (by ID)."""
        if product_id not in self.pid_to_idx:
            return pd.DataFrame()
        idx = self.pid_to_idx[product_id]
        return self.search(self.embeddings[idx], top_k=top_k, exclude_ids={product_id})

    def search_by_category(self, category: str, top_k: int = 10) -> pd.DataFrame:
        """Cold-start: recommend from a category centroid."""
        if category not in self.centroids.index:
            return pd.DataFrame()
        centroid = self.centroids.loc[category].values.astype("float32")
        return self.search(centroid, top_k=top_k)

    def search_by_vector(self, vector: np.ndarray, top_k: int = 10, exclude_ids: set = None) -> pd.DataFrame:
        """Search with an arbitrary vector (e.g. a user embedding)."""
        return self.search(vector, top_k=top_k, exclude_ids=exclude_ids)

    def get_embedding(self, product_id: str) -> np.ndarray | None:
        """Return the embedding for a product_id, or None if not found."""
        if product_id not in self.pid_to_idx:
            return None
        return self.embeddings[self.pid_to_idx[product_id]]

    def get_category_centroid(self, category: str) -> np.ndarray | None:
        """Return the centroid vector for a category, or None if not found."""
        if category not in self.centroids.index:
            return None
        vec = self.centroids.loc[category].values.astype("float32")
        return vec / np.linalg.norm(vec)

    @property
    def categories(self) -> list[str]:
        return list(self.centroids.index)

    @property
    def n_products(self) -> int:
        return self.index.ntotal
