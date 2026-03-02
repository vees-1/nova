"""
NOVA — Recommender
The main entry point. Ties together the index, embedder, and session management.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.index    import ProductIndex
from src.session  import UserSession
from src.embedder import Embedder
from src.config   import DEFAULT_TOP_K


class Recommender:
    """
    High-level recommendation interface for NOVA.
    Manages a pool of active user sessions in memory.
    """

    def __init__(self):
        self.index   = ProductIndex()
        self.embedder = Embedder()
        self._sessions: dict[str, UserSession] = {}
        print("NOVA Recommender ready ✅")

    # Session management 

    def get_or_create_session(self, user_id: str, initial_category: Optional[str] = None) -> UserSession:
        """Return existing session or create a new one."""
        if user_id not in self._sessions:
            session = UserSession(user_id=user_id, initial_category=initial_category)
            if initial_category:
                centroid = self.index.get_category_centroid(initial_category)
                if centroid is not None:
                    session.initialize_from_category(centroid)
            self._sessions[user_id] = session
        return self._sessions[user_id]

    def delete_session(self, user_id: str):
        self._sessions.pop(user_id, None)

    # Recommendations 

    def recommend(
        self,
        user_id: str,
        initial_category: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict:
        """
        Get recommendations for a user.
        Creates a cold-start session if user is new.
        """
        session = self.get_or_create_session(user_id, initial_category)

        if session.vector is None:
            # Completely unknown user with no category — return popular-ish items
            results = self.index.search_by_category(
                self.index.categories[0], top_k=top_k
            )
        else:
            results = self.index.search_by_vector(
                session.vector,
                top_k=top_k,
                exclude_ids=session.interacted_product_ids,
            )

        return {
            "user_id":     user_id,
            "cold_start":  session.is_cold,
            "interactions": session.interaction_count,
            "recommendations": results.to_dict(orient="records"),
        }

    def recommend_by_product(self, product_id: str, top_k: int = DEFAULT_TOP_K) -> dict:
        """
        Stateless: recommend similar products to a given product.
        Used for 'you might also like' widgets.
        """
        results = self.index.search_by_product(product_id, top_k=top_k)
        return {
            "product_id":      product_id,
            "recommendations": results.to_dict(orient="records"),
        }

    def recommend_by_category(self, category: str, top_k: int = DEFAULT_TOP_K) -> dict:
        """
        Stateless: recommend top products for a category page.
        """
        results = self.index.search_by_category(category, top_k=top_k)
        return {
            "category":        category,
            "recommendations": results.to_dict(orient="records"),
        }

    def recommend_by_query(self, query_text: str, top_k: int = DEFAULT_TOP_K) -> dict:
        """
        Stateless: embed a free-text query and retrieve similar products.
        """
        query_vec = self.embedder.embed_one(query_text)
        results   = self.index.search_by_vector(query_vec, top_k=top_k)
        return {
            "query":           query_text,
            "recommendations": results.to_dict(orient="records"),
        }

    # User interaction update 

    def record_interaction(
        self,
        user_id: str,
        product_id: str,
        event_type: str = "view",
        initial_category: Optional[str] = None,
    ) -> dict:
        """
        Record a user interaction and update their embedding.
        """
        product_vector = self.index.get_embedding(product_id)
        if product_vector is None:
            return {"error": f"product_id '{product_id}' not found in index"}

        session = self.get_or_create_session(user_id, initial_category)
        session.update(product_id, product_vector, event_type=event_type)

        return {
            "user_id":          user_id,
            "product_id":       product_id,
            "event_type":       event_type,
            "interaction_count": session.interaction_count,
            "is_cold":          session.is_cold,
        }

    # Info 

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    @property
    def available_categories(self) -> list[str]:
        return self.index.categories

    @property
    def n_products(self) -> int:
        return self.index.n_products
