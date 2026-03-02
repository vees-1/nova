"""
NOVA — UserSession
Manages a user's evolving embedding vector across a session.
Transitions from cold-start (category centroid) to warm-start (interaction-driven).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from src.config import DEFAULT_ALPHA, INTERACTION_WEIGHTS


@dataclass
class UserSession:
    """
    Stateful user embedding that updates incrementally with each interaction.

    Update rule:
        user_vector = (1 - α * weight) * user_vector + (α * weight) * product_vector

    where weight depends on interaction type (view < add_to_cart < purchase).
    """

    user_id: str
    initial_category: Optional[str] = None
    alpha: float = DEFAULT_ALPHA
    _vector: Optional[np.ndarray] = field(default=None, repr=False)
    _interaction_log: list = field(default_factory=list, repr=False)

    # Initialization 

    def initialize_from_category(self, centroid: np.ndarray):
        """Set the starting vector from a category centroid (cold-start)."""
        self._vector = centroid.copy().astype("float32")
        self._vector /= np.linalg.norm(self._vector)

    # Properties 

    @property
    def vector(self) -> Optional[np.ndarray]:
        return self._vector

    @property
    def is_cold(self) -> bool:
        return len(self._interaction_log) == 0

    @property
    def interaction_count(self) -> int:
        return len(self._interaction_log)

    @property
    def interacted_product_ids(self) -> set:
        return {log["product_id"] for log in self._interaction_log}

    # Update 

    def update(self, product_id: str, product_vector: np.ndarray, event_type: str = "view"):
        """
        Update user vector based on an interaction.

        Args:
            product_id:     ID of the product interacted with
            product_vector: The product's embedding vector
            event_type:     One of 'view', 'add_to_cart', 'purchase'
        """
        weight          = INTERACTION_WEIGHTS.get(event_type, 0.1)
        effective_alpha = self.alpha * weight
        product_vec     = product_vector.astype("float32")

        if self._vector is None:
            self._vector = product_vec.copy()
        else:
            self._vector = (1 - effective_alpha) * self._vector + effective_alpha * product_vec

        # Keep normalized
        norm = np.linalg.norm(self._vector)
        if norm > 0:
            self._vector /= norm

        self._interaction_log.append({
            "product_id": product_id,
            "event_type": event_type,
            "weight":     weight,
        })

    # Serialization 

    def to_dict(self) -> dict:
        return {
            "user_id":          self.user_id,
            "initial_category": self.initial_category,
            "alpha":            self.alpha,
            "is_cold":          self.is_cold,
            "interaction_count": self.interaction_count,
            "vector":           self._vector.tolist() if self._vector is not None else None,
            "interaction_log":  self._interaction_log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserSession":
        session = cls(
            user_id=data["user_id"],
            initial_category=data.get("initial_category"),
            alpha=data.get("alpha", DEFAULT_ALPHA),
        )
        if data.get("vector"):
            session._vector = np.array(data["vector"], dtype="float32")
        session._interaction_log = data.get("interaction_log", [])
        return session

    # Summary 

    def __repr__(self):
        status = "cold" if self.is_cold else f"warm ({self.interaction_count} interactions)"
        return f"UserSession(user_id={self.user_id!r}, status={status})"
