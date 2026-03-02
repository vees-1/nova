"""
NOVA — API Schemas
Pydantic models for all request and response bodies.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from src.config import DEFAULT_TOP_K, MAX_TOP_K


# Shared 

class ProductResult(BaseModel):
    product_id:   str
    category_en:  Optional[str]
    product_text: Optional[str]
    avg_price:    Optional[float]
    similarity:   float


# Requests 

class RecommendRequest(BaseModel):
    user_id:          str = Field(..., example="user_abc123")
    initial_category: Optional[str] = Field(None, example="sports_leisure")
    top_k:            int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class ProductSimilarityRequest(BaseModel):
    product_id: str = Field(..., example="abc123def456")
    top_k:      int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class CategoryRequest(BaseModel):
    category: str = Field(..., example="health_beauty")
    top_k:    int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class QueryRequest(BaseModel):
    query: str = Field(..., example="affordable running shoes")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class InteractionRequest(BaseModel):
    user_id:          str  = Field(..., example="user_abc123")
    product_id:       str  = Field(..., example="abc123def456")
    event_type:       str  = Field("view", example="purchase")
    initial_category: Optional[str] = Field(None, example="sports_leisure")


# Responses 

class RecommendResponse(BaseModel):
    user_id:         str
    cold_start:      bool
    interactions:    int
    recommendations: List[ProductResult]


class ProductSimilarityResponse(BaseModel):
    product_id:      str
    recommendations: List[ProductResult]


class CategoryResponse(BaseModel):
    category:        str
    recommendations: List[ProductResult]


class QueryResponse(BaseModel):
    query:           str
    recommendations: List[ProductResult]


class InteractionResponse(BaseModel):
    user_id:          str
    product_id:       str
    event_type:       str
    interaction_count: int
    is_cold:          bool


class HealthResponse(BaseModel):
    status:          str
    n_products:      int
    active_sessions: int
    available_categories: int
