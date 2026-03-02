"""
NOVA — FastAPI Application
Serves cold-start and warm-start product recommendations over HTTP.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.recommender import Recommender
from api.schemas import (
    RecommendRequest,    RecommendResponse,
    ProductSimilarityRequest, ProductSimilarityResponse,
    CategoryRequest,     CategoryResponse,
    QueryRequest,        QueryResponse,
    InteractionRequest,  InteractionResponse,
    HealthResponse,
)

# App lifecycle 

recommender: Recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    print("Starting NOVA API...")
    recommender = Recommender()
    print("NOVA API ready ✅")
    yield
    print("Shutting down NOVA API...")


app = FastAPI(
    title="NOVA",
    description="Neural Object-Vector Architecture — Cold-Start Ecommerce Recommendations",
    version="0.1.0",
    lifespan=lifespan,
)


# Health 

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check API status and index stats."""
    return HealthResponse(
        status="ok",
        n_products=recommender.n_products,
        active_sessions=recommender.active_sessions,
        available_categories=len(recommender.available_categories),
    )


# Recommendations 

@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(req: RecommendRequest):
    """
    Get personalized recommendations for a user.
    
    - If the user is new, returns cold-start recommendations from their category.
    - If the user has interactions, returns warm recommendations from their evolving vector.
    """
    result = recommender.recommend(
        user_id=req.user_id,
        initial_category=req.initial_category,
        top_k=req.top_k,
    )
    return RecommendResponse(**result)


@app.post("/recommend/product", response_model=ProductSimilarityResponse, tags=["Recommendations"])
def recommend_by_product(req: ProductSimilarityRequest):
    """
    Find products similar to a given product.
    Useful for 'You might also like' widgets. Stateless — no user session needed.
    """
    result = recommender.recommend_by_product(req.product_id, top_k=req.top_k)
    if not result["recommendations"]:
        raise HTTPException(status_code=404, detail=f"Product '{req.product_id}' not found in index")
    return ProductSimilarityResponse(**result)


@app.post("/recommend/category", response_model=CategoryResponse, tags=["Recommendations"])
def recommend_by_category(req: CategoryRequest):
    """
    Get top products for a category page.
    Pure cold-start — no user history needed.
    """
    result = recommender.recommend_by_category(req.category, top_k=req.top_k)
    if not result["recommendations"]:
        raise HTTPException(status_code=404, detail=f"Category '{req.category}' not found")
    return CategoryResponse(**result)


@app.post("/recommend/query", response_model=QueryResponse, tags=["Recommendations"])
def recommend_by_query(req: QueryRequest):
    """
    Search for products using a free-text query.
    The query is embedded and matched against the product index.
    """
    result = recommender.recommend_by_query(req.query, top_k=req.top_k)
    return QueryResponse(**result)


# User interactions 

@app.post("/interact", response_model=InteractionResponse, tags=["Users"])
def record_interaction(req: InteractionRequest):
    """
    Record a user interaction with a product.
    Updates the user's embedding vector in real time.
    
    event_type: 'view' | 'add_to_cart' | 'purchase'
    """
    valid_events = {"view", "add_to_cart", "purchase"}
    if req.event_type not in valid_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type '{req.event_type}'. Must be one of: {valid_events}"
        )

    result = recommender.record_interaction(
        user_id=req.user_id,
        product_id=req.product_id,
        event_type=req.event_type,
        initial_category=req.initial_category,
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return InteractionResponse(**result)


# Info 

@app.get("/categories", tags=["System"])
def list_categories():
    """List all available product categories."""
    return {"categories": recommender.available_categories}


# Entry point

if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
