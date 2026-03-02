"""
NOVA — Central configuration
All paths and constants live here. Change once, applies everywhere.
"""

from pathlib import Path

# Paths 
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
EMBED_DIR  = DATA_DIR / "processed" / "embeddings"
INDEX_DIR  = DATA_DIR / "processed" / "index"
EVAL_DIR   = DATA_DIR / "processed" / "evaluation"

# Embedding model 
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384

# FAISS index 
INDEX_FILE        = INDEX_DIR / "nova_product.index"
PRODUCT_IDS_FILE  = EMBED_DIR / "product_ids.npy"
EMBEDDINGS_FILE   = EMBED_DIR / "product_embeddings.npy"
METADATA_FILE     = EMBED_DIR / "product_metadata.csv"
CENTROIDS_FILE    = EMBED_DIR / "category_centroids.csv"

# UserSession
DEFAULT_ALPHA = 0.3

INTERACTION_WEIGHTS = {
    "view":        0.1,
    "add_to_cart": 0.3,
    "purchase":    1.0,
}

# API
API_HOST      = "0.0.0.0"
API_PORT      = 8000
DEFAULT_TOP_K = 10
MAX_TOP_K     = 50
