# NOVA 🌌

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Traditional recommendation systems break down at the most critical moment: when you have a new user or a new product with no history. This is the cold-start problem, and it's especially painful in ecommerce where:

- New users see generic, irrelevant recommendations and immediately bounce
- New products take weeks to accumulate enough interactions to surface organically
- Small stores with sparse data can't compete with large-catalog platforms

NOVA is being built to solve this from first principles.

Instead of relying on historical interaction data (clicks, purchases, ratings), NOVA will:

- Represent both users and products as dense vectors in a shared semantic embedding space
- Use semantic similarity to identify relevant items based on content and context
- Retrieve recommendations via fast approximate nearest-neighbor search, enabling real-time inference at scale

```
User Profile → Embedding Model → User Vector ──┐
                                                 ├──▶ FAISS Index ──▶ Top-K Products
Product Catalog → Embedding Model → Product Vectors ──┘
```

---

## Status

This project is currently under active development as part of a 6-month ML engineering internship. Below is the current progress:

| Module | Status |
|---|---|
| Dataset loading & EDA | 🔄 In Progress |
| Product embedding generation | ⏳ Planned |
| Dual-encoder model | ⏳ Planned |
| FAISS index & retrieval | ⏳ Planned |
| Online embedding updates | ⏳ Planned |
| Evaluation framework | ⏳ Planned |
| FastAPI serving layer | ⏳ Planned |
| Demo dashboard | ⏳ Planned |

---

## Project Context

NOVA is being developed during a ML engineering internship at Edreamz Technologies | Shopify Experts Partner Agency and Web Solutions Providers. The motivation is a recurring client pain point: small Shopify stores struggle with product discovery for new users, leading to high bounce rates on landing and collection pages. NOVA is designed to be lightweight enough to run on modest infrastructure while still providing meaningful recommendations from the very first session.

---

## Getting Started

*Setup instructions will be added as the project structure is finalized.*

---

## Roadmap

- [ ] Exploratory data analysis on Olist dataset
- [ ] Product embedding generation with Sentence-Transformers
- [ ] FAISS-based ANN index and retrieval
- [ ] Dual-encoder model training
- [ ] Online embedding updates (cold → warm transition)
- [ ] Offline evaluation framework (Precision@K, NDCG, Coverage)
- [ ] Baseline comparisons (random, popularity, TF-IDF)
- [ ] FastAPI serving layer
- [ ] Streamlit demo dashboard
- [ ] Multi-modal embeddings (product images + text)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with curiosity and way too much coffee ☕</p>



