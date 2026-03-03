#  N.O.V.A.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B.svg)](https://cqnvutni5v6hneidbuvwvc.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Neural Object-Vector Architecture

Cold-start recommendation system for ecommerce. Surfaces relevant products from the first visit — no purchase history needed.

---

## The Problem

96%+ of ecommerce customers have only one order. Traditional collaborative filtering fails completely for new users — you can't recommend based on behaviour that doesn't exist yet. New products face the same problem: they take weeks to accumulate enough interactions to surface organically.

NOVA solves this from first principles. Instead of relying on interaction history, it represents users and products as dense vectors in a shared semantic space and retrieves recommendations via fast nearest-neighbour search — working from the very first page view.

---

## Results

Evaluated using leave-one-out on 2,000 real customer purchase sequences, benchmarked against random and popularity baselines.

| Metric | Random | Popularity | NOVA |
|---|---|---|---|
| NDCG@10 | 0.0010 | 0.0047 | 0.0076 |
| Precision@10 | 0.0005 | 0.0020 | 0.0033 |
| Coverage@10 | ~100% | 4% | 79% |
| Median Latency | — | — | 0.67ms |
| p99 Latency | — | — | 1.96ms |

*62.5% improvement* in NDCG@10 over the popularity baseline. Coverage jumps from 4% to 79% — meaning NOVA actually surfaces the long tail of the catalog instead of recommending the same top products to everyone.

---

## Architecture

```
Product Catalog
      │
      ▼
 Text Representation
 "{category}, {price_tier}, {size_tier}, {photo_count}"
      │
      ▼
 SentenceTransformer (all-MiniLM-L6-v2)
      │
      ▼
 384-dim Embeddings  ──────────────────────────────────────┐
      │                                                    │
      ▼                                                    ▼
 FAISS IndexFlatIP                               Category Centroids
 (32,341 products)                               (cold-start anchors)
      │                                                    │
      └──────────────────────┬─────────────────────────────┘
                             ▼
                    Top-K Recommendations
                    (0.67ms median latency)
```

Online Learning — As users interact, their session vector updates via exponential moving average:

```
user_vec = (1 − α·w) × user_vec + (α·w) × product_vec
```

Where `w` is interaction weight: view=0.1, cart add=0.3, purchase=1.0. The system transitions from cold to warm without retraining.

---

## License

MIT — see [LICENSE](LICENSE) for details.
