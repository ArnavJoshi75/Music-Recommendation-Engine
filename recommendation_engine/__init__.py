"""
recommendation_engine
=====================
Content-based music recommendation pipeline using CRNN latent embeddings
and cosine-similarity nearest-neighbor search.

Modules:
    feature_extractor  — Model surgery & audio preprocessing
    build_vector_db    — Batch embedding extraction over FMA-Small
    recommend          — KNN-based recommendation CLI
"""
