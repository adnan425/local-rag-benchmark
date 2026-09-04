"""Dense (vector), sparse (BM25), and hybrid retrieval. Each exposes
retrieve(query, top_k) -> list[(doc_id, doc_text, score)]."""
from __future__ import annotations

from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .vectorstore import VectorStore


class DenseRetriever:
    def __init__(self, vectorstore: VectorStore, embedder):
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        q_emb = self.embedder.embed([query])[0]
        results = self.vectorstore.query(q_emb, top_k)
        # Chroma returns distance (lower = closer); convert to a similarity-like score.
        return [(doc_id, doc, 1.0 / (1.0 + dist)) for doc_id, doc, dist in results]


class BM25Retriever:
    def __init__(self, doc_ids: List[str], documents: List[str]):
        self.doc_ids = doc_ids
        self.documents = documents
        tokenized = [d.lower().split() for d in documents]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.doc_ids, self.documents, scores), key=lambda x: x[2], reverse=True)
        return ranked[:top_k]


class HybridRetriever:
    """Weighted fusion of dense + BM25 scores (min-max normalized per query)."""

    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever, alpha: float = 0.5, pool_k: int = 20):
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha  # weight on dense score; (1-alpha) on sparse
        self.pool_k = pool_k

    @staticmethod
    def _normalize(scored: List[Tuple[str, str, float]]) -> dict:
        if not scored:
            return {}
        vals = [s for _, _, s in scored]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        return {doc_id: (score - lo) / span for doc_id, _, score in scored}

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        dense_results = self.dense.retrieve(query, self.pool_k)
        sparse_results = self.sparse.retrieve(query, self.pool_k)

        dense_norm = self._normalize(dense_results)
        sparse_norm = self._normalize(sparse_results)

        text_by_id = {doc_id: doc for doc_id, doc, _ in dense_results}
        text_by_id.update({doc_id: doc for doc_id, doc, _ in sparse_results})

        all_ids = set(dense_norm) | set(sparse_norm)
        fused = [
            (doc_id, text_by_id[doc_id], self.alpha * dense_norm.get(doc_id, 0.0) + (1 - self.alpha) * sparse_norm.get(doc_id, 0.0))
            for doc_id in all_ids
        ]
        fused.sort(key=lambda x: x[2], reverse=True)
        return fused[:top_k]


def build_retriever(strategy: str, **kwargs):
    if strategy == "dense":
        return DenseRetriever(kwargs["vectorstore"], kwargs["embedder"])
    if strategy == "sparse":
        return BM25Retriever(kwargs["doc_ids"], kwargs["documents"])
    if strategy == "hybrid":
        dense = DenseRetriever(kwargs["vectorstore"], kwargs["embedder"])
        sparse = BM25Retriever(kwargs["doc_ids"], kwargs["documents"])
        return HybridRetriever(dense, sparse, alpha=kwargs.get("alpha", 0.5))
    raise ValueError(f"Unknown retrieval strategy: {strategy}")
