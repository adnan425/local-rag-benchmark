"""Persistent local vector store backed by Chroma."""
from __future__ import annotations

from typing import List, Sequence

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str, reset: bool = True):
        self._client = chromadb.PersistentClient(path=persist_dir)
        if reset:
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass
            self._collection = self._client.create_collection(collection_name)
        else:
            self._collection = self._client.get_or_create_collection(collection_name)

    def count(self) -> int:
        return self._collection.count()

    def add(self, ids: Sequence[str], embeddings: Sequence[List[float]], documents: Sequence[str]):
        self._collection.add(ids=list(ids), embeddings=list(embeddings), documents=list(documents))

    def query(self, embedding: List[float], top_k: int):
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = result["ids"][0]
        docs = result["documents"][0]
        distances = result["distances"][0]
        return list(zip(ids, docs, distances))
