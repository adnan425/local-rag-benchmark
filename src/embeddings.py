"""Embedding backends: Ollama-served models, or sentence-transformers as a
CPU-only fallback."""
from __future__ import annotations

from pathlib import Path
from typing import List

try:
    import ollama
except ImportError:
    ollama = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class CachedEmbedder:
    """Caches vectors by text so repeated queries aren't re-embedded."""

    def __init__(self, inner):
        self.inner = inner
        self.model_name = getattr(inner, "model_name", None)
        self._cache: dict = {}

    def embed(self, texts: List[str]) -> List[List[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self.inner.embed(missing)):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]

    @property
    def name(self) -> str:
        return self.inner.name


class OllamaEmbedder:
    def __init__(self, model_name: str = "nomic-embed-text"):
        if ollama is None:
            raise RuntimeError("ollama package not installed: pip install ollama")
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        result = ollama.embed(model=self.model_name, input=texts, keep_alive="5m")
        embeddings = getattr(result, "embeddings", None)
        if embeddings is None:
            embeddings = result["embeddings"]
        return list(embeddings)

    @property
    def name(self) -> str:
        return f"ollama:{self.model_name}"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed: pip install sentence-transformers")
        local_dir = Path("models") / model_name
        load_path = str(local_dir) if local_dir.exists() else model_name
        self._model = SentenceTransformer(load_path)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"


def build_embedder(model_name: str, backend: str = "ollama"):
    if backend == "ollama":
        return CachedEmbedder(OllamaEmbedder(model_name))
    if backend == "sentence-transformers":
        return CachedEmbedder(SentenceTransformerEmbedder(model_name))
    raise ValueError(f"Unknown embedding backend: {backend}")
