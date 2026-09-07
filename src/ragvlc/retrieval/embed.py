"""Dense + sparse embedding, with the query/document asymmetry in one place.

* **Dense** -- ``BAAI/bge-small-en-v1.5`` via FastEmbed (CPU, ONNX). BGE was
  trained with an instruction prefix on queries but *not* on documents:
  documents go in raw, queries get :attr:`Embedder.query_prefix` prepended.
  FastEmbed's ``query_embed`` does **not** add this prefix for bge-small (it
  returns the same vector as ``embed``), so we prepend the string ourselves --
  and only here, so ingestion and query time cannot disagree about it.

* **Sparse** -- ``Qdrant/bm25`` via FastEmbed. Documents get term-frequency
  weights; queries get presence weights (``query_embed``). Inverse document
  frequency is applied by Qdrant at search time (the collection's sparse
  vector carries ``Modifier.IDF``), which is why the corpus statistics never
  need to be computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding, TextEmbedding


@dataclass(frozen=True)
class SparseVec:
    """A sparse embedding as Qdrant wants it: parallel index/value lists."""

    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: SparseVec


class Embedder:
    """Loads both FastEmbed models once and embeds documents or queries.

    Models load lazily on first use so importing this module (and constructing
    an ``Embedder``) stays cheap.
    """

    def __init__(self, dense_model: str, sparse_model: str, query_prefix: str) -> None:
        self._dense_model_name = dense_model
        self._sparse_model_name = sparse_model
        self._query_prefix = query_prefix
        self._dense: TextEmbedding | None = None
        self._sparse: SparseTextEmbedding | None = None

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def dense_dim(self) -> int:
        return self._dense_model().embedding_size

    def _dense_model(self) -> TextEmbedding:
        if self._dense is None:
            from fastembed import TextEmbedding

            self._dense = TextEmbedding(self._dense_model_name)
        return self._dense

    def _sparse_model(self) -> SparseTextEmbedding:
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            self._sparse = SparseTextEmbedding(self._sparse_model_name)
        return self._sparse

    # ------------------------------------------------------------------ #
    def embed_documents(self, texts: list[str], batch_size: int = 128) -> list[Embedding]:
        """Embed chunk texts for indexing -- no prefix, ever."""
        dense = self._dense_model().embed(texts, batch_size=batch_size)
        sparse = self._sparse_model().embed(texts, batch_size=batch_size)
        return [
            Embedding(
                dense=[float(x) for x in d],
                sparse=SparseVec(indices=[int(i) for i in s.indices], values=[float(v) for v in s.values]),
            )
            for d, s in zip(dense, sparse)
        ]

    def embed_query_dense(self, text: str) -> list[float]:
        """Dense query vector. This is the *only* place the BGE instruction
        prefix is added -- documents are embedded raw in ``embed_documents``."""
        vector = next(iter(self._dense_model().embed([self._query_prefix + text])))
        return [float(x) for x in vector]

    def embed_query_sparse(self, text: str) -> SparseVec:
        """Sparse (BM25) query vector -- presence weights; Qdrant applies IDF."""
        sparse = next(iter(self._sparse_model().query_embed([text])))
        return SparseVec(
            indices=[int(i) for i in sparse.indices],
            values=[float(v) for v in sparse.values],
        )

    def embed_query(self, text: str) -> Embedding:
        """Both query vectors. Phase D's search times the two halves
        separately and skips whichever a single-vector mode does not need, so
        it calls the two methods above rather than this one."""
        return Embedding(dense=self.embed_query_dense(text), sparse=self.embed_query_sparse(text))
