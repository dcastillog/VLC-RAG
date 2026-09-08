"""IDF-weighted lexical overlap between a question and its gold spans.

This is the property the Phase F regression plots against retrieval-mode
performance: for each question, how much of its rare vocabulary is literally
shared with the text that answers it? Computed **before any retrieval, from
the question/gold-span pair alone** -- the same value regardless of which
chunker or mode is being scored, which is what makes "x = overlap score, y =
sparse RR - dense RR" a fair plot to draw a line through.

Two pieces, split by dependency:

* :func:`compute_idf` / :func:`overlap_score` are pure -- given token-id sets,
  no model involved. This is what is unit tested without a model download.
* :class:`Bm25Tokenizer` is the one part that calls the model: it turns text
  into a token-id set using the *same* ``Qdrant/bm25`` tokenizer (stemming,
  stopword removal, hashing) the sparse retrieval index uses, so "the same
  token" here means the same thing it means at retrieval time.

IDF is computed over one chunker's chunk set as the reference corpus
(``eval.idf_corpus_chunker`` -- arbitrary but fixed, see the config comment);
each chunk counts as one "document" for document-frequency purposes, matching
what Qdrant's server-side IDF sees.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding


@dataclass(frozen=True)
class IdfIndex:
    idf: dict[int, float]
    n_docs: int

    def get(self, token_id: int) -> float:
        return self.idf.get(token_id, self.unseen_idf)

    @property
    def unseen_idf(self) -> float:
        """IDF assigned to a token that appears in none of the corpus
        documents -- the df=0 case of the same formula, i.e. maximally rare."""
        return math.log(1 + (self.n_docs + 0.5) / 0.5)


def compute_idf(doc_token_sets: list[set[int]]) -> IdfIndex:
    """Robertson-Sparck-Jones IDF, per token id, over a corpus of documents
    each represented as a token-id set (presence, not frequency, per doc --
    document frequency is what IDF means)."""
    n_docs = len(doc_token_sets)
    df: Counter[int] = Counter()
    for tokens in doc_token_sets:
        df.update(tokens)
    idf = {token: math.log(1 + (n_docs - count + 0.5) / (count + 0.5)) for token, count in df.items()}
    return IdfIndex(idf=idf, n_docs=n_docs)


def overlap_score(idf_index: IdfIndex, question_tokens: set[int], answer_tokens: set[int]) -> float:
    """Sum of IDF over tokens in both sets, divided by the sum of IDF over
    every question token. Range 0..1; 0.0 for an empty question."""
    if not question_tokens:
        return 0.0
    shared = question_tokens & answer_tokens
    denominator = sum(idf_index.get(t) for t in question_tokens)
    if denominator == 0:
        return 0.0
    return sum(idf_index.get(t) for t in shared) / denominator


class Bm25Tokenizer:
    """Text -> token-id set, via the same tokenizer used for sparse retrieval
    (stemming, stopword removal, then hashed to an id)."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self._model_name = model_name
        self._model: SparseTextEmbedding | None = None

    def _load(self) -> SparseTextEmbedding:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(self._model_name)
        return self._model

    def token_sets(self, texts: list[str]) -> list[set[int]]:
        """One token-id set per text, in order. Empty text -> empty set."""
        non_empty = [t for t in texts if t.strip()]
        if not non_empty:
            return [set() for _ in texts]
        embedded = iter(self._load().embed(non_empty))
        return [set(next(embedded).indices.tolist()) if t.strip() else set() for t in texts]
