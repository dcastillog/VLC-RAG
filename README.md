# rag-vlc

Retrieval-augmented-generation pipeline over Visible Light Communication (VLC)
research papers. This stage covers parsing, chunking, indexing, and retrieval
evaluation; generation is out of scope.

## Pipeline

```
PDF ──► data/normalized/{id}.txt + .json      (frozen canonical text + units)
     ──► data/chunks/{chunker}.jsonl          scripts/build_chunks.py
     ──► Qdrant vlc_fixed / vlc_section        scripts/ingest.py
     ──► retrieval + evaluation                scripts/query.py, ...
```

Run `uv run python scripts/verify_corpus.py` before and after any work: it is
the tripwire against normalized-text drift and eval-set inconsistency.

## Retrieval configuration decisions

Knobs live in `config/default.yaml` (`retrieval:`), not in code, so they can be
swept. The non-obvious ones:

- **`prefetch_limit: 100`** — each hybrid branch (dense, sparse) fetches this
  many candidates before fusion. Fusion only *reorders* the union of the two
  branches, so `prefetch_limit` — not the query `limit` — is the recall ceiling
  for every hybrid mode. It is set to 100 rather than Qdrant's more usual 50 as
  a deliberate choice: the Phase E judging pool is built **once** from these
  retrieval results, so this value also permanently bounds the ground truth
  that every later metric is measured against. A relevant chunk that no
  configuration surfaces in its top 100 is invisible to judging forever.
  Against ~3.3k vectors with exact search (see below) the wider prefetch costs
  nothing measurable.

- **`rrf_k: 2`** — Reciprocal Rank Fusion rank constant. This is Qdrant's
  default; the original RRF paper (Cormack et al. 2009) uses 60. Phase G
  sweeps `k ∈ {1, 2, 4, 10, 20, 60}`.

- **Dense HNSW is not built.** Both collections sit below Qdrant's
  `indexing_threshold`, so dense search is exact (brute force), not
  approximate. `collection_info.indexed_vectors_count` is still non-zero
  because the sparse/BM25 inverted index is always fully built — it does not
  indicate a dense HNSW index. Phase G measures what forcing HNSW would cost.

- **Query/document embedding asymmetry.** BGE (`bge-small-en-v1.5`) is trained
  with the instruction prefix `"Represent this sentence for searching relevant
  passages: "` on queries but not on documents. The prefix is applied in
  exactly one place (`Embedder.embed_query_dense`); ingestion embeds documents
  raw.
