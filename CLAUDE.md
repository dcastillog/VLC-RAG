# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`rag-vlc` is an early-stage retrieval-augmented-generation pipeline over Visible Light Communication (VLC) research papers. As of the initial commit it is essentially a skeleton: `src/rag_vlc/__init__.py` only has a placeholder `main()`, and `smoke-test.py` verifies the embedding model loads. The real ingestion/retrieval code has not been written yet.

## Environment & commands

Uses [uv](https://docs.astral.sh/uv/) with the `uv_build` backend; Python >= 3.12.

```bash
uv sync                    # create/update .venv from uv.lock
uv run rag-vlc             # run the console entry point (rag_vlc:main)
uv run python smoke-test.py # sanity-check that fastembed downloads/loads the model (prints 384)
```

There is no test suite, linter, or CI configured yet. If adding tests, `pytest` under `uv run` is the natural choice.

## Architecture notes

- **Vector store**: Qdrant, run locally. `qdrant_storage/` in the repo root is a Qdrant server data directory (gitignored) — it implies Qdrant is expected to run as a local server (typically `docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant`), reachable at `localhost:6333`. `qdrant-client[fastembed]` also supports an embedded/in-memory mode; pick one deliberately and be consistent.
- **Embeddings**: `fastembed` with `BAAI/bge-small-en-v1.5` (384-dim vectors), as pinned in `smoke-test.py`. fastembed downloads the ONNX model on first use and caches it in the HuggingFace cache.
- **Planned data layout**: `.gitignore` reserves `data/pdfs/` (source PDFs) and `data/parsed/` (extracted/chunked text) — the intended flow is PDF -> parse/chunk -> embed -> upsert into Qdrant -> query. No PDF parsing dependency is in `pyproject.toml` yet; one will need to be added.
- **Config**: `.env` is gitignored but nothing reads it yet.
