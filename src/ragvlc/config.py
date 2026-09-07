"""Central configuration for the RAG-VLC pipeline.

Two kinds of configuration live here, kept deliberately separate:

* **Infrastructure** -- URLs and contact emails -- comes from the process
  environment or a ``.env`` file and is modelled by :class:`Settings`.
* **Experiment knobs** -- values that change how a run behaves and that we may
  want to sweep later -- come from ``config/default.yaml`` and are modelled by
  :class:`ExperimentConfig`.

Configuration is validated lazily on first use via the cached accessors
:func:`get_settings`, :func:`get_experiment` and :func:`get_paths`. Importing
this module (or any module that merely imports a pure helper alongside it) does
*not* require a populated ``.env``; the first accessor call does, and fails
immediately with a clear error rather than halfway through a 36-PDF run. Tests
can populate the caches directly to inject config without touching the
environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root(start: Path) -> Path:
    """Walk upwards from ``start`` until a directory containing ``pyproject.toml``.

    Used instead of a hard-coded relative path so the config resolves correctly
    regardless of the current working directory (``uv run`` from a subdirectory,
    pytest, an editor, ...).
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        f"could not locate project root (no pyproject.toml above {start})"
    )


PROJECT_ROOT: Path = _find_project_root(Path(__file__).resolve())


# --------------------------------------------------------------------------- #
# Infrastructure settings (environment / .env)
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Infrastructure values read from the environment or ``.env``.

    Field names map to upper-case environment variables case-insensitively
    (``grobid_url`` <- ``GROBID_URL``).
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    grobid_url: str = Field(
        default="http://localhost:8070",
        description="Base URL of the GROBID service.",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Base URL of the Qdrant server (local Docker, per CLAUDE.md).",
    )
    crossref_mailto: str = Field(
        description="Contact email sent as ?mailto= to the Crossref API.",
    )
    unpaywall_email: str = Field(
        description="Contact email required as ?email= by the Unpaywall API.",
    )

    @field_validator("crossref_mailto", "unpaywall_email")
    @classmethod
    def _looks_like_email(cls, value: str) -> str:
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError(f"does not look like an email address: {value!r}")
        return value

    @field_validator("grobid_url", "qdrant_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


# --------------------------------------------------------------------------- #
# Experiment configuration (config/default.yaml)
# --------------------------------------------------------------------------- #
class _StrictModel(BaseModel):
    """Base for YAML-backed models: reject unknown keys so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class GrobidConfig(_StrictModel):
    """Knobs for the GROBID full-text request."""

    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0  # linear backoff multiplier between retries
    consolidate_header: int = 1
    consolidate_citations: int = 0
    include_raw_affiliations: int = 0
    segment_sentences: int = 0


class CrossrefConfig(_StrictModel):
    """Politeness/timeout settings for the Crossref / Unpaywall calls.

    Retries reuse ``grobid.max_retries`` / ``grobid.retry_backoff_seconds``
    (the "same retry policy" PROMPT_1 specifies) rather than duplicating them.
    """

    sleep_seconds: float = 0.2
    timeout_seconds: float = 15.0


class QualityGates(_StrictModel):
    """Thresholds ``parse_corpus.py`` checks each paper against.

    A handful of other flags it prints (any PUA character, any U+FFFD,
    missing DOI, non-CC licence) are plain presence checks with no
    meaningful threshold to tune, so they aren't modeled here.
    """

    min_sections: int = 4
    min_chars: int = 5_000
    min_mean_unit_chars: float = 100.0
    min_alpha_ratio: float = 0.7
    max_non_ascii_ratio: float = 0.02


class ParsingConfig(_StrictModel):
    """Everything that governs stage 1 (PDF -> canonical text)."""

    grobid: GrobidConfig = Field(default_factory=GrobidConfig)
    crossref: CrossrefConfig = Field(default_factory=CrossrefConfig)
    quality_gates: QualityGates = Field(default_factory=QualityGates)

    min_unit_chars: int = 30
    unit_separator: str = "\n\n"


class FixedChunkingConfig(_StrictModel):
    """Knobs specific to the structure-blind ``fixed`` chunker."""

    overlap_tokens: int = 60  # tokens shared between consecutive chunks


class SectionAwareChunkingConfig(_StrictModel):
    """Knobs specific to the ``section_aware`` chunker."""

    # A unit smaller than this is merged into an adjacent unit -- but only when
    # both units carry the same section_heading (see the chunker for why).
    min_chunk_tokens: int = 50

    # After chunking, a chunk with fewer than this many tokens is dropped
    # rather than indexed: a handful-of-tokens caption fragment carries almost
    # no retrievable signal but still consumes a top-k slot and pool-judging
    # effort. Deliberately applied to section_aware only -- fixed's short
    # chunks are one-per-paper tails of real content, and keeping the asymmetry
    # keeps the comparison honest about what each strategy actually indexes.
    min_indexed_tokens: int = 15


class ChunkingConfig(_StrictModel):
    """Everything that governs stage 2's chunking (normalized text -> chunks).

    ``max_tokens`` is a single budget both chunkers obey. It is deliberately
    below the embedding model's hard 512-token limit: the model truncates
    silently past 512, so a chunk measured at exactly the limit could still
    lose its tail once a query-time prefix or the [CLS]/[SEP] pair is added.
    """

    tokenizer_model: str = "BAAI/bge-small-en-v1.5"
    max_tokens: int = 400
    fixed: FixedChunkingConfig = Field(default_factory=FixedChunkingConfig)
    section_aware: SectionAwareChunkingConfig = Field(default_factory=SectionAwareChunkingConfig)


class QdrantIndexConfig(_StrictModel):
    """Where each chunker's chunks land in Qdrant, and how they get there."""

    # chunker name -> collection name. The CLI alias "section" maps to
    # "section_aware" (see scripts/ingest.py); collections stay short.
    collections: dict[str, str] = Field(
        default_factory=lambda: {"fixed": "vlc_fixed", "section_aware": "vlc_section"}
    )
    upsert_batch_size: int = 128
    # Lowering this in a Phase G experiment forces Qdrant to build an HNSW
    # index even for this small corpus; the default (20 MB) leaves the
    # collections doing exact search. Kept here so that experiment is a config
    # change, not a code change.
    indexing_threshold_kb: int | None = None  # None -> Qdrant's default


class RetrievalConfig(_StrictModel):
    """Embedding models and the query/document asymmetry.

    ``query_prefix`` lives here and nowhere else: BGE models are trained with
    this instruction prepended to queries but not to documents, and if the two
    sides ever disagreed on it, recall would drop with nothing in the code to
    point at. Ingestion embeds documents raw; only query embedding prepends it.
    """

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Hybrid retrieval: each prefetch branch (dense, sparse) fetches this many
    # candidates before fusion. Fusion only *reorders* their union, so this
    # number is the recall ceiling for every hybrid mode -- a gold chunk that
    # neither branch returns in its top prefetch_limit can never be retrieved.
    # Set to 100 (not Qdrant-ish 50): the Phase E pool is built once from these
    # results and permanently bounds the judged ground truth; exact search over
    # ~3.3k vectors makes the wider prefetch essentially free.
    prefetch_limit: int = 100
    # RRF rank constant. Qdrant defaults to 2; the original RRF paper
    # (Cormack et al. 2009) uses 60. Larger k flattens the contribution
    # gap between ranks. Phase G sweeps this.
    rrf_k: int = 2

    qdrant: QdrantIndexConfig = Field(default_factory=QdrantIndexConfig)


class ExperimentConfig(_StrictModel):
    """Root of the YAML config."""

    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @model_validator(mode="after")
    def _dense_model_matches_tokenizer(self) -> ExperimentConfig:
        # Chunk token budgets are counted with chunking.tokenizer_model; the
        # embedder truncates with retrieval.dense_model's tokenizer. If these
        # are different models the budgets stop meaning anything.
        if self.chunking.tokenizer_model != self.retrieval.dense_model:
            raise ValueError(
                "chunking.tokenizer_model and retrieval.dense_model must be the same model "
                f"({self.chunking.tokenizer_model!r} != {self.retrieval.dense_model!r})"
            )
        return self


DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "default.yaml"


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    """Load and validate the experiment config from a YAML file.

    Raises ``FileNotFoundError`` if the file is missing and
    ``pydantic.ValidationError`` (naming the offending key) if it is malformed.
    """
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig.model_validate(raw)


# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
class Paths(BaseModel):
    """Absolute paths for pipeline inputs and outputs, derived from the root.

    These are computed, not configured: the directory layout is part of the
    project contract, not something to tune per run.
    """

    root: Path
    pdfs: Path
    tei: Path
    normalized: Path
    crossref: Path
    config_dir: Path
    manifest_csv: Path
    doi_overrides_csv: Path
    eval_dir: Path
    questions_jsonl: Path

    @classmethod
    def from_root(cls, root: Path) -> Paths:
        data = root / "data"
        return cls(
            root=root,
            pdfs=data / "pdfs",
            tei=data / "tei",
            normalized=data / "normalized",
            crossref=data / "crossref",
            config_dir=root / "config",
            manifest_csv=data / "manifest.csv",
            doi_overrides_csv=data / "doi_overrides.csv",
            eval_dir=data / "eval",
            # The frozen evaluation set. "v2" is the working version referenced by
            # PROMPT_2; the earlier data/eval/questions.jsonl is kept only for history.
            questions_jsonl=data / "eval" / "questions.v2.jsonl",
        )

    def mkdirs(self) -> None:
        """Create the writable pipeline directories if they do not exist."""
        for directory in (self.pdfs, self.tei, self.normalized, self.crossref):
            directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Cached accessors -- the first call is the validation step.
# --------------------------------------------------------------------------- #
# @lru_cache makes each of these a lazily-built singleton: validation happens on
# first call (still fail-fast) but not at import time. Tests can bypass the
# environment entirely, e.g.
#     get_settings.cache_clear()
#     config.get_settings = lambda: Settings(crossref_mailto=..., unpaywall_email=...)
# or by priming the cache via the real constructor.


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the infrastructure settings, read once from the environment / ``.env``."""
    return Settings()  # type: ignore[call-arg]  # values come from env / .env


@lru_cache(maxsize=1)
def get_experiment() -> ExperimentConfig:
    """Return the experiment config, loaded once from ``config/default.yaml``."""
    return load_experiment_config()


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Return the pipeline filesystem layout, derived from the project root."""
    return Paths.from_root(PROJECT_ROOT)
