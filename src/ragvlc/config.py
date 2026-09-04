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
from pydantic import BaseModel, ConfigDict, Field, field_validator
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

    @field_validator("grobid_url")
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
    """Thresholds below which ``parse_corpus.py`` flags a paper as suspicious."""

    min_sections: int = 4
    min_chars: int = 5_000
    min_mean_unit_chars: float = 100.0
    min_alpha_ratio: float = 0.7


class ParsingConfig(_StrictModel):
    """Everything that governs stage 1 (PDF -> canonical text)."""

    grobid: GrobidConfig = Field(default_factory=GrobidConfig)
    crossref: CrossrefConfig = Field(default_factory=CrossrefConfig)
    quality_gates: QualityGates = Field(default_factory=QualityGates)

    min_unit_chars: int = 30
    unit_separator: str = "\n\n"


class ExperimentConfig(_StrictModel):
    """Root of the YAML config. Only parsing keys exist at this stage."""

    parsing: ParsingConfig = Field(default_factory=ParsingConfig)


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
