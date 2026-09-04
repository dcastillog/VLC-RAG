"""Resolve a paper's DOI and fetch authoritative metadata from Crossref and
Unpaywall, to make up for GROBID's unreliable header parsing (see `tei.py`).

Unlike `tei.py`, the response models here are `pydantic.BaseModel`s: this data
crosses a trust boundary (a third-party API we don't control), so it gets
validated on the way in rather than trusted the way our own extracted TEI
structure is.

DOI resolution order (never guessed, never constructed): GROBID's header ->
`data/doi_overrides.csv` -> give up (`doi=None`). See `resolve_doi`.

Caching mirrors `grobid.py`: `fetch_metadata`'s cache check happens *before*
any request, and stores the two APIs' **raw** JSON responses (not the derived
fields) to `data/crossref/{paper_id}.json`, so a later improvement to the
extraction rules below benefits already-cached papers without re-fetching --
and so a permanently-404 DOI doesn't get re-queried on every run either.
`force=True` bypasses the cache and re-fetches.

Every degraded outcome (no DOI, Crossref 404, Unpaywall failure, no usable
licence, a licence that isn't a CC variant, an unversioned licence) is
recorded in `PaperMetadata.warnings` and only there -- this module returns
data, it doesn't print. `parse_corpus.py` (a later stage) owns terminal
output and decides how to present these, including PROMPT_1's "warning
listing every such paper at the end of the run".
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import NamedTuple

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ragvlc.config import get_experiment, get_paths, get_settings


class CrossrefError(RuntimeError):
    """Raised when Crossref or Unpaywall is unreachable or erroring in a way
    that isn't a per-DOI 404 (which degrades gracefully instead, see
    `_fetch_crossref_raw`). Always names the DOI.
    """


# --------------------------------------------------------------------------- #
# Response models (external, untrusted data) -- deliberately narrow: only the
# fields PROMPT_1 asks for are modeled, everything else in these APIs' large,
# evolving schemas is ignored rather than rejected.
# --------------------------------------------------------------------------- #
class _CrossrefAuthor(BaseModel):
    model_config = ConfigDict(extra="ignore")

    given: str | None = None
    family: str | None = None
    name: str | None = None  # organisational authors have only this


class _CrossrefLicense(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    url: str = Field(alias="URL")
    content_version: str = Field(alias="content-version", default="")


class _CrossrefDateParts(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    date_parts: list[list[int | None]] = Field(alias="date-parts", default_factory=list)

    @property
    def year(self) -> int | None:
        if self.date_parts and self.date_parts[0]:
            return self.date_parts[0][0]
        return None


class CrossrefWork(BaseModel):
    """The `message` object from `GET /works/{doi}`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    title: list[str] = Field(default_factory=list)
    author: list[_CrossrefAuthor] = Field(default_factory=list)
    issued: _CrossrefDateParts | None = None
    published_print: _CrossrefDateParts | None = Field(alias="published-print", default=None)
    published_online: _CrossrefDateParts | None = Field(alias="published-online", default=None)
    container_title: list[str] = Field(alias="container-title", default_factory=list)
    license: list[_CrossrefLicense] = Field(default_factory=list)


class _UnpaywallLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    license: str | None = None


class UnpaywallResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_oa: bool | None = None
    oa_status: str | None = None
    best_oa_location: _UnpaywallLocation | None = None


class _CachedRaw(BaseModel):
    """The on-disk shape of `data/crossref/{paper_id}.json`: the raw API
    payloads plus enough status information to reconstruct what happened
    without re-parsing HTTP semantics. `crossref`/`unpaywall` are left as
    untyped dicts here deliberately -- this is the *raw* response, kept in
    full even though `CrossrefWork`/`UnpaywallResponse` above only look at a
    slice of it.
    """

    doi: str
    crossref: dict | None = None
    crossref_status: str = "ok"  # "ok" | "not_found"
    unpaywall: dict | None = None
    unpaywall_status: str = "ok"  # "ok" | "error"
    unpaywall_error: str | None = None


class PaperMetadata(BaseModel):
    """Normalized, derived metadata -- what `pipeline.py`/`manifest.csv`
    actually consume. Re-derived fresh from `_CachedRaw` every time (see
    `fetch_metadata`), whether that came from the network or the cache.
    """

    doi: str | None
    title: str | None
    authors: list[str] = Field(default_factory=list)
    year: int | None
    venue: str | None
    licence: str  # "UNKNOWN" if nothing usable was found
    is_oa: bool | None = None
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# DOI resolution -- never guessed, never constructed
# --------------------------------------------------------------------------- #
def load_doi_overrides(path: Path | None = None) -> dict[str, str]:
    """Read `data/doi_overrides.csv` (columns: `paper_id,doi`) into a dict.

    Missing file -> empty dict: the override file is optional, most papers
    won't need one.
    """
    csv_path = path or get_paths().doi_overrides_csv
    if not csv_path.is_file():
        return {}
    overrides: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            paper_id = (row.get("paper_id") or "").strip()
            doi = (row.get("doi") or "").strip()
            if paper_id and doi:
                overrides[paper_id] = doi
    return overrides


def resolve_doi(paper_id: str, grobid_doi: str | None, overrides: dict[str, str] | None = None) -> str | None:
    """DOI resolution order: GROBID header -> `data/doi_overrides.csv` -> None.

    Never guesses or constructs a DOI -- a paper with neither source gets
    `None` here, which `fetch_metadata` turns into `doi=None, licence="UNKNOWN"`.
    `overrides` can be pre-loaded (e.g. once per batch run) to avoid re-reading
    the CSV for every paper; defaults to loading it fresh.
    """
    if grobid_doi and grobid_doi.strip():
        return grobid_doi.strip()
    overrides = load_doi_overrides() if overrides is None else overrides
    return overrides.get(paper_id)


# --------------------------------------------------------------------------- #
# HTTP: same retry policy as grobid.py (retry timeout/connection-error/5xx,
# linear backoff, fail naming the DOI) -- but status-code interpretation is
# left to each caller, because a 404 means something different to a Crossref
# caller (the DOI doesn't exist) than it would to an Unpaywall caller.
# --------------------------------------------------------------------------- #
def _get_with_retry(url: str, *, doi: str, service: str, timeout: float, max_retries: int, backoff_seconds: float) -> httpx.Response:
    max_attempts = 1 + max_retries
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.get(url, timeout=timeout)
            if response.status_code >= 500:
                last_error = CrossrefError(f"{service} returned {response.status_code} for DOI {doi}")
            else:
                return response
        except httpx.HTTPError as exc:
            last_error = exc  # timeout, connection error, ... -- retry

        if attempt < max_attempts:
            time.sleep(backoff_seconds * attempt)

    raise CrossrefError(f"{service} failed for DOI {doi} after {max_attempts} attempt(s): {last_error}")


def _fetch_crossref_raw(doi: str, mailto: str, *, timeout: float, max_retries: int, backoff_seconds: float) -> tuple[dict | None, str]:
    """Return (raw `message` dict or None, status). status is "not_found" for
    a 404 (the DOI genuinely doesn't exist in Crossref -- not retried, and
    reported back rather than raised, so one bad DOI doesn't abort a batch).
    Anything else unrecoverable (5xx exhausted, connection error exhausted, an
    unexpected non-404 4xx) raises `CrossrefError`.
    """
    url = f"https://api.crossref.org/works/{doi}?mailto={mailto}"
    response = _get_with_retry(
        url, doi=doi, service="Crossref", timeout=timeout, max_retries=max_retries, backoff_seconds=backoff_seconds
    )
    if response.status_code == 404:
        return None, "not_found"
    if response.status_code >= 400:
        raise CrossrefError(f"Crossref returned {response.status_code} for DOI {doi}: {response.text[:300]!r}")
    return response.json()["message"], "ok"


def _fetch_unpaywall_raw(
    doi: str, email: str, *, timeout: float, max_retries: int, backoff_seconds: float
) -> tuple[dict | None, str, str | None]:
    """Return (raw dict or None, status, error_message). Never raises: any
    failure here (network, timeout, non-2xx) degrades to (None, "error", msg)
    so Unpaywall going down never takes a Crossref-derived record down with it.
    """
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        response = _get_with_retry(
            url, doi=doi, service="Unpaywall", timeout=timeout, max_retries=max_retries, backoff_seconds=backoff_seconds
        )
    except CrossrefError as exc:
        return None, "error", str(exc)
    if response.status_code >= 400:
        return None, "error", f"HTTP {response.status_code}"
    return response.json(), "ok", None


# --------------------------------------------------------------------------- #
# Field extraction -- all messier than they look, per PROMPT_1
# --------------------------------------------------------------------------- #
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Crossref titles routinely carry inline markup (`<i>`, `<sub>`, `<scp>`,
    ...) for italics/subscripts/small-caps -- strip the tags, keep the text.
    """
    return _HTML_TAG_RE.sub("", text)


def _extract_title(work: CrossrefWork) -> str | None:
    if not work.title:
        return None
    return _strip_html(work.title[0]).strip() or None


def _extract_authors(work: CrossrefWork) -> list[str]:
    names: list[str] = []
    for author in work.author:
        if author.given or author.family:
            name = " ".join(part for part in (author.given, author.family) if part)
        elif author.name:
            name = author.name
        else:
            continue  # an author entry with none of these is unusable
        name = name.strip()
        if name:
            names.append(name)
    return names


def _extract_year(work: CrossrefWork) -> int | None:
    # issued -> published-print -> published-online, first one with a year wins.
    for date_parts in (work.issued, work.published_print, work.published_online):
        if date_parts is not None and date_parts.year is not None:
            return date_parts.year
    return None


def _extract_venue(work: CrossrefWork) -> str | None:
    if not work.container_title:
        return None
    return work.container_title[0].strip() or None


# Creative Commons URL, e.g. "https://creativecommons.org/licenses/by-nc/4.0/"
# or the public-domain dedication "https://creativecommons.org/publicdomain/zero/1.0/".
_CC_URL_RE = re.compile(
    r"creativecommons\.org/(?:licenses/(?P<variant>[a-z-]+)|publicdomain/(?P<zero>zero))/(?P<version>\d+\.\d+)?",
    re.IGNORECASE,
)
# Unpaywall's bare-string form, e.g. "cc-by", "cc-by-nc-sa", "cc0". Never
# carries a version -- Unpaywall just doesn't give one.
_CC_BARE_RE = re.compile(r"^cc[-_]?(?P<variant>by[a-z-]*|0)$", re.IGNORECASE)


class _CCLicence(NamedTuple):
    variant: str  # e.g. "BY", "BY-NC-SA", or "0" for the public-domain dedication
    version: str | None  # None when the source doesn't carry one


def _parse_cc_licence(raw: str) -> _CCLicence | None:
    """Parse a CC licence URL or Unpaywall's bare string into (variant,
    version). Returns None if `raw` doesn't look like a CC reference at all --
    the caller then treats it as an unrecognized, non-CC licence.
    """
    text = raw.strip()

    m = _CC_URL_RE.search(text)
    if m:
        variant = "0" if m.group("zero") else m.group("variant").upper()
        return _CCLicence(variant, m.group("version"))

    m = _CC_BARE_RE.match(text)
    if m:
        variant = m.group("variant")
        return _CCLicence("0" if variant == "0" else variant.upper(), None)

    return None


def _format_cc_licence(licence: _CCLicence) -> str:
    prefix = "CC0" if licence.variant == "0" else f"CC-{licence.variant}"
    return f"{prefix}-{licence.version}" if licence.version else prefix


def _derive_licence(unpaywall: UnpaywallResponse | None, work: CrossrefWork | None) -> tuple[str, str | None]:
    """Prefer Unpaywall's `best_oa_location.license` over Crossref's `license`
    array; if falling back to Crossref, prefer the entry with
    `content-version: "vor"` (version of record) over e.g. an accepted
    manuscript, which can carry different terms.

    Unpaywall's value is a bare string ("cc-by") with no version. Rather than
    assume one -- fabricating the field that carries this project's licensing
    claim would be worse than admitting it's unknown -- a missing version is
    first recovered from Crossref's licence URL, *only* when Crossref agrees
    on the variant (never borrow a different variant's version). If neither
    source gives a version, the licence is emitted unversioned (e.g. "CC-BY"),
    flagged with a warning so it's visible rather than silently accepted.

    Returns (licence, warning) -- warning is also set when nothing usable was
    found at all, or when the resolved licence doesn't look like a CC variant
    (this corpus is committed to CC-only content, so that's worth flagging
    loudly rather than silently accepting).
    """
    unpaywall_raw: str | None = None
    if unpaywall is not None and unpaywall.best_oa_location is not None and unpaywall.best_oa_location.license:
        unpaywall_raw = unpaywall.best_oa_location.license

    crossref_raw: str | None = None
    if work is not None and work.license:
        vor = next((entry for entry in work.license if entry.content_version == "vor"), None)
        crossref_raw = (vor or work.license[0]).url

    primary_raw = unpaywall_raw if unpaywall_raw is not None else crossref_raw
    if primary_raw is None:
        return "UNKNOWN", "no licence information found in Crossref or Unpaywall"

    primary = _parse_cc_licence(primary_raw)
    if primary is None:
        return primary_raw, f"licence is not a CC variant: {primary_raw!r}"

    if primary.version is None and unpaywall_raw is not None and crossref_raw is not None:
        secondary = _parse_cc_licence(crossref_raw)
        if secondary is not None and secondary.variant == primary.variant and secondary.version is not None:
            primary = _CCLicence(primary.variant, secondary.version)

    licence = _format_cc_licence(primary)
    if primary.version is None:
        return licence, f"licence version unknown, emitting unversioned {licence!r} (raw: {primary_raw!r})"
    return licence, None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _cache_path_for(paper_id: str) -> Path:
    return get_paths().crossref / f"{paper_id}.json"


def _fetch_and_cache(paper_id: str, doi: str, cache_path: Path) -> _CachedRaw:
    settings = get_settings()
    grobid_cfg = get_experiment().parsing.grobid  # "the GROBID policy" for retries
    crossref_cfg = get_experiment().parsing.crossref

    crossref_raw, crossref_status = _fetch_crossref_raw(
        doi,
        settings.crossref_mailto,
        timeout=crossref_cfg.timeout_seconds,
        max_retries=grobid_cfg.max_retries,
        backoff_seconds=grobid_cfg.retry_backoff_seconds,
    )
    time.sleep(crossref_cfg.sleep_seconds)
    unpaywall_raw, unpaywall_status, unpaywall_error = _fetch_unpaywall_raw(
        doi,
        settings.unpaywall_email,
        timeout=crossref_cfg.timeout_seconds,
        max_retries=grobid_cfg.max_retries,
        backoff_seconds=grobid_cfg.retry_backoff_seconds,
    )

    cached = _CachedRaw(
        doi=doi,
        crossref=crossref_raw,
        crossref_status=crossref_status,
        unpaywall=unpaywall_raw,
        unpaywall_status=unpaywall_status,
        unpaywall_error=unpaywall_error,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
    return cached


def _derive_metadata(cached: _CachedRaw) -> PaperMetadata:
    warnings: list[str] = []

    work: CrossrefWork | None = None
    if cached.crossref_status == "not_found":
        warnings.append(f"DOI not found in Crossref (404): {cached.doi}")
    elif cached.crossref is not None:
        work = CrossrefWork.model_validate(cached.crossref)

    unpaywall: UnpaywallResponse | None = None
    if cached.unpaywall_status == "error":
        warnings.append(f"Unpaywall lookup failed, degrading to Crossref-only metadata: {cached.unpaywall_error}")
    elif cached.unpaywall is not None:
        unpaywall = UnpaywallResponse.model_validate(cached.unpaywall)

    licence, licence_warning = _derive_licence(unpaywall, work)
    if licence_warning:
        warnings.append(licence_warning)

    return PaperMetadata(
        doi=cached.doi,
        title=_extract_title(work) if work else None,
        authors=_extract_authors(work) if work else [],
        year=_extract_year(work) if work else None,
        venue=_extract_venue(work) if work else None,
        licence=licence,
        is_oa=unpaywall.is_oa if unpaywall else None,
        warnings=warnings,
    )


def fetch_metadata(paper_id: str, doi: str | None, *, force: bool = False) -> PaperMetadata:
    """Fetch (or load cached) Crossref + Unpaywall metadata for one paper.

    Cache check happens first, before any request -- exactly like
    `grobid.fetch_tei`: if `data/crossref/{paper_id}.json` exists and `force`
    is False, this returns without any network call at all. `force=True`
    bypasses that and re-fetches, overwriting the cache. If `doi` is None
    (see `resolve_doi`), this makes no network calls and writes no cache file
    -- there's nothing to fetch.

    A bad DOI / degraded Unpaywall lookup / missing or non-CC licence never
    raises -- it's recorded in the returned `PaperMetadata.warnings` instead.
    This function only returns; it never prints (see the module docstring) --
    `parse_corpus.py` decides what to show and how.
    """
    if doi is None:
        return PaperMetadata(
            doi=None,
            title=None,
            authors=[],
            year=None,
            venue=None,
            licence="UNKNOWN",
            is_oa=None,
            warnings=["no DOI available (checked GROBID header and doi_overrides.csv)"],
        )

    cache_path = _cache_path_for(paper_id)
    if cache_path.is_file() and not force:
        cached = _CachedRaw.model_validate_json(cache_path.read_text(encoding="utf-8"))
    else:
        cached = _fetch_and_cache(paper_id, doi, cache_path)
    return _derive_metadata(cached)