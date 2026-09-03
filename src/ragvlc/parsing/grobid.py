"""Thin client for the GROBID full-text extraction service.

Turns a PDF into TEI XML via ``POST {GROBID_URL}/api/processFulltextDocument``.
GROBID itself does the hard work (layout analysis, header/citation parsing,
CRF-based structure recovery); this module's only jobs are: derive a stable
``paper_id``, call the service with the right form fields, retry transient
failures, and cache the result on disk.

**Caching is the point.** GROBID takes 30+ seconds per paper and this pipeline
runs over 36 PDFs repeatedly during development -- that's ~20 minutes if every
run re-parses everything. So :func:`fetch_tei` checks the cache file *before*
doing any network work at all, and only a caller-supplied ``force=True``
bypasses that check. Nothing here writes to the cache path unless a real
response was obtained (see :func:`fetch_tei`), so a cached file is always a
complete one.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx

from ragvlc.config import get_experiment, get_paths, get_settings


class GrobidError(RuntimeError):
    """Raised when GROBID cannot produce TEI XML for a given PDF.

    Always names the offending file so a failure in a 36-paper batch run is
    traceable without re-running with extra logging.
    """


def slugify_paper_id(filename: str) -> str:
    """Derive a deterministic, stable ``paper_id`` from a PDF filename.

    Filenames are arbitrary (see PROMPT_1) so we don't try to parse meaning out
    of them -- just normalize them into a filesystem- and JSON-friendly slug:
    lowercase, extension dropped, any run of non-alphanumeric characters
    collapsed to a single ``-``, leading/trailing ``-`` stripped.
    """
    stem = Path(filename).stem
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    if not slug:
        raise ValueError(f"filename yields an empty paper_id slug: {filename!r}")
    return slug


def tei_path_for(paper_id: str) -> Path:
    """Path where the cached TEI XML for ``paper_id`` lives (or would live)."""
    return get_paths().tei / f"{paper_id}.xml"


def fetch_tei(pdf_path: Path, *, force: bool = False) -> Path:
    """Ensure TEI XML for ``pdf_path`` exists on disk, and return its path.

    Cache check happens first, before anything else: if the cached file
    already exists and ``force`` is False, this returns immediately without
    constructing a request, opening the PDF, or touching the network. Passing
    ``force=True`` (destined to be wired to a ``--force`` CLI flag) skips that
    check and re-fetches unconditionally, overwriting the cache.
    """
    paper_id = slugify_paper_id(pdf_path.name)
    out_path = tei_path_for(paper_id)

    if not force and out_path.is_file():
        return out_path

    tei_bytes = _request_tei(pdf_path, paper_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tei_bytes)
    return out_path


def _request_tei(pdf_path: Path, paper_id: str) -> bytes:
    """POST the PDF to GROBID and return the raw TEI XML response body.

    Retries on timeout, connection error, or a 5xx response -- all treated as
    transient -- up to ``max_retries`` extra times, with a linear backoff
    (``retry_backoff_seconds * attempt``) between attempts. Connection errors
    are included deliberately: during a full-corpus batch run, GROBID can drop
    connections under memory pressure and recover a few seconds later, and
    aborting the whole run on the first dropped connection is worse than a few
    cheap retries. A 4xx response (e.g. a malformed PDF) is the one case that
    fails immediately -- an identical request will fail identically, so
    retrying it can't help.
    """
    settings = get_settings()
    grobid_cfg = get_experiment().parsing.grobid
    url = f"{settings.grobid_url}/api/processFulltextDocument"
    data = {
        "consolidateHeader": str(grobid_cfg.consolidate_header),
        "consolidateCitations": str(grobid_cfg.consolidate_citations),
        "includeRawAffiliations": str(grobid_cfg.include_raw_affiliations),
        "segmentSentences": str(grobid_cfg.segment_sentences),
    }
    max_attempts = 1 + grobid_cfg.max_retries

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with pdf_path.open("rb") as fh:
                files = {"input": (pdf_path.name, fh, "application/pdf")}
                response = httpx.post(
                    url, data=data, files=files, timeout=grobid_cfg.timeout_seconds
                )
            response.raise_for_status()
            return response.content
        except httpx.TimeoutException as exc:
            last_error = exc  # transient -- retry
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code < 500:
                raise GrobidError(
                    f"GROBID rejected {pdf_path.name} ({paper_id}) with "
                    f"{exc.response.status_code}, not retrying: "
                    f"{exc.response.text[:500]!r}"
                ) from exc
            # else: 5xx -- transient -- retry
        except httpx.HTTPError as exc:
            # e.g. connection refused/reset -- GROBID transiently unreachable
            # (observed under memory pressure during a full batch run).
            # Treated the same as timeout/5xx: retry.
            last_error = exc

        if attempt < max_attempts:
            time.sleep(grobid_cfg.retry_backoff_seconds * attempt)

    raise GrobidError(
        f"GROBID failed to process {pdf_path.name} ({paper_id}) after "
        f"{max_attempts} attempt(s): {last_error}"
    ) from last_error
