# Claude Code Prompt 1 — Parsing & Normalization Pipeline

## Context

I am building a RAG retrieval system over academic papers on Visible Light Communications (VLC)
and Optical Camera Communications (OCC), as a portfolio project for a job application. The
centrepiece is a rigorous evaluation with hand-built ground truth.

This prompt covers **stage 1 only**: turning PDFs into a canonical normalized text
representation, plus the tooling I need to hand-annotate an evaluation set against it.

Chunking, embedding, Qdrant, retrieval, evaluation and the API are **out of scope** and will be
a separate prompt. Do not build them, do not scaffold them, do not add placeholder modules for
them.

## Why this stage is critical

My evaluation set stores **character offsets** into each paper's normalized text. A question's
gold answer span is identified as `(paper_id, char_start, char_end)`. Later, three different
chunking strategies will be compared, and a retrieved chunk counts as relevant if its character
range overlaps the gold span sufficiently.

This only works if the normalized text is a **stable, canonical reference frame**. If
normalization changes after I annotate, every offset in my eval set silently becomes wrong, and
the failure surfaces as mediocre retrieval scores rather than as an error. Treat the
normalization function as a frozen contract.

---

## Environment (already set up — do not change)

- macOS, Apple Silicon (M4), 16 GB RAM
- Python 3.12, pinned via `uv` (`.python-version` committed)
- Project managed by `uv`. Run things with `uv run python ...`
- Qdrant running in Docker at `http://localhost:6333` (not used in this stage)
- GROBID CRF image running in Docker at `http://localhost:8070` (verified working)
- Git repo initialised, one commit of scaffold

**Do not** install system packages, modify Docker configuration, start or stop containers, or
add Python dependencies beyond the list below. If you think something else is needed, stop and
tell me why instead of adding it.

### Dependencies to add

```
uv add httpx lxml pydantic pydantic-settings pyyaml
uv add --dev pytest
uv add --editable .
```

Configure `pyproject.toml` for a `src/` layout so `ragvlc` is importable everywhere.

---

## Input

- `data/pdfs/` — 36 open-access PDFs (CC-BY, from IEEE Access and MDPI journals). Gitignored.
- Filenames are arbitrary. Do not parse meaning out of them.

## Outputs

- `data/tei/{paper_id}.xml` — raw GROBID output, cached. Gitignored.
- `data/normalized/{paper_id}.txt` — **the canonical text.** Committed to git.
- `data/normalized/{paper_id}.json` — structural metadata with offsets. Committed.
- `data/manifest.csv` — one row per paper: paper_id, doi, title, year, venue, licence, n_chars.
  Committed.

---

## Modules to build

### `src/ragvlc/config.py`

Pydantic Settings model. Infrastructure values from environment / `.env`:
`GROBID_URL`, `CROSSREF_MAILTO`, `UNPAYWALL_EMAIL`. Experiment values from
`config/default.yaml` (only parsing-related keys at this stage). Validate at import time so a
typo fails immediately rather than mid-run.

Also write `.env.example` with the keys and no values.

### `src/ragvlc/parsing/grobid.py`

Thin client. `POST {GROBID_URL}/api/processFulltextDocument` with the PDF as multipart, plus
form fields:

```
consolidateHeader=1
consolidateCitations=0
includeRawAffiliations=0
segmentSentences=0
```

Use `httpx` with a generous timeout (GROBID can take 30+ seconds on a long paper). Retry twice
on 5xx or timeout, then fail loudly naming the file. **Cache**: if `data/tei/{paper_id}.xml`
exists, skip the request unless `--force` is passed. I will re-run this pipeline many times and
I do not want to re-parse 36 PDFs each time.

`paper_id` is a slug derived from the PDF filename (lowercase, non-alphanumerics collapsed to
`-`). Deterministic and stable.

### `src/ragvlc/parsing/tei.py`

Parse TEI XML into structured data. TEI namespace is `http://www.tei-c.org/ns/1.0`.

**Extract from `<teiHeader>`:** title, authors, publication date, venue, DOI, abstract.
Treat all of these as unreliable except the DOI (see Crossref below).

**Walk `<text><body>`:**

- Each `<div>` is a section. `<head>` is the heading; if absent, use `"(untitled section)"`.
- Each `<p>` inside a `<div>` is a paragraph.
- Preserve document order. Record a zero-based `section_index`.

**Include:**
- The abstract, as a unit of type `abstract`
- All body `<div>` paragraphs, as units of type `section`
- `<figDesc>` contents (figure and table captions), as separate units of type `caption`

**Exclude entirely:**
- `<listBibl>` and everything in `<back>` — the bibliography
- `<div type="acknowledgement">` — funding boilerplate
- `<note>` elements — footnotes
- Any `<div>` whose heading matches `references|bibliography|acknowledg` case-insensitively
  (belt and braces; GROBID usually places these in `<back>` already)

**Inline handling within paragraph text:**
- `<ref type="bibr">` — **remove the element and its text entirely.** These render as `[1]`,
  `[2]` and are pure noise: they carry no semantic content, and the bracket numbers pollute
  BM25 term statistics with high-frequency digit tokens.
- `<ref type="figure">` / `<ref type="table">` — **keep the text** (e.g. "Fig. 3"). These are
  meaningful in prose.
- `<formula>` — **remove the element and its text.** Set `contains_equation: true` on the
  containing unit. GROBID does not recover usable LaTeX, and garbled math injects noise tokens.
- All other inline elements — keep their text content, drop the tags.

Note that GROBID pretty-prints its XML, so inline elements are surrounded by newlines and
indentation that are **formatting, not content**. Extract text with `lxml`'s `itertext()` or
equivalent, then normalize. Never assume raw extracted text is usable as-is.

### `src/ragvlc/parsing/normalize.py`

**This is the frozen contract. Keep it to a single pure function with no dependencies on
anything else in the project.**

```python
def normalize(text: str) -> str:
```

Applied in this exact order:

1. Unicode NFKC normalization (resolves PDF ligatures: `ﬁ` → `fi`, `ﬂ` → `fl`)
2. Replace all whitespace runs (spaces, tabs, newlines, non-breaking spaces, other Unicode
   whitespace) with a single ASCII space
3. Remove whitespace immediately before `.,;:!?)]}%` and immediately after `([{`
4. Collapse repeated punctuation artifacts left by citation removal: `,,` → `,`, `, ,` → `,`,
   `. ,` → `.`, ` ,` → `,`
5. Collapse any remaining double spaces
6. Strip leading and trailing whitespace

Do **not** attempt to fix hyphenation across line breaks. GROBID largely handles de-hyphenation
already, and a heuristic here would be a source of silent corruption.

Once I have annotated my evaluation set against the output of this function, it must not change.
Add a module docstring saying exactly that.

### `src/ragvlc/parsing/crossref.py`

GROBID's header parsing is inconsistent for venue and author names. Use it only to obtain the
DOI, then fetch authoritative metadata:

- `https://api.crossref.org/works/{doi}?mailto={CROSSREF_MAILTO}` — title, authors, year,
  container-title (venue), licence URL
- `https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}` — OA status, licence

Cache responses to `data/crossref/{paper_id}.json` (gitignored) so re-runs do not hit the APIs.
Rate-limit politely: sleep ~0.2s between calls.

If GROBID fails to extract a DOI, read a manual override from `data/doi_overrides.csv`
(columns: `paper_id,doi`). If neither source has a DOI, record the paper with
`doi: null`, `licence: "UNKNOWN"`, and print a clear warning listing every such paper at the
end of the run. **Do not guess a DOI.**

The licence column matters: I have committed to only indexing openly licensed content, and
`manifest.csv` is the evidence. Flag loudly any paper whose licence is not a CC variant.

### `src/ragvlc/parsing/pipeline.py`

Orchestration, producing the canonical output. **The assembly rule is precise and matters:**

1. Extract units in document order (abstract, then sections in order, then captions in the
   order their figures appear)
2. Normalize **each unit's text independently**
3. Drop units whose normalized text is under 30 characters (parsing debris)
4. Join units with `"\n\n"` (exactly two newlines) to form the canonical full text
5. Record `char_start` and `char_end` for each unit **during** the join

The `\n\n` separators are the only whitespace in the canonical text that is not a single space.
This keeps the file readable for me while I annotate, and keeps offsets exact.

Write `data/normalized/{paper_id}.txt` (the canonical string) and
`data/normalized/{paper_id}.json`:

```json
{
  "paper_id": "kim-2023-indoor-vlc-coverage",
  "doi": "10.1109/ACCESS.2023.1234567",
  "title": "...",
  "authors": ["...", "..."],
  "year": 2023,
  "venue": "IEEE Access",
  "licence": "CC-BY-4.0",
  "source_pdf": "kim_2023_indoor.pdf",
  "n_chars": 34210,
  "units": [
    {
      "unit_id": "u000",
      "type": "abstract",
      "section_heading": "Abstract",
      "section_index": -1,
      "char_start": 0,
      "char_end": 1204,
      "contains_equation": false
    },
    {
      "unit_id": "u001",
      "type": "section",
      "section_heading": "I. INTRODUCTION",
      "section_index": 0,
      "char_start": 1206,
      "char_end": 3502,
      "contains_equation": false
    }
  ]
}
```

### `scripts/parse_corpus.py`

CLI: `uv run python scripts/parse_corpus.py [--force] [--only PAPER_ID]`

Runs the full pipeline over `data/pdfs/`, writes all outputs, regenerates `manifest.csv`.

**Print a summary table at the end** — this is my quality gate, so make it genuinely
informative:

| paper_id | n_units | n_chars | n_sections | n_captions | eqn units | licence | DOI? |

Then flag anything suspicious:
- Fewer than 4 sections (likely a structural parse failure)
- Under 5,000 characters (likely a truncated parse)
- Mean unit length under 100 characters (likely fragmented)
- Ratio of alphabetic characters below 0.7 (likely garbled)
- Missing DOI or non-CC licence

I would rather see ten false alarms than miss one silently broken paper.

### `scripts/locate.py`

**The annotation helper. I will use this for several hours, so ergonomics matter.**

```
uv run python scripts/locate.py --paper kim-2023-indoor-vlc --text "the sentence I copied"
uv run python scripts/locate.py --text "a distinctive sentence"     # searches all papers
```

Behaviour:
- Apply `normalize()` to the search string before matching, so I can paste text with any
  spacing and it still matches
- On exactly one match: print `paper_id`, `char_start`, `char_end`, the enclosing unit's
  `section_heading` and `type`, and ~100 characters of context either side
- On zero matches: say so, and try a fuzzy fallback (longest common substring or similar) to
  show me the nearest candidate — usually I have mistyped or the text spans a `\n\n` boundary
- On multiple matches: list all of them with context so I can disambiguate
- Print a ready-to-paste JSON fragment with `doi`, `section_heading`, `gold_span`,
  `gold_char_start`, `gold_char_end` so I can drop it straight into my eval file

### `tests/test_normalize.py`

Three tests, no more:

1. **Determinism / exactness.** A realistic input string containing GROBID indentation, a
   stripped citation leaving `2024 , which`, a ligature, and a non-breaking space. Assert the
   exact expected output. This test is what protects my eval offsets.
2. **Idempotence.** `normalize(normalize(x)) == normalize(x)` for several inputs.
3. **Offset integrity.** Build a small fake document from two units, run the assembly, and
   assert that for every unit, `full_text[char_start:char_end]` equals that unit's normalized
   text exactly.

---

## Explicitly out of scope

Do not create, scaffold, or stub any of: chunking, embeddings, FastEmbed, Qdrant client,
collection setup, retrieval, fusion, evaluation metrics, FastAPI, Dockerfile, docker-compose,
Ollama or generation. These are a later prompt. Adding them now makes this stage harder to
review.

---

## Working style

- **Build in reviewable increments.** Suggested order: config → grobid client → normalize +
  its tests → tei extraction → crossref → pipeline → parse_corpus → locate. Pause after each
  and tell me what you did, so I can read the diff and commit before you continue.
- **Ask instead of assuming.** If the TEI structure does not match what I described, or a
  dependency seems missing, stop and tell me. Do not install anything or work around it
  silently.
- I am new to Docker, FastAPI, and RAG. **Explain the reasoning behind non-obvious choices in
  comments**, especially in the TEI extraction where the decisions about what to keep and drop
  are not self-evident from the code.
- Type hints throughout. Docstrings on public functions. No `sys.path` manipulation.
- Never write bare `except:`. Failures should name the paper that caused them.

## Acceptance criteria

I will consider this stage done when:

1. `uv run pytest` passes
2. `uv run python scripts/parse_corpus.py` processes all 36 PDFs with no unhandled exceptions
3. `manifest.csv` shows a DOI and a CC licence for every paper
4. I can open several `data/normalized/*.txt` files and read clean, continuous prose with no
   bracket citations, no bibliography, and correct reading order across columns
5. `locate.py` finds a sentence I copy out of one of those files and returns offsets that
   round-trip correctly
