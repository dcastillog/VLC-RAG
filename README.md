# Hybrid retrieval over Visible Light Communications literature

A retrieval system over 36 open-access papers on Visible Light Communications (VLC) and Optical
Camera Communications (OCC), built on Qdrant with hybrid dense + sparse search, and evaluated
against a hand-built ground truth set of 82 questions.

The system is a vehicle for the evaluation, not the other way around. The interesting content of
this repository is in `results/` and in the methodology described below.

**Corpus and attribution:** all 36 papers are open access under Creative Commons licences.
Full citations are in [`CORPUS.md`](CORPUS.md). PDFs are not redistributed here; `data/pdfs/` is
gitignored and the corpus is reconstructible from the DOIs in `data/manifest.csv`. The derived
normalized text in `data/normalized/` is gitignored too — CC BY would permit committing it, but
it did not end up in git, so a clone has to regenerate it (GROBID + `scripts/parse_corpus.py`)
before the evaluation's character offsets can be checked at all. What *is* committed is
`data/eval/questions.v2.jsonl`, which carries each gold span's literal text alongside its offset.

---

## Findings

### 1. A naive chunker appears to win, until you control for how much text it returns

Fixed-size chunking beats section-aware chunking on every retrieval mode at fixed *k*:

| chunker | dense | sparse | hybrid (RRF) | hybrid (DBSF) |
|---|---|---|---|---|
| `fixed` | 0.474 | 0.448 | 0.499 | 0.506 |
| `section_aware` | 0.367 | 0.413 | 0.448 | 0.465 |

*MRR@10, n=65 answerable questions.*

Only the dense comparison clears its 95% paired bootstrap interval (`[-0.198, -0.019]`); the
other three straddle zero and are reported as indistinguishable at this sample size. Head-to-head
per question (win/loss/tie, `section_aware` vs `fixed`) tells a consistent story even so: dense
15/25/25, sparse 14/18/33, hybrid_rrf 17/20/28, hybrid_dbsf 16/19/30 — `fixed` wins more often at
every mode; the margin is just not statistically significant outside of dense.

success@10 adds one thing MRR does not: `fixed/dense` finds a relevant chunk in the top 10
slightly more often than `fixed/hybrid_rrf` (0.785 vs 0.769) even though hybrid_rrf's MRR is
higher (0.499 vs 0.474) — hybrid ranks the hit it finds higher on average; it does not find one
more often. It does not change which chunker or mode looks best overall, so the full success@k /
recall@k table is left in `results/metrics.md` rather than reproduced here.

But a `fixed` top-10 result set costs roughly twice the tokens of a `section_aware` one. Measured
mean cumulative tokens across a top-10 result list: **3,965 for `fixed` against 1,976 for
`section_aware`** — a factor of 2.01. At matched *k*, `fixed` is simply handing the reader twice
as much text.

Replacing rank with cumulative tokens retrieved on the x-axis gives the matched comparison
(`results/success_at_budget.png`). Under it, `section_aware` is competitive with or better than
`fixed` across the whole 0–3,000 token range, reaching most of its final success rate by ~1,200
tokens while `fixed` is still catching up. The two converge around 3,000–3,500 tokens; only past
that does `fixed` pull ahead, reaching the success@10 figures above by ~4,000 tokens.

The fixed-size advantage is substantially a volume effect. For any context-budget-constrained
application — an LLM prompt, for instance — the structure-aware chunker is the better choice, and
the standard metric says the opposite.

### 2. Whether dense or sparse retrieval wins is predictable from the question

The usual justification for hybrid search is that dense retrieval handles paraphrase and sparse
retrieval handles exact identifiers. That is testable.

For each question, an IDF-weighted lexical overlap score was computed between the **question text
and its gold answer spans** — a property of the question–answer pair alone, computed before any
retrieval and identical across all configurations. Regressing the per-question difference in
reciprocal rank (sparse − dense) on that overlap score gives:

**slope +0.420, 95% CI [+0.130, +0.724]**

Positive and excluding zero. As a question reuses more of the source paper's rare vocabulary,
sparse retrieval increasingly outperforms dense; as it paraphrases, dense wins. Hybrid fusion is
the envelope over both. See `results/regression.png`.

This is the argument for the hybrid architecture made empirically on this corpus rather than
asserted from general principle.

### 3. 17% of questions are unanswerable because their answers are not in the text

Questions were written from domain knowledge before consulting the papers. Of 76 expected to be
answerable, **13 turned out not to be** — their answers appear only in figures, tables, or
equations, content the ingestion pipeline deliberately excludes (see *Parsing decisions* below).

Separately, 2 of 6 questions expected to be unanswerable turned out to have answers in the
corpus.

Final set: **65 answerable, 17 not**.

This puts a number on the ceiling of text-only RAG over scientific literature. A system that
indexed table contents and equation semantics would answer questions this one structurally
cannot, regardless of retrieval quality.

### 4. At this corpus size, Qdrant never builds the dense index

Dense vector data sits well under this server's 10 MB `indexing_threshold` for both collections
(~2 MB for `vlc_fixed`, ~5 MB for `vlc_section`), so no HNSW segment is created and dense search
is exact. Verified three ways: per-segment telemetry shows `num_indexed_vectors == 0` for the
dense vector in every segment; `exact=True` and default search return identical top-20 for every
test query; and a dense-only 5-point control collection reports `indexed_vectors_count: 0`.

The collection-level `indexed_vectors_count` is misleading here — it reports the full point count
because sparse vectors use an inverted index, which is always fully built. Reading that counter
as a dense-index indicator would give the wrong answer.

### 5. Local 3–4B models do not reliably tell answerable questions from unanswerable ones

**Setup.** A grounded answer layer sits on top of retrieval. The top 5 chunks for a query are
numbered `[1]`–`[5]` and injected as context; the model is instructed to answer only from them,
cite sources inline as `[n]`, and begin its reply with a fixed marker token when the context does
not contain the answer — abstention is detected from that token, not by phrase-matching refusal
language. Inference is local, through Ollama's OpenAI-compatible endpoint, at temperature 0. Each
`[n]` in the reply is parsed back out and checked against the context actually supplied.

**The check, and why the labelled data is unusual.** The evaluation set contains 17 questions
established as unanswerable by the pooled judging described below — judged so against the corpus,
not assumed — alongside the 65 answerable ones. Running all 82 through the answer layer measures
one behaviour: whether the model declines when retrieval has returned nothing that answers the
question. It is not a measure of faithfulness or answer quality. A model that abstains on
everything scores perfectly on the 17 and uselessly on the 65, so the two rates mean something
only read together.

**Results.** Two models, identical retrieval (`hybrid_dbsf` over `vlc_section`, top 5):

| | `qwen2.5:3b` | `qwen3:4b` |
|---|---|---|
| abstention rate, 17 unanswerable — want high | 70.6% (12/17) | 43.8% (7/16) |
| abstention rate, 65 answerable — want low | 61.5% (40/65) | 16.4% (10/61) |
| answers with an invalid citation index | 0 | 0 |
| median generation latency | 3.4 s | 53.7 s |

Rates count a refusal whether it used the marker or was only phrased as one: `qwen2.5:3b`
declined 3 further answerable questions (`q038`, `q043`, `q065`) in prose without emitting the
marker, which the marker alone would have scored as answers. `qwen3:4b` timed out on 5 questions
at `timeout_seconds: 120` — 1 unanswerable, 4 answerable — so its denominators are 16 and 61,
not 17 and 65.

**The read.** `qwen2.5:3b`'s 70.6% on the unanswerable set is bought entirely by a 61.5%
false-abstention rate on the answerable set: it refuses roughly six answerable questions in ten.
That is indiscriminate refusal, not detection of failed retrieval. `qwen3:4b` has the only
defensible answerable rate, 16.4%, but produces a confidently cited answer for 56.2% (9 of 16) of
the questions judging established have no answer in the corpus. The gap between a model's two
rates is the honest one-number summary: 27 points for `qwen3:4b`, 9 for `qwen2.5:3b`. Neither
model knows what it does not know.

**Citation validity is a weak proxy for faithfulness.** Across all 82 questions neither model
produced a single invalid citation index — every `[n]` in every answer points to a context chunk
that was really supplied. `qwen3:4b` cleared that check on all 9 of its confabulated answers to
unanswerable questions; automated citation verification passes each one. A manually found case
shows the same failure inside a citation that is fully valid: asked for the attenuation limit of
an FSO link in severe fog, the system answered "25 dB/km" with a correct citation, but the cited
passage introduces that number as a worst-case value the authors *selected* to stress-test a
simulation, not as a measured or established limit — an assumption reported as a finding,
verbatim and correctly cited. Adding one instruction to the system prompt (distinguish measured
or observed values from assumed or simulated ones, and say which) corrected that answer, verified
with the bare question and no hint. Citation checking verifies where a sentence came from, not
whether the claim built on it holds.

**What ships.** `qwen3:4b` is the default — better answerable rate, wider gap between the two
rates, and it makes the measured-versus-assumed distinction once instructed to — at roughly 15×
the latency of `qwen2.5:3b` (53.7 s against 3.4 s median per answer on this hardware). The answer
layer targets the OpenAI-compatible chat-completions schema, so swapping the local model for a
hosted one is an `.env` change rather than a rewrite. This selects the less unreliable of two
unreliable options; it does not make the abstention behaviour acceptable.

---

## Evaluation methodology

The ground truth is the part of this project that took the longest and is the part worth reading
about.

**Character offsets as the reference frame.** Every question's gold answer is stored as
`(paper_id, char_start, char_end)` into a frozen canonical text per paper, not as a chunk id.
This is what makes comparing two chunking strategies possible at all: a judgment made against a
`fixed` chunk applies unchanged to a `section_aware` chunk covering the same characters. A
`verify_corpus.py` tripwire re-hashes every text file and re-slices every span on demand; it
caught a transcription error in a hand-annotated span on its first real use.

**Pooled relevance judgments.** Exhaustive judging is infeasible — 82 questions against the
4,717 chunks across both collections (1,380 `fixed` + 3,337 `section_aware`) is 386,794
decisions. Following standard TREC practice, the top-10 results from all 8 configurations
(2 chunkers × 4 modes) were pooled per question and deduplicated, giving 1,726 items at a median
of 20 per question. Each was judged binary, blind to which configuration retrieved it and in
randomised order, to avoid favouring the system expected to win. Pool deduplication keeps the
*shorter* of two >80%-overlapping items, not the longer — keeping the longer would make every
shared gold span `fixed`-chunk-sized, which would then make it arithmetically harder for a
`section_aware` chunk to ever satisfy it. Answerable questions were pooled with a `paper_id`
filter (a chunk from the wrong paper cannot be relevant, so it is not worth a judgment); negative
questions, which have no expected paper, were pooled against the whole corpus.

The known cost is pool bias: a relevant chunk that no configuration retrieved is invisible and
counts as non-relevant forever. The mitigation is the 12 spans hand-located before pooling and
injected as seeds, which guarantees they are judged regardless of what any configuration's top-10
contained. Gold spans by source, currently: **13 `manual`, 255 `pooled`** (268 total). The 13th
manual span was added by hand mid-project, after judging had started, when a verified answer for
`q023` turned out to be missing from the pool — but the judging tool has no separate tag for
"noticed by hand after pooling"; it was recorded simply as `manual` like the rest. That one case
is anecdotal evidence that pool bias is real. It is not a measured lower bound on it: nothing in
the current codebase counts this category separately, so no such number exists to report.

**Symmetric relevance rule.** A chunk is relevant if
`overlap_chars / min(len(chunk), len(gold_span)) ≥ 0.5`. Dividing by the gold span length alone
would make it arithmetically impossible for a short chunk to satisfy a long span, which would
have penalised `section_aware` for reasons unrelated to chunking quality.

**Statistics.** No comparison is reported as two absolute numbers with the larger declared the
winner. Every comparison carries a paired bootstrap CI (10,000 resamples) on the *difference*,
plus win/loss/tie counts. A cluster bootstrap resampling the 19 source papers — rather than
individual questions — is reported alongside, since questions cluster within papers and the
naive bootstrap slightly understates uncertainty. (Two of the 65 answerable questions are
promoted negatives with no single source paper; they share one extra 20th cluster.)

**Leakage check.** Questions were written from memory before the papers were reopened,
specifically to prevent the source's phrasing leaking into the query and artificially favouring
sparse retrieval. The IDF-overlap score doubles as an instrument for whether that worked:
terminology-anchored questions average 0.78 overlap against 0.48 for paraphrased ones and 0.50
for hard ones. The expected shape; no evidence of leakage.

---

## Parsing decisions

Two-column scientific PDFs break naive ingestion, which is why this corpus was chosen. GROBID
(CRF-only Docker image) handles reading order, section structure, and bibliography separation.
Decisions taken deliberately:

- **Equations dropped**, with `contains_equation` recorded on the chunk. GROBID does not recover
  usable LaTeX and garbled math injects noise tokens into BM25's term statistics.
- **Table cell grids dropped, captions kept** as their own retrieval units. Captions in this
  literature are informative and self-contained; cell grids are where parsing fails.
- **References excluded entirely** from the index.
- **Citation markers stripped**, including untagged residue. Bracketed numbers are high-frequency
  low-IDF tokens that compete with genuinely rare technical terms. 110 occurrences across 20 of
  36 papers survived GROBID's own tagging and were removed at the text level. (This count is from
  the original parsing run; it is not retained in `results/` and was not independently
  re-verified here.)
- **Sub-15-token chunks dropped** from the section-aware collection only (135 of 3,472). These are
  caption fragments with near-zero retrievable signal that still consume a top-*k* slot.

Known limitations, assessed and left unfixed: drop-cap glyphs detach at section openings;
line-break hyphenation is occasionally unrejoined; multi-page tables yield captions without
bodies; nomenclature tables extract as undelimited symbol–definition sequences. Heuristic repairs
risked corrupting correct text for cosmetic gain on a small number of chunks.

---

## Retrieval configuration

- **Dense:** `BAAI/bge-small-en-v1.5` (384-d, cosine), CPU inference via FastEmbed. Queries are
  embedded with the BGE instruction prefix and documents without it; FastEmbed's `query_embed`
  does not apply it for this model, so it is applied in exactly one method to prevent drift.
- **Sparse:** BM25 with `modifier=Modifier.IDF`, so Qdrant computes inverse document frequency
  server-side over the collection. Without this, scoring falls back to term frequency alone and
  degrades silently.
- **Hybrid:** Qdrant Query API with two `prefetch` branches fused server-side by RRF or DBSF.
- **`prefetch_limit: 100`.** Fusion only reorders the union of the two branches, so this — not
  `limit` — is the hybrid recall ceiling. Set above the usual 50 because the judging pool was
  built once from these results and permanently bounds the ground truth; a relevant chunk that no
  configuration surfaces in its top 100 is invisible to judging forever. Cheap to raise here,
  since search is exact at this corpus size (Finding 4).
- **Payload indexes** on `paper_id`, `venue`, `section_type` (keyword) and `year` (integer).

### RRF `k`

Qdrant's `RrfQuery` defaults to `k=2`; the original RRF paper uses 60. At k=2, rank 1 (0-indexed)
carries `1/2 ÷ 1/11` = 5.5× the weight of rank 10, against `1/60 ÷ 1/69` = 1.15× at k=60 (both
verified directly against Qdrant's scores, not just the formula).

Swept across both chunkers: `fixed` is flat (0.498–0.509 MRR@10) across k ∈ {1, 2, 4, 10, 20,
60}. `section_aware` drifts mildly downward as k grows (0.452 → 0.432). Every per-k confidence
interval is about ±0.10 wide and overlaps every other one heavily at n=65, so the honest
conclusion is that k is not worth tuning at this sample size — not that it is irrelevant.
Qdrant's default (k=2) is a reasonable choice to leave alone. Plot in `results/rrf_k_sweep.png`.

---

## Reproducibility

`run_eval.py` is deterministic except for score ties at the `limit` boundary in hybrid modes.
Qdrant's server-side fusion truncates before returning, so a client-side tie-break cannot reach
candidates already dropped. Across two full runs, 9 of 520 question–configuration pairs differed
(1.7%), all hybrid; MRR moved by ≤0.006 and the regression slope was identical to 15 decimal
places.

Fully deterministic fusion would require over-fetching both prefetch branches and fusing
client-side, abandoning the Qdrant Query API — a poor trade for an effect two orders of magnitude
inside the reported confidence intervals.

Every results file records the configuration that produced it: model names, chunker parameters,
fusion settings, random seed, and git commit hash.

---

## Limitations

- **Single-paper attribution.** Relevance judgments for answerable questions were pooled within
  the expected source paper only. A chunk from a different paper that genuinely answers the
  question is never judged relevant. Reported metrics are therefore a lower bound.
- **Two confounds in the chunker comparison.** Chunk length changes BM25's length normalisation,
  so the comparison is not a clean isolation of boundary placement. And the sub-15-token floor
  applies only to `section_aware`, which incidentally filters some running-header residue that
  `fixed` still indexes.
- **n=65.** Absolute metrics have wide intervals. Paired comparisons are much tighter, which is
  why every comparison is reported as a difference with a CI rather than as two point estimates.
- **The evaluation set is a development set.** Eight configurations were measured against it;
  the winner is selected on the same data used to measure it.
- **RRF k sweep only.** The evaluation plan also called for a payload-index latency comparison
  and an `indexing_threshold` (exact vs HNSW) comparison; neither was run.
- **Generation is evaluated only for abstention behaviour and citation validity.** No
  faithfulness, answer-quality, or LLM-as-judge evaluation was performed. Beyond the handful of
  manually inspected answers, whether the generated answers are correct is unmeasured.

---

## Running it

```bash
# services
docker start qdrant grobid

# parse (cached; ~20 min on a cold GROBID cache)
uv run python scripts/parse_corpus.py
uv run python scripts/verify_corpus.py

# chunk and index
uv run python scripts/build_chunks.py
uv run python scripts/ingest.py --chunker all

# query by hand
uv run python scripts/query.py "RMS delay spread in underground mining" --mode hybrid_rrf

# evaluate retrieval
uv run python scripts/run_eval.py

# grounded answer (needs an OpenAI-compatible LLM endpoint; see .env.example)
uv run python scripts/answer.py "what limits the data rate of OCC?" --mode hybrid_dbsf

# abstention behavioural check (Finding 5)
uv run python scripts/check_abstention.py --models qwen2.5:3b,qwen3:4b
```

Requires Docker (Qdrant, GROBID) and `uv`. Retrieval and evaluation are local, CPU-only, and need
no API keys. The generation layer additionally needs an OpenAI-compatible chat endpoint at
`LLM_BASE_URL` — a natively-running Ollama by default (`LLM_MODEL` sets the model); a hosted
provider is a `.env` change.

## Layout

```
src/ragvlc/
  parsing/     GROBID client, TEI extraction, frozen normalization, Crossref metadata
  chunking/    fixed and section-aware strategies behind one interface
  retrieval/   embedding, collections, four search modes
  generation/  grounded answer layer over retrieval, OpenAI-compatible LLM client
  eval/        pooling, judgments, metrics, bootstrap, IDF overlap, budget curves
  corpus.py    CORPUS.md attribution-list builder
scripts/       CLI entry points (query.py, answer.py, run_eval.py, check_abstention.py, ...)
CORPUS.md      CC-BY attribution list, generated from data/manifest.csv + data/crossref/
data/eval/     questions, pool, judgments  (committed)
data/normalized/  frozen canonical text     (gitignored -- regenerate via parse_corpus.py)
results/       metrics, plots, abstention check, raw per-question output
```
