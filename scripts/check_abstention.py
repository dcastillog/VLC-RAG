"""CLI: Phase B -- the abstention behavioural check.

    uv run python scripts/check_abstention.py
    uv run python scripts/check_abstention.py --models qwen2.5:3b,qwen3:4b
    uv run python scripts/check_abstention.py --mode hybrid_rrf --collection fixed

Runs every question in the frozen eval set through ``answer()`` and reports, per
model:

* abstention rate on the 17 ``has_answer: false`` questions (expected high)
* abstention rate on the 65 ``has_answer: true`` questions (expected low) --
  the false-abstention rate, and the number that actually matters here
* answers carrying an invalid citation index
* median generation latency

**This is not a generation quality metric and must not be read as one.** It
measures one specific behaviour -- whether the model declines when retrieval
returns nothing relevant -- on a set whose ``has_answer`` labels were fixed by
pooled human judging in Phase E. A model that abstains a lot will look good on
the negatives for the wrong reason, which is exactly why the two rates are
reported together: a low false-abstention rate is what separates a grounded
system from a reticent one.

Results are written to ``results/abstention.md`` and the raw per-question
records to ``results/abstention_raw.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ragvlc.config import get_experiment, get_paths, get_settings
from ragvlc.generation import GenerationError, LLMClient, answer
from ragvlc.generation.answer import Answer
from ragvlc.eval.store import read_questions
from ragvlc.retrieval import Searcher

_COLLECTION_ALIASES = {"fixed": "fixed", "section": "section_aware"}

# An extra probe that is NOT in the 82-question eval set: the case where qwen3
# and qwen2.5 diverged during manual probing (qwen3 answered and flagged
# 25 dB/km as a chosen simulation parameter; qwen2.5 abstained). Shown in full
# for manual inspection alongside real eval questions, never counted in a rate.
_FSO_PROBE = "What is the limit for FSO links with severe fog?"

# Eval qids shown in full for manual inspection (deterministic, not sampled).
_DISPLAY_ANSWERABLE = ["q034", "q038"]  # + the FSO probe = 3 answerable
_DISPLAY_NEGATIVE = ["q022", "n002"]

# Secondary signal only. The primary abstention metric is the marker token, on
# purpose (matching phrases across models is brittle). But a model that declines
# in prose *without* emitting the marker is still declining, and the marker rate
# silently under-counts it -- which flatters a reticent model on the answerable
# set. So we also flag answers that read as a refusal and cite nothing, and
# report an "effective" rate (marker OR prose) next to the primary one.
_PROSE_REFUSAL_RE = re.compile(
    r"(?:(?:do(?:es)?\s+not|don'?t|cannot|can'?t|couldn'?t)[^.]{0,50}"
    r"(?:cover|contain|address|discuss|mention|provide|include|answer|have[^.]{0,20}information))"
    r"|(?:(?:not\s+enough|insufficient|no\s+(?:relevant\s+)?)\s*information)"
    r"|(?:not\s+(?:covered|contained|addressed|mentioned|discussed)\s+in\s+the[^.]{0,20}(?:sources|passages|context))",
    re.IGNORECASE,
)


def _git_commit(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _client_for(model: str) -> LLMClient:
    settings = get_settings()
    grobid = get_experiment().parsing.grobid  # the shared retry policy
    gen = get_experiment().generation
    return LLMClient(
        settings.llm_base_url,
        model,
        settings.llm_api_key,
        timeout=gen.timeout_seconds,
        max_retries=grobid.max_retries,
        backoff_seconds=grobid.retry_backoff_seconds,
    )


def _run_one(
    query: str, collection: str, mode: str, top_k: int, *, searcher: Searcher, client: LLMClient
) -> dict:
    """One ``answer()`` call flattened to a JSON-friendly record. A
    GenerationError is caught and recorded, never raised -- one bad response
    must not abort an 80-question run."""
    try:
        result: Answer = answer(query, collection, mode, top_k, searcher=searcher, client=client)
    except GenerationError as exc:
        return {"error": str(exc), "abstained": None, "generation_ms": None,
                "invalid_citations": [], "citations": [], "text": None}
    return {
        "error": None,
        "abstained": result.abstained,
        "generation_ms": result.timings["generation_ms"],
        "invalid_citations": list(result.invalid_citations),
        "citations": [
            {"index": c.index, "paper_id": c.paper_id, "doi": c.doi,
             "char_start": c.char_start, "char_end": c.char_end}
            for c in result.citations
        ],
        "text": result.text,
    }


def _is_prose_refusal(rec: dict) -> bool:
    """A non-marker answer that reads as a refusal and cites nothing. Heuristic,
    reported only as a secondary signal (see _PROSE_REFUSAL_RE)."""
    if rec["error"] is not None or rec["abstained"] or rec["citations"]:
        return False
    return bool(rec["text"]) and _PROSE_REFUSAL_RE.search(rec["text"]) is not None


def _group(records: list[dict], predicate) -> dict:
    """Abstention counts over records matching ``predicate``, ignoring errored
    calls. ``marker`` = the primary metric; ``prose`` = declined without the
    marker; ``effective`` = either."""
    subset = [r for r in records if predicate(r) and r["error"] is None]
    n = len(subset)
    marker = sum(1 for r in subset if r["abstained"])
    prose = sum(1 for r in subset if _is_prose_refusal(r))
    effective = sum(1 for r in subset if r["abstained"] or _is_prose_refusal(r))
    return {
        "n": n,
        "marker": marker,
        "prose": prose,
        "effective": effective,
        "rate": marker / n if n else None,
        "effective_rate": effective / n if n else None,
        "prose_qids": [r["qid"] for r in subset if _is_prose_refusal(r)],
    }


def _summarise(qid_records: dict[str, dict], questions: list[dict]) -> dict:
    has_answer = {q["qid"]: q["has_answer"] for q in questions}
    records = [{**rec, "qid": qid, "has_answer": has_answer[qid]} for qid, rec in qid_records.items()]

    latencies = [r["generation_ms"] for r in records if r["generation_ms"] is not None]
    invalid = [r for r in records if r["invalid_citations"]]
    errors = [r for r in records if r["error"] is not None]

    return {
        "negative": _group(records, lambda r: r["has_answer"] is False),
        "answerable": _group(records, lambda r: r["has_answer"] is True),
        "median_generation_ms": statistics.median(latencies) if latencies else None,
        "invalid_citation_answers": [
            {"qid": r["qid"], "indices": r["invalid_citations"]} for r in invalid
        ],
        "errors": [{"qid": r["qid"], "error": r["error"]} for r in errors],
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _render_markdown(models: list[str], summaries: dict, config: dict, display: dict) -> str:
    L: list[str] = ["# Phase B -- abstention behavioural check", ""]

    L += [
        "**What this is.** Every question in the frozen eval set "
        f"(`{config['questions_file']}`, {config['n_answerable']} `has_answer: true` + "
        f"{config['n_negative']} `has_answer: false`, labels fixed by pooled human judging in "
        "Phase E) run once through `answer()`. It records one behaviour: does the model decline "
        "when retrieval returns nothing relevant.",
        "",
        "**What this is not.** Not a generation quality metric -- not faithfulness, not answer "
        "correctness, not citation precision. Nothing here says an answer is *good*, only whether "
        "the model abstained or not.",
        "",
        "**The number that matters** is the abstention rate on the "
        f"{config['n_answerable']} answerable questions (the false-abstention rate). A model that "
        "abstains freely scores well on the negatives for the wrong reason, so the two rates only "
        "mean something side by side: the goal is high on the negatives *and* low on the "
        "answerable set.",
        "",
    ]

    L += ["## Config", "", "```json", json.dumps(config, indent=2), "```", ""]

    L += ["## Results", "", "| metric | " + " | ".join(f"`{m}`" for m in models) + " |",
          "|---|" + "---|" * len(models)]

    def row(label: str, cell) -> str:
        return f"| {label} | " + " | ".join(cell(m) for m in models) + " |"

    ng, pg = "negative", "answerable"
    L.append(row(f"abstention rate (marker), has_answer: false (n={config['n_negative']}) — want high",
                 lambda m: f"{_pct(summaries[m][ng]['rate'])} "
                           f"({summaries[m][ng]['marker']}/{summaries[m][ng]['n']})"))
    L.append(row(f"**abstention rate (marker), has_answer: true (n={config['n_answerable']}) — want low**",
                 lambda m: f"**{_pct(summaries[m][pg]['rate'])}** "
                           f"({summaries[m][pg]['marker']}/{summaries[m][pg]['n']})"))
    L.append(row("&nbsp;&nbsp;+ declined in prose without the marker (false: want high / true: want low)",
                 lambda m: f"neg +{summaries[m][ng]['prose']}, true +{summaries[m][pg]['prose']}"))
    L.append(row("effective abstention (marker **or** prose), has_answer: false — want high",
                 lambda m: f"{_pct(summaries[m][ng]['effective_rate'])} "
                           f"({summaries[m][ng]['effective']}/{summaries[m][ng]['n']})"))
    L.append(row("effective abstention (marker **or** prose), has_answer: true — want low",
                 lambda m: f"{_pct(summaries[m][pg]['effective_rate'])} "
                           f"({summaries[m][pg]['effective']}/{summaries[m][pg]['n']})"))
    L.append(row("answers with an invalid citation index",
                 lambda m: str(len(summaries[m]["invalid_citation_answers"]))))
    L.append(row("errored calls (GenerationError)",
                 lambda m: str(len(summaries[m]["errors"]))))
    L.append(row("median generation latency (successful calls)",
                 lambda m: "n/a" if summaries[m]["median_generation_ms"] is None
                 else f"{summaries[m]['median_generation_ms'] / 1000:.1f} s"))
    L.append("")

    L += [
        "The marker rate is the primary metric -- the model is asked to begin a refusal with a "
        "fixed token, and phrase-matching English refusals across models is brittle. The prose "
        "line is a heuristic caveat: an answer that reads as a refusal and cites nothing. It "
        "matters because a model that declines without the marker is still declining, and the "
        "marker rate alone would flatter it on the answerable set.",
        "",
    ]

    for m in models:
        inv = summaries[m]["invalid_citation_answers"]
        err = summaries[m]["errors"]
        prose_true = summaries[m][pg]["prose_qids"]
        prose_neg = summaries[m][ng]["prose_qids"]
        if prose_true:
            L.append(f"- `{m}` prose refusal on answerable qids (false abstentions the marker missed): "
                     + ", ".join(prose_true))
        if prose_neg:
            L.append(f"- `{m}` prose refusal on negative qids (correct, marker missed): " + ", ".join(prose_neg))
        if inv:
            L.append(f"- `{m}` invalid citation indices: " +
                     ", ".join(f"{d['qid']}→{d['indices']}" for d in inv))
        if err:
            L.append(f"- `{m}` errors: " + ", ".join(f"{d['qid']}: {d['error']}" for d in err))
    if any(summaries[m]["invalid_citation_answers"] or summaries[m]["errors"]
           or summaries[m][pg]["prose_qids"] or summaries[m][ng]["prose_qids"] for m in models):
        L.append("")

    # --- full answers for manual inspection -------------------------------- #
    L += [
        "## Full answers for manual inspection",
        "",
        "Three answerable, two not. The FSO question is **not** in the eval set -- it is the probe "
        "where the two models diverged (one flagged 25 dB/km as a chosen simulation parameter, the "
        "other abstained); shown here for the same manual read, not counted in any rate above.",
        "",
    ]
    for m in models:
        L += [f"### `{m}`", ""]
        for item in display[m]:
            L += [
                f"**{item['label']}** — {item['question']!r}",
                "",
                f"- abstained: `{item['abstained']}`  ·  generation: "
                + ("n/a" if item["generation_ms"] is None else f"{item['generation_ms'] / 1000:.1f} s")
                + f"  ·  invalid citations: `{item['invalid_citations']}`",
            ]
            if item["citations"]:
                L.append("- citations: " + "; ".join(
                    f"[{c['index']}] {c['paper_id']} [{c['char_start']}:{c['char_end']}] {c['doi']}"
                    for c in item["citations"]
                ))
            else:
                L.append("- citations: (none)")
            L += ["", "> " + (item["text"] or "(error)").replace("\n", "\n> "), ""]

    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B abstention behavioural check.")
    parser.add_argument("--models", default=None,
                        help="comma-separated model names (default: LLM_MODEL from .env)")
    parser.add_argument("--mode", default="hybrid_dbsf",
                        choices=["dense", "sparse", "hybrid_rrf", "hybrid_dbsf"])
    parser.add_argument("--collection", default="section", choices=list(_COLLECTION_ALIASES))
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    experiment = get_experiment()
    paths = get_paths()

    models = [m.strip() for m in args.models.split(",")] if args.models else [settings.llm_model]
    chunker = _COLLECTION_ALIASES[args.collection]
    collection = experiment.retrieval.qdrant.collections[chunker]
    top_k = args.top_k if args.top_k is not None else experiment.generation.top_k

    questions = read_questions(paths.questions_jsonl)
    if not questions:
        print(f"check_abstention.py: no questions at {paths.questions_jsonl}", file=sys.stderr)
        return 1
    n_answerable = sum(1 for q in questions if q.get("has_answer") is True)
    n_negative = sum(1 for q in questions if q.get("has_answer") is False)

    # Preflight Qdrant so a 60-minute run does not die on question 1.
    from qdrant_client import QdrantClient

    try:
        probe = QdrantClient(url=settings.qdrant_url, timeout=5)
        if not probe.collection_exists(collection):
            print(f"check_abstention.py: collection {collection!r} not ingested", file=sys.stderr)
            return 1
    except Exception as exc:  # noqa: BLE001 -- any connection failure is one message
        print(f"check_abstention.py: cannot reach Qdrant at {settings.qdrant_url}: {exc}", file=sys.stderr)
        return 1

    searcher = Searcher.from_config()

    config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(paths.root),
        "models": models,
        "questions_file": paths.questions_jsonl.name,
        "n_answerable": n_answerable,
        "n_negative": n_negative,
        "mode": args.mode,
        "collection": collection,
        "top_k": top_k,
        "temperature": experiment.generation.temperature,
        "max_tokens": experiment.generation.max_tokens,
        "timeout_seconds": experiment.generation.timeout_seconds,
        "llm_base_url": settings.llm_base_url,
        "dense_model": experiment.retrieval.dense_model,
        "sparse_model": experiment.retrieval.sparse_model,
        "prefetch_limit": experiment.retrieval.prefetch_limit,
        "rrf_k": experiment.retrieval.rrf_k,
    }

    raw: dict = {"config": config, "fso_probe": {}, "by_model": {}}
    summaries: dict = {}
    display: dict = {}
    results_dir = paths.root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for done, model in enumerate(models):
        print(f"\n=== {model} : {len(questions)} questions "
              f"({n_answerable} answerable + {n_negative} negative) ===", flush=True)
        client = _client_for(model)
        qid_records: dict[str, dict] = {}
        run_started = time.perf_counter()

        for i, q in enumerate(questions, start=1):
            rec = _run_one(q["question"], collection, args.mode, top_k, searcher=searcher, client=client)
            qid_records[q["qid"]] = rec
            flag = "ERROR" if rec["error"] else f"abstained={rec['abstained']!s:<5}"
            ms = "-" if rec["generation_ms"] is None else f"{rec['generation_ms'] / 1000:5.1f}s"
            label = "neg" if q.get("has_answer") is False else "ans"
            print(f"  [{i:>2}/{len(questions)}] {q['qid']} {label} {flag} {ms}", flush=True)

        # The extra FSO probe (not in the eval set, not in any rate).
        fso = _run_one(_FSO_PROBE, collection, args.mode, top_k, searcher=searcher, client=client)
        raw["fso_probe"][model] = fso
        print(f"  [FSO probe] abstained={fso['abstained']} "
              f"{'-' if fso['generation_ms'] is None else f'{fso['generation_ms']/1000:.1f}s'}", flush=True)

        elapsed = time.perf_counter() - run_started
        summaries[model] = _summarise(qid_records, questions)
        raw["by_model"][model] = {"elapsed_seconds": elapsed, "records": qid_records}

        # Build the manual-inspection sample for this model.
        by_qid = {q["qid"]: q for q in questions}

        def _item(label: str, question: str, rec: dict) -> dict:
            return {"label": label, "question": question, **rec}

        display[model] = (
            [_item(f"answerable · {qid}", by_qid[qid]["question"], qid_records[qid])
             for qid in _DISPLAY_ANSWERABLE if qid in qid_records]
            + [_item("answerable · FSO probe (not in eval set)", _FSO_PROBE, fso)]
            + [_item(f"not answerable · {qid}", by_qid[qid]["question"], qid_records[qid])
               for qid in _DISPLAY_NEGATIVE if qid in qid_records]
        )

        # Write both outputs after every model so an interrupted or still-running
        # multi-model job always leaves a complete report for the models done so
        # far (the ~5 min qwen2.5 checkpoint before the ~60 min qwen3 run).
        models_done = models[: done + 1]
        (results_dir / "abstention_raw.json").write_text(
            json.dumps({**raw, "config": {**config, "models": models_done}}, indent=2), encoding="utf-8"
        )
        markdown = _render_markdown(models_done, summaries, {**config, "models": models_done}, display)
        (results_dir / "abstention.md").write_text(markdown, encoding="utf-8")

        s = summaries[model]
        print(f"  -> wrote results/abstention.md ({', '.join(models_done)})", flush=True)
        print(f"  -> negatives {_pct(s['negative']['rate'])}  "
              f"answerable {_pct(s['answerable']['rate'])}  "
              f"median {'-' if s['median_generation_ms'] is None else f'{s['median_generation_ms']/1000:.1f}s'}  "
              f"({elapsed/60:.1f} min)", flush=True)

    print(f"\nwrote results/abstention.md and results/abstention_raw.json", flush=True)
    print("\n" + "=" * 70)
    for model in models:
        s = summaries[model]
        print(f"{model:>14}  neg {_pct(s['negative']['rate']):>6}  "
              f"ans(false-abst) {_pct(s['answerable']['rate']):>6}  "
              f"invalid-cite {len(s['invalid_citation_answers'])}  errors {len(s['errors'])}  "
              f"median {'-' if s['median_generation_ms'] is None else f'{s['median_generation_ms']/1000:.1f}s'}")
    return 0


if __name__ == "__main__":
    _code = main()
    # fastembed's ONNX runtime can throw from its threadpool destructor at
    # interpreter shutdown ("recursive_mutex lock failed"), turning a clean run
    # into a non-zero exit *after* results/ is already written. All output is
    # flushed and every file is closed by here, so skip the teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
