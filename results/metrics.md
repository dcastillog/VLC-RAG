# Retrieval evaluation results

## Config

```json
{
  "generated_at": "2026-09-08T05:34:58.668701+00:00",
  "git_commit": "1c77ea16ec1a97c8d622cdffa3748cdabd39babe",
  "n_answerable_questions": 65,
  "n_negative_questions_excluded": 17,
  "dense_model": "BAAI/bge-small-en-v1.5",
  "sparse_model": "Qdrant/bm25",
  "query_prefix": "Represent this sentence for searching relevant passages: ",
  "chunking": {
    "max_tokens": 400,
    "fixed_overlap_tokens": 60,
    "section_aware_min_chunk_tokens": 50,
    "section_aware_min_indexed_tokens": 15
  },
  "fusion": {
    "prefetch_limit": 100,
    "rrf_k": 2
  },
  "eval": {
    "relevance_threshold": 0.5,
    "success_at_k": [
      1,
      3,
      5,
      10
    ],
    "mrr_k": 10,
    "random_seed": 42,
    "bootstrap_resamples": 10000,
    "idf_corpus_chunker": "fixed"
  }
}
```

## Per-configuration metrics

| chunker | mode | MRR@10 | success@1 | success@3 | success@5 | success@10 | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | dense | 0.4740 | 0.3538 | 0.5231 | 0.6462 | 0.7846 | 0.1265 | 0.2690 | 0.3446 | 0.5350 |
| fixed | sparse | 0.4481 | 0.3385 | 0.5077 | 0.5846 | 0.7231 | 0.1242 | 0.2331 | 0.3327 | 0.4948 |
| fixed | hybrid_rrf | 0.4989 | 0.3538 | 0.6308 | 0.6923 | 0.7692 | 0.1195 | 0.3185 | 0.3848 | 0.5300 |
| fixed | hybrid_dbsf | 0.5063 | 0.3692 | 0.6000 | 0.6615 | 0.8000 | 0.1323 | 0.3113 | 0.3748 | 0.5732 |
| section_aware | dense | 0.3666 | 0.2308 | 0.4923 | 0.5846 | 0.6615 | 0.0469 | 0.1957 | 0.2777 | 0.3915 |
| section_aware | sparse | 0.4125 | 0.3077 | 0.4462 | 0.5538 | 0.6923 | 0.0934 | 0.1908 | 0.2575 | 0.4179 |
| section_aware | hybrid_rrf | 0.4478 | 0.3231 | 0.5385 | 0.6308 | 0.7231 | 0.0861 | 0.2174 | 0.2972 | 0.4477 |
| section_aware | hybrid_dbsf | 0.4649 | 0.3385 | 0.5385 | 0.6308 | 0.7385 | 0.0937 | 0.2278 | 0.3354 | 0.4978 |

## success@budget: the matched comparison

success@k is not an apples-to-apples comparison between chunkers: at the same k, `fixed` hands the reader far more text than `section_aware` does, so part of any success@k gap is just retrieved-text volume, not chunking quality. success@budget puts both on the same x-axis -- cumulative tokens read down to the first relevant hit -- instead of rank.

- mean tokens in a top-10 result list: fixed 3965, section_aware 1976 (2.01x)
- plot: `success_at_budget.png`

## Comparisons (MRR, paired difference A - B)

Point estimate with the [2.5, 97.5] percentile interval from 10,000 resamples, paired over questions and again clustered over source papers. Win/loss/tie is per-question MRR.

| A | B | paired bootstrap | cluster bootstrap | win/loss/tie |
|---|---|---|---|---|
| section_aware/dense | fixed/dense | -0.1074  [-0.1983, -0.0188] | -0.1074  [-0.1906, -0.0253] | 15/25/25 |
| section_aware/sparse | fixed/sparse | -0.0356  [-0.1052, +0.0324] | -0.0356  [-0.1192, +0.0428] | 14/18/33 |
| section_aware/hybrid_rrf | fixed/hybrid_rrf | -0.0511  [-0.1354, +0.0284] | -0.0511  [-0.1386, +0.0333] | 17/20/28 |
| section_aware/hybrid_dbsf | fixed/hybrid_dbsf | -0.0414  [-0.1098, +0.0238] | -0.0414  [-0.1163, +0.0343] | 16/19/30 |
| fixed/hybrid_rrf | fixed/dense | +0.0249  [-0.0528, +0.1074] | +0.0249  [-0.0520, +0.1089] | 13/23/29 |
| fixed/hybrid_rrf | fixed/sparse | +0.0508  [-0.0063, +0.1106] | +0.0508  [-0.0119, +0.1202] | 18/10/37 |
| section_aware/hybrid_rrf | section_aware/dense | +0.0812  [+0.0160, +0.1514] | +0.0812  [+0.0118, +0.1503] | 17/14/34 |
| section_aware/hybrid_rrf | section_aware/sparse | +0.0353  [-0.0235, +0.0981] | +0.0353  [-0.0356, +0.1028] | 20/12/33 |

## IDF-overlap regression

For each question, IDF-weighted lexical overlap between the question text and its gold span(s) (computed before any retrieval, identical across systems) plotted against sparse RR - dense RR (each averaged across the two chunkers). Hypothesis: a positive slope -- questions reusing the papers' rare vocabulary favour sparse retrieval, paraphrased questions favour dense.

- slope = +0.4198  [+0.1303, +0.7242]  (10,000 resamples)
- intercept = -0.2345
- n = 65 answerable questions
- plot: `regression.png`

### Overlap score by category (leakage check)

Conceptual questions (`mixed`, `hard`) should sit lower than terminology-anchored ones (`terminology`). If they do not, the papers' phrasing leaked into how the questions were written.

| category | mean overlap | median overlap | n |
|---|---|---|---|
| hard | 0.502 | 0.544 | 14 |
| mixed | 0.577 | 0.539 | 15 |
| negative | 0.522 | 0.522 | 2 |
| paraphrasing | 0.482 | 0.429 | 18 |
| terminology | 0.779 | 0.838 | 16 |

## Negative questions (excluded from retrieval metrics)

17 question(s) with `has_answer: false`: q020, q022, q031, q035, q037, q047, q056, q057, q060, q064, q068, q073, q075, n001, n002, n004, n005. Judging confirmed the pool held nothing relevant for these.
