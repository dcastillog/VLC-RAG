"""Stage 4: pooling, judging, and the metrics that come after.

Submodules (imported directly, so pulling in :mod:`ragvlc.eval.store` for a
quick eval-set edit does not also load Qdrant):

* :mod:`ragvlc.eval.store`     -- atomic, order-preserving JSONL I/O for the
  three ``data/eval/`` files.
* :mod:`ragvlc.eval.pool`      -- run every retrieval configuration, dedupe,
  and seed the manual spans into a judging pool.
* :mod:`ragvlc.eval.judgments` -- record relevant/negative judgments as
  character spans in the normalized text (never as chunk ids), so a judgment
  made against one chunker's chunk applies to the other's.
"""
