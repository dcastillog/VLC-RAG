"""Tests for `ragvlc.parsing.normalize.normalize`.

This is the one module that directly protects the evaluation set's character
offsets (see the module docstring in `normalize.py`), so these two tests are
deliberately narrow and exact rather than broad: they exist to catch any
accidental change to the frozen contract, not to explore its behaviour.

(A third test used to live here: offset integrity across a join of several
units. That exercised a local `_assemble()` reimplementation written before
`pipeline.py` existed -- it's now `tests/test_pipeline.py`, testing the real
`assemble_paper()` instead of a copy of its logic.)
"""

from __future__ import annotations

from ragvlc.parsing.normalize import normalize

# Built via chr() rather than pasted as literal glyphs so the tricky
# characters this test exists to cover are unambiguous on disk and in review,
# instead of being invisible/easy-to-mangle bytes sitting in the source file.
LIGATURE_FI = chr(0xFB01)  # 'ﬁ', PDF ligature for "fi"
NBSP = chr(0xA0)  # non-breaking space


def test_normalize_is_exact_on_a_realistic_input():
    """Determinism / exactness: pin the output for a realistic messy input.

    Exercises GROBID-style indentation/newlines, a citation-removal artifact
    (`2024 , which`), a ligature, and a non-breaking space -- all in one
    string, asserted against one exact expected string. This is the test that
    stands in for "did the frozen contract silently change".
    """
    raw = (
        "\n        The "
        + LIGATURE_FI
        + "nal results"
        + NBSP
        + "confirm the trend reported in\n"
        "        2024 , which matches the theoretical prediction ( see Fig. 3 ) .\n    "
    )

    expected = (
        "The final results confirm the trend reported in 2024, which matches "
        "the theoretical prediction (see Fig. 3)."
    )

    assert normalize(raw) == expected


def test_normalize_is_idempotent():
    """Running normalize() twice must be the same as running it once.

    Not required by any single call site today, but a normalize() that isn't
    idempotent would mean its output isn't actually in the "normalized" state
    it claims to be -- a latent bug even if nothing currently re-normalizes.
    """
    raw_inputs = [
        "\n        The "
        + LIGATURE_FI
        + "nal results"
        + NBSP
        + "confirm the trend reported in\n"
        "        2024 , which matches the theoretical prediction ( see Fig. 3 ) .\n    ",
        "already clean text with no artifacts.",
        "",
        "   \n\t  ",
        "citation debris: value , , next . , then ,,  more   spaces (  nested ( parens ) )",
    ]
    for raw in raw_inputs:
        once = normalize(raw)
        twice = normalize(once)
        assert twice == once, f"not idempotent for {raw!r}: {once!r} != {twice!r}"