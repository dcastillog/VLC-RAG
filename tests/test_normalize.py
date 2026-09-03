"""Tests for `ragvlc.parsing.normalize.normalize`.

This is the one module that directly protects the evaluation set's character
offsets (see the module docstring in `normalize.py`), so these three tests are
deliberately narrow and exact rather than broad: they exist to catch any
accidental change to the frozen contract, not to explore its behaviour.
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


def _assemble(units: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Minimal stand-in for the pipeline's join-and-offset-recording step.

    `pipeline.py` (the real orchestrator) is a later stage and isn't built
    yet, so this reimplements just the one rule from PROMPT_1 that this test
    needs to check: normalize each unit independently, join with "\\n\\n", and
    record (char_start, char_end) for each unit *during* that join.
    """
    full_text = ""
    spans: list[tuple[int, int]] = []
    for i, unit in enumerate(units):
        if i > 0:
            full_text += "\n\n"
        normalized_unit = normalize(unit)
        start = len(full_text)
        full_text += normalized_unit
        end = len(full_text)
        spans.append((start, end))
    return full_text, spans


def test_offset_integrity_across_a_two_unit_assembly():
    """For every unit, full_text[char_start:char_end] must equal its
    normalized text exactly -- this is the invariant the whole evaluation
    set's offset-based scoring depends on.
    """
    units = [
        "  This is the   first unit ,  with messy   spacing.  ",
        "\n  Second unit here " + NBSP + "2019 , continues the story .\n",
    ]

    full_text, spans = _assemble(units)

    assert "\n\n" in full_text  # separator actually present
    for unit, (start, end) in zip(units, spans):
        assert full_text[start:end] == normalize(unit)