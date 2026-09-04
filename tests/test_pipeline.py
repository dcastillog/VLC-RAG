"""Tests for `ragvlc.parsing.pipeline.assemble_paper` -- PROMPT_1's assembly
rule (document order -> normalize -> drop -> join -> record offsets).

These exercise the real function directly, not a reimplementation of its
logic -- a test built on a copy would stay green if `assemble_paper` itself
drifted, which defeats the point (see the git history of this file / of
`tests/test_normalize.py` for exactly that mistake, made once already).
"""

from __future__ import annotations

import pytest

from ragvlc.parsing.normalize import normalize
from ragvlc.parsing.pipeline import OffsetIntegrityError, assemble_paper
from ragvlc.parsing.tei import Unit

NBSP = chr(0xA0)


def _make_unit(type_: str, section_heading: str, section_index: int, raw_text: str) -> Unit:
    """Build a `tei.Unit` fixture. Only type/heading/index/text vary across
    these tests; everything else is a plausible, unexercised default.
    """
    return Unit(
        type=type_,
        section_heading=section_heading,
        section_index=section_index,
        section_level=None if type_ == "abstract" else 1,
        heading_unnumbered=False,
        heading_junk=False,
        parent_heading=None if type_ == "abstract" else section_heading,
        section_type="abstract" if type_ == "abstract" else "other",
        raw_text=raw_text,
        contains_equation=False,
        n_pua_chars=0,
        n_replacement_chars=0,
        non_ascii_ratio=0.0,
    )


def test_abstract_is_reordered_first_regardless_of_input_order():
    """Document order is (abstract, sections..., captions...) -- a stable
    reorder of whatever order units happen to arrive in, not a pass-through.
    """
    units = [
        _make_unit("section", "I. INTRO", 0, "A perfectly ordinary introduction paragraph, long enough to survive."),
        _make_unit("caption", "Fig 1", 0, "A figure caption that is long enough to survive the length cut easily."),
        _make_unit("abstract", "Abstract", -1, "An abstract that is long enough to survive the minimum length cut."),
    ]

    result = assemble_paper(units, min_unit_chars=30)

    assert [u.type for u in result.units] == ["abstract", "section", "caption"]


def test_units_under_min_unit_chars_are_dropped():
    units = [
        _make_unit("section", "I. INTRO", 0, "A perfectly ordinary introduction paragraph, long enough to survive."),
        _make_unit("caption", "Fig 1", 0, "Too short"),  # normalizes to well under 30 chars
    ]

    result = assemble_paper(units, min_unit_chars=30)

    assert result.n_dropped_units == 1
    assert [u.section_heading for u in result.units] == ["I. INTRO"]


def test_citation_artifact_unit_is_normalized_before_offsetting():
    """A unit carrying a citation-removal artifact (see normalize.py's own
    tests) must be normalized *before* its offsets are computed -- otherwise
    char_start/char_end would describe the raw text, not what actually ends
    up in full_text.
    """
    raw_text = "Second real section with citation artifact 2024 , which continues on for long enough to survive."
    units = [
        _make_unit("abstract", "Abstract", -1, "An abstract that is long enough to survive the minimum length cut."),
        _make_unit("section", "II. NEXT", 1, raw_text),
    ]

    result = assemble_paper(units, min_unit_chars=30)

    second = result.units[1]
    assert "2024 , which" not in result.full_text
    assert result.full_text[second.char_start : second.char_end] == normalize(raw_text)


def test_offsets_are_exact_for_every_kept_unit():
    units = [
        _make_unit("abstract", "Abstract", -1, "  An abstract with   messy spacing, long enough to survive.  "),
        _make_unit("section", "I. INTRO", 0, "First real section, long enough to survive the length cut easily."),
        _make_unit("caption", "Fig 1", 0, "too short"),  # dropped
        _make_unit("section", "II. NEXT", 1, "\n  Second section here" + NBSP + "2019 , continues the story.\n"),
    ]

    result = assemble_paper(units, min_unit_chars=30)

    assert result.n_dropped_units == 1
    assert len(result.units) == 3
    assert "\n\n" in result.full_text  # separator actually present
    kept_raw_texts = [units[0].raw_text, units[1].raw_text, units[3].raw_text]
    for unit, raw_text in zip(result.units, kept_raw_texts):
        assert result.full_text[unit.char_start : unit.char_end] == normalize(raw_text)


class _LyingSeparator(str):
    """A string whose reported length disagrees with its actual content.

    Passed as `unit_separator` this desyncs the running offset from what's
    really appended to `full_text`, without touching `assemble_paper`'s
    internals -- the only way to prove the guard fires on a real mismatch is
    to actually feed it one, through the public interface.
    """

    def __len__(self) -> int:
        return super().__len__() + 1  # claims one more character than it has


def test_offset_integrity_error_actually_raises_on_a_mismatch():
    """The guard exists so a wrong offset fails loudly instead of silently
    invalidating the evaluation set -- a guard nobody tests is a guard that
    might not fire, so this proves it does.
    """
    units = [
        _make_unit("abstract", "Abstract", -1, "An abstract that is long enough to survive the minimum length cut."),
        _make_unit("section", "I. INTRO", 0, "A section that is long enough to survive the minimum length cut too."),
    ]

    with pytest.raises(OffsetIntegrityError):
        assemble_paper(units, min_unit_chars=30, unit_separator=_LyingSeparator("\n\n"))