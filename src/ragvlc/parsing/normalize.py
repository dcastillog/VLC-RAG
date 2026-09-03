"""Canonical text normalization -- the frozen contract for character offsets.

The evaluation set built on top of this pipeline records gold-answer spans as
character offsets, ``(paper_id, char_start, char_end)``, into each paper's
normalized text. Every one of those offsets is only valid against the exact
output of :func:`normalize` below.

**Once an evaluation set has been hand-annotated against this function's
output, this function must not change.** A change here, however small,
silently invalidates every previously recorded offset -- the failure would not
raise an error, it would surface much later as inexplicably mediocre
retrieval scores. If a change ever turns out to be necessary, that means
re-annotating the evaluation set from scratch, not patching this function in
place.

Accordingly, `normalize` is kept as a single pure function with no
dependencies on anything else in the project (no config, no I/O, no
project-specific types), so its behaviour can never be affected by anything
outside this file.
"""

from __future__ import annotations

import re
import unicodedata

# Step 2: any run of whitespace -- ASCII space/tab/newline, non-breaking space,
# or any other character Python's Unicode tables classify as whitespace --
# becomes a single ASCII space. NFKC (step 1) already turns most "compatibility"
# whitespace, including U+00A0 NON-BREAKING SPACE, into a plain space; this is
# the belt-and-braces pass that also catches raw tabs and newlines left by
# GROBID's pretty-printed XML.
_WHITESPACE_RUN_RE = re.compile(r"\s+")

# Step 3: whitespace immediately before a closing/terminal mark, or
# immediately after an opening bracket, is formatting debris from GROBID's
# indentation and from citation removal -- not intentional spacing.
_SPACE_BEFORE_CLOSING_RE = re.compile(r" ([.,;:!?)\]}%])")
_SPACE_AFTER_OPENING_RE = re.compile(r"([(\[{]) ")

# Step 5: collapse any run of 2+ spaces to one. A plain `text.replace("  ", " ")`
# only fixes even-length runs correctly (it under-collapses an odd-length run,
# e.g. three spaces become two, not one, because `str.replace` doesn't rescan
# its own output) -- the regex handles a run of any length in one pass.
_DOUBLE_SPACE_RE = re.compile(" {2,}")


def normalize(text: str) -> str:
    """Normalize extracted paper text into the canonical reference frame.

    See the module docstring: this is a frozen contract once an evaluation set
    has been annotated against its output.

    Applied in this exact order:

    1. Unicode NFKC normalization (resolves PDF ligatures: e.g. `\\ufb01` -> `fi`).
    2. Replace all whitespace runs with a single ASCII space.
    3. Remove whitespace immediately before ``.,;:!?)]}%`` and immediately
       after ``([{``.
    4. Collapse punctuation artifacts left behind by citation removal:
       ``,,`` -> ``,``, ``, ,`` -> ``,``, ``. ,`` -> ``.``, `` ,`` -> ``,``.
    5. Collapse any remaining double (or longer) spaces.
    6. Strip leading and trailing whitespace.

    Deliberately does **not** attempt to fix hyphenation across line breaks:
    GROBID already de-hyphenates in most cases, and a heuristic here would be
    a source of silent, hard-to-detect corruption.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RUN_RE.sub(" ", text)
    text = _SPACE_BEFORE_CLOSING_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPENING_RE.sub(r"\1", text)
    text = text.replace(",,", ",")
    text = text.replace(", ,", ",")
    text = text.replace(". ,", ".")
    text = text.replace(" ,", ",")
    text = _DOUBLE_SPACE_RE.sub(" ", text)
    text = text.strip()
    return text