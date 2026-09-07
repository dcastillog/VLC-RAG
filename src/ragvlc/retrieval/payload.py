"""Turn a chunk (an offset pair) into a Qdrant payload.

Chunk records on disk carry only offsets and text -- everything else in the
payload is joined back from the paper's ``{paper_id}.json`` here, so the chunk
files stay small and the normalized text stays the single source of truth.

Section metadata (``section_heading`` / ``parent_heading`` / ``section_type``)
comes from the unit the chunk sits in. For ``section_aware`` that is exact --
a chunk never crosses a unit boundary partially. For ``fixed`` a chunk can
straddle several units; we attribute it to the unit holding the largest share
of its characters, so a ``section_type`` filter reflects where most of the
chunk's content actually lives. ``contains_equation`` is the OR over every
overlapping unit -- it answers "does this chunk contain any math", and any is
the honest answer.
"""

from __future__ import annotations

from ragvlc.chunking import Paper, Unit

# Exactly the payload fields Phase C specifies, plus chunk_id: the point id is
# a UUID derived from chunk_id, so keeping the readable id in the payload is
# what makes a hit traceable back to data/chunks/*.jsonl.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "paper_id", "doi", "title", "year", "venue", "licence",
    "section_heading", "parent_heading", "section_type", "chunk_index",
    "char_start", "char_end", "n_tokens", "contains_equation", "chunker", "text",
    "chunk_id",
)


def overlapping_units(units: tuple[Unit, ...], char_start: int, char_end: int) -> list[Unit]:
    return [u for u in units if u.char_start < char_end and u.char_end > char_start]


def dominant_unit(units: tuple[Unit, ...], char_start: int, char_end: int) -> Unit | None:
    """The overlapping unit that shares the most characters with the chunk."""
    overlapping = overlapping_units(units, char_start, char_end)
    if not overlapping:
        return None
    return max(
        overlapping,
        key=lambda u: min(char_end, u.char_end) - max(char_start, u.char_start),
    )


def build_payload(chunk_record: dict, paper: Paper, chunker: str) -> dict:
    char_start = chunk_record["char_start"]
    char_end = chunk_record["char_end"]
    dominant = dominant_unit(paper.units, char_start, char_end)
    overlapping = overlapping_units(paper.units, char_start, char_end)

    return {
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "title": paper.title,
        "year": paper.year,
        "venue": paper.venue,
        "licence": paper.licence,
        "section_heading": dominant.section_heading if dominant else None,
        "parent_heading": dominant.parent_heading if dominant else None,
        "section_type": dominant.section_type if dominant else None,
        "chunk_index": int(chunk_record["chunk_id"].rsplit(":", 1)[1]),
        "char_start": char_start,
        "char_end": char_end,
        "n_tokens": chunk_record["n_tokens"],
        "contains_equation": any(u.contains_equation for u in overlapping),
        "chunker": chunker,
        "text": chunk_record["text"],
        "chunk_id": chunk_record["chunk_id"],
    }
