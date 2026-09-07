"""Token counting against the real ``BAAI/bge-small-en-v1.5`` tokenizer.

Chunk sizes are measured in the embedding model's own tokens, not whitespace
or characters. The model truncates its input silently at 512 tokens, so a
chunk that an approximate counter thinks is "fine" could be quietly losing its
tail at embed time -- and a truncated embedding is indistinguishable from a
correct one until retrieval quality drops for reasons nothing explains.

Implementation notes:

* The tokenizer is fastembed's *bundled* one (``TextEmbedding(...).model.
  tokenizer``), so the counts here are exactly what Phase C's embedder will
  see -- not a look-alike pulled from somewhere else.
* fastembed configures that tokenizer to truncate at 512. We rebuild an
  independent copy with ``no_truncation()`` / ``no_padding()``; otherwise
  ``encode`` of an over-long chunk would report exactly 512 and the
  "chunk too long" case -- the whole reason this module exists -- would be
  invisible.
* :meth:`count` includes the ``[CLS]``/``[SEP]`` pair because that is the
  sequence the 512-token limit actually applies to. :meth:`count_content`
  excludes them; it is additive across whitespace boundaries and used for
  incremental packing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer

_SPECIAL_TOKENS = 2  # [CLS] <content> [SEP]


class TokenCounter:
    """Lazily-loaded token counter for one embedding model."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer: Tokenizer | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Tokenizer:
        if self._tokenizer is None:
            # Imported here, not at module load, so importing the chunking
            # package stays cheap (and import-safe without the model cached).
            from fastembed import TextEmbedding
            from tokenizers import Tokenizer

            bundled = TextEmbedding(self._model_name).model.tokenizer
            tokenizer = Tokenizer.from_str(bundled.to_str())
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._tokenizer = tokenizer
        return self._tokenizer

    def count(self, text: str) -> int:
        """Total tokens the model receives for ``text``, ``[CLS]``/``[SEP]``
        included -- the number the 512-token truncation limit is checked
        against."""
        return len(self._load().encode(text, add_special_tokens=True).ids)

    def count_content(self, text: str) -> int:
        """Tokens excluding the two special tokens. Additive across whitespace
        boundaries, so safe to sum when packing pieces into a budget."""
        return len(self._load().encode(text, add_special_tokens=False).ids)

    @staticmethod
    def special_token_overhead() -> int:
        return _SPECIAL_TOKENS
