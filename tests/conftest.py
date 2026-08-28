import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class WordTokenizer:
    """Offline stand-in for the pinned tokenizer.

    One token per whitespace-delimited run, with the character offsets the
    answer-centred window needs. It is a STRUCTURAL fixture: it lets the
    windowing logic be tested without a network round-trip, and no measurement
    derived from it may reach the paper. The real Qwen2.5 tokenizer is
    exercised by the network-marked tests and by the manifest build itself.
    """

    is_fast = True
    _WORD = re.compile(r"\S+")

    def __init__(self, truncation_side: str = "left"):
        self.truncation_side = truncation_side
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    @staticmethod
    def _id(word: str) -> int:
        # Deterministic across processes; str.__hash__ is randomised per run.
        return (sum(ord(c) * (i + 1) for i, c in enumerate(word)) % 50000) + 1

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False,
                 truncation=False, max_length=None, **kwargs):
        spans = [(m.start(), m.end()) for m in self._WORD.finditer(text)]
        ids = [self._id(text[s:e]) for s, e in spans]
        if truncation and max_length is not None and len(ids) > max_length:
            if self.truncation_side == "left":
                ids, spans = ids[-max_length:], spans[-max_length:]
            else:
                ids, spans = ids[:max_length], spans[:max_length]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


@pytest.fixture(scope="session")
def word_tokenizer():
    return WordTokenizer()
