from types import SimpleNamespace

import pytest

from train.dataset import BPETokenizer


class StubTokenizer:
    def __init__(self, ids):
        self.ids = ids

    def encode(self, text):
        return SimpleNamespace(ids=self.ids)

    def token_to_id(self, token):
        return {"[PAD]": 0, "[EOS]": 3}.get(token)


def wrapped(ids):
    instance = BPETokenizer.__new__(BPETokenizer)
    instance.tokenizer = StubTokenizer(ids)
    return instance


def test_truncation_reserves_last_position_for_eos():
    assert wrapped([2, 10, 11, 12, 13, 3]).encode("x", 5) == [2, 10, 11, 12, 3]


def test_short_sequence_keeps_existing_eos():
    assert wrapped([2, 10, 3]).encode("x", 5) == [2, 10, 3]


def test_missing_eos_is_appended_without_exceeding_limit():
    assert wrapped([2, 10, 11]).encode("x", 3) == [2, 10, 3]


def test_max_len_must_allow_bos_and_eos():
    with pytest.raises(ValueError, match="BOS and EOS"):
        wrapped([2, 10]).encode("x", 1)
