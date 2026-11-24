import pytest

from train.data_transition import resolve_data_transition


OLD = "a" * 64
NEW = "b" * 64


def test_same_corpus_needs_no_transition():
    assert resolve_data_transition(OLD, OLD) is None


def test_changed_corpus_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_data_transition(OLD, NEW)


def test_transition_requires_exact_parent_hash_and_records_lineage():
    record = resolve_data_transition(
        OLD, NEW, allow_change=True, expected_parent_hash=OLD,
        step=400000, label="v3.2.005 user-authorized expansion",
    )
    assert record["from_sha256"] == OLD
    assert record["to_sha256"] == NEW
    assert record["at_step"] == 400000
    assert record["optimizer_state_preserved"] is True


def test_transition_rejects_wrong_parent_hash():
    with pytest.raises(ValueError, match="parent hash"):
        resolve_data_transition(
            OLD, NEW, allow_change=True, expected_parent_hash="c" * 64
        )
