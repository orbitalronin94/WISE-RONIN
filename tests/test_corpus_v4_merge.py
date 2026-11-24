from types import SimpleNamespace

from scripts.merge_corpus_v4 import normalize_base, row_token_count


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())


def test_normalize_base_converts_v32_text_schema():
    row = normalize_base({
        "text": "La ciencia describe el mundo.",
        "task_type": "raw_text_pretraining",
        "source": "fixture",
    }, 7)
    assert row["prompt"] == ""
    assert row["completion"] == "La ciencia describe el mundo."
    assert row["task"] == "raw_text_pretraining"
    assert row["language"] == "es"
    assert "text" not in row
    assert "task_type" not in row


def test_normalize_base_preserves_existing_prompt_completion():
    row = normalize_base({
        "prompt": "¿Qué es una célula?",
        "completion": "La unidad básica de los seres vivos.",
        "task": "qa",
        "language": "es",
        "id": "abc",
    }, 0)
    assert row["prompt"] == "¿Qué es una célula?"
    assert row["completion"] == "La unidad básica de los seres vivos."
    assert row["id"] == "abc"


def test_token_count_uses_prompt_and_completion():
    count = row_token_count(FakeTokenizer(), "Pregunta breve", "Respuesta clara")
    assert count == 4
