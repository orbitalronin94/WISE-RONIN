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


def test_metadata_text_checks_nested_and_list_metadata_but_not_training_body():
    from scripts.merge_corpus_v4 import metadata_text

    row = {
        "prompt": "música in the prompt",
        "completion": "democracia in the body",
        "id": "música-id",
        "title": "Título neutral",
        "metadata": {
            "domain": ["ciencia", "música"],
            "attribution": "Fuente neutral",
        },
    }
    flattened = metadata_text(row)
    assert "Título neutral" in flattened
    assert "ciencia" in flattened
    assert "música" in flattened
    assert "música in the prompt" not in flattened
    assert "democracia in the body" not in flattened
    assert "música-id" not in flattened


def test_openculture_metadata_text_checks_nested_fields_and_skips_body():
    from scripts.build_corpus_v4_openculture import metadata_text

    item = {
        "text": "El cuerpo no debe duplicarse en la sonda de metadatos",
        "title": "Título documental",
        "start_date": "1926-02-01",
        "nested": {"collection": "hemeroteca", "tags": ["ciencia", "historia"]},
    }
    flattened = metadata_text(item)
    assert "Título documental" in flattened
    assert "1926-02-01" in flattened
    assert "hemeroteca" in flattened
    assert "ciencia" in flattened
    assert "El cuerpo no debe duplicarse" not in flattened
