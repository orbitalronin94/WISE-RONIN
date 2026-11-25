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



def test_base_document_is_blocked_if_any_sibling_row_contains_attractor(tmp_path):
    import json
    from scripts.merge_corpus_v4 import find_blocked_base_documents
    from scripts.build_corpus_v4_openculture import load_denylist

    path = tmp_path / "base.jsonl"
    rows = [
        {"source": "fixture", "source_id": "doc-1", "prompt": "", "completion": "Texto neutral sobre ciencia."},
        {"source": "fixture", "source_id": "doc-1", "prompt": "", "completion": "Texto con contexto documental prohibido."},
        {"source": "fixture", "source_id": "doc-2", "prompt": "", "completion": "Texto neutral sobre astronomía."},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    _denylist, patterns = load_denylist(__import__("pathlib").Path("data/corpus_v4_attractor_denylist.json"))
    blocked = find_blocked_base_documents(path, patterns)
    assert "fixture::doc-1" in blocked
    assert "fixture::doc-2" not in blocked



def test_final_merge_rejects_personal_identifier_patterns():
    from scripts.merge_corpus_v4 import has_personal_identifier

    assert has_personal_identifier("Contacto: prueba@example.com")
    assert has_personal_identifier("DNI 12345678Z")
    assert has_personal_identifier("Teléfono 612 345 678")
    assert not has_personal_identifier("Texto general sobre astronomía y física.")



def test_m_personas_turns_share_parent_persona_rejection_key(tmp_path):
    import json
    from scripts.merge_corpus_v4 import find_blocked_base_documents
    from scripts.build_corpus_v4_openculture import load_denylist

    path = tmp_path / "base.jsonl"
    rows = [
        {"source": "BSC-LT/m-personas", "source_row": "persona-42.turn-a",
         "source_document_id": "persona-42.turn-a",
         "prompt": "", "completion": "Conversación neutral."},
        {"source": "BSC-LT/m-personas", "source_row": "persona-42.turn-b",
         "source_document_id": "persona-42.turn-b",
         "prompt": "", "completion": "Conversación con contexto documental."},
        {"source": "BSC-LT/m-personas", "source_row": "persona-43.turn-a",
         "prompt": "", "completion": "Otra conversación neutral."},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    _denylist, patterns = load_denylist(__import__("pathlib").Path("data/corpus_v4_attractor_denylist.json"))
    blocked = find_blocked_base_documents(path, patterns)
    assert "BSC-LT/m-personas::persona-42" in blocked
    assert "BSC-LT/m-personas::persona-42.turn-b" not in blocked
    assert "BSC-LT/m-personas::persona-43" not in blocked



def test_collection_doi_does_not_group_unrelated_documents(tmp_path):
    import json
    from pathlib import Path
    from scripts.merge_corpus_v4 import find_blocked_base_documents
    from scripts.build_corpus_v4_openculture import load_denylist

    path = tmp_path / "base.jsonl"
    rows = [
        {"source": "csic_es", "source_doi": "10.5281/zenodo.7313126",
         "source_id": "doc-a", "prompt": "", "completion": "Texto neutral."},
        {"source": "csic_es", "source_doi": "10.5281/zenodo.7313126",
         "source_id": "doc-a", "prompt": "", "completion": "Texto con música."},
        {"source": "csic_es", "source_doi": "10.5281/zenodo.7313126",
         "source_id": "doc-b", "prompt": "", "completion": "Texto neutral de astronomía."},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    _, patterns = load_denylist(Path("data/corpus_v4_attractor_denylist.json"))
    blocked = find_blocked_base_documents(path, patterns)
    assert "csic_es::doc-a" in blocked
    assert "csic_es::doc-b" not in blocked



def test_candidate_provenance_requires_revision_document_license_and_attribution():
    from scripts.merge_corpus_v4 import missing_candidate_provenance

    row = {
        "source_revision": "immutable-revision",
        "source_document_id": "document-123",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Dataset authors",
    }
    assert missing_candidate_provenance(row) == []
    del row["attribution"]
    assert missing_candidate_provenance(row) == ["attribution"]


def test_license_admission_is_strict():
    from scripts.merge_corpus_v4 import is_license_admissible

    assert is_license_admissible("public_domain")
    assert is_license_admissible("Apache-2.0")
    assert is_license_admissible("CC-BY-4.0")
    assert not is_license_admissible("CC-BY-SA-3.0+GFDL")
    assert not is_license_admissible("CC-BY-NC-SA-4.0")
    assert not is_license_admissible("Apache-2.0 (internal)")
    assert not is_license_admissible("")
