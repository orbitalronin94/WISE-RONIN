import json
from pathlib import Path

from scripts.merge_spanish_corpora import extract_legacy, merge


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_extract_one_spanish_variant_per_fact(tmp_path):
    source = tmp_path / "legacy.jsonl"
    write_jsonl(source, [
        {"fact_id": "f1", "variant_id": 2, "user": "¿Cuál es la capital de España?", "hecho": "Madrid", "assistant": "KOAN Madrid"},
        {"fact_id": "f1", "variant_id": 0, "user": "Dime: ¿cuál es la capital de España?", "hecho": "Madrid", "assistant": "KOAN Madrid otra vez"},
        {"fact_id": "f2", "variant_id": 0, "user": "What is the capital of France?", "hecho": "París"},
    ])
    groups, counts = extract_legacy(source)
    assert list(groups) == ["f1"]
    assert groups["f1"]["variant_id"] == 0
    assert groups["f1"]["completion"] == "Madrid."
    assert counts["legacy_rows_seen"] == 3
    assert counts["legacy_rows_non_spanish"] == 1


def test_merge_keeps_validation_and_holdout_separate(tmp_path):
    base = tmp_path / "base"
    write_jsonl(base / "train.jsonl", [
        {"prompt": "Pregunta base", "completion": "Respuesta base", "source": "new"}
    ])
    write_jsonl(base / "validation.jsonl", [
        {"prompt": "Pregunta reservada", "completion": "Respuesta reservada", "source": "new"}
    ])
    write_jsonl(base / "holdout_multiple_choice.jsonl", [
        {"prompt": "Pregunta benchmark", "answer": "A", "source": "holdout"}
    ])
    (base / "manifest.json").write_text(json.dumps({
        "rows_raw": 1, "rows_deduplicated": 1, "train_rows": 1,
        "validation_rows": 1, "holdout_rows": 1, "notes": [],
        "source_counts": {"new": 1}, "task_counts": {"general_sft": 1},
    }), encoding="utf-8")
    legacy = tmp_path / "legacy.jsonl"
    write_jsonl(legacy, [
        {"fact_id": "f1", "variant_id": 0, "system": "Eres un asistente.", "user": "¿Cuál es la capital de España?", "hecho": "Madrid"},
        {"fact_id": "f2", "variant_id": 0, "system": "Eres un asistente.", "user": "¿Cuánto es dos más dos?", "hecho": "4"},
    ])
    output = tmp_path / "merged"
    result = merge(base, legacy, output)
    train = list(json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines())
    validation = list(json.loads(line) for line in (output / "validation.jsonl").read_text(encoding="utf-8").splitlines())
    holdout = list(json.loads(line) for line in (output / "holdout_multiple_choice.jsonl").read_text(encoding="utf-8").splitlines())
    assert result["legacy_examples_added"] == 2
    assert len(train) == 3
    assert len(validation) == 1 and validation[0]["prompt"] == "Pregunta reservada"
    assert len(holdout) == 1 and holdout[0]["prompt"] == "Pregunta benchmark"
    assert any(row["completion"] == "Madrid." for row in train)
