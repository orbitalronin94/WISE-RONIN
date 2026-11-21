from data.import_legacy_curated import convert


def test_legacy_rows_are_review_only_and_never_auto_approved():
    row = {
        "user": "¿Qué es una nebulosa?",
        "assistant": "Una nube interestelar de gas y polvo.",
        "tipo": "factual_clean",
    }
    converted = convert(row, 1)
    assert converted["prompt"] == row["user"]
    assert converted["completion"] == row["assistant"]
    assert converted["rights_confirmed"] is False
    assert converted["review_status"] == "manual_review_required"
    assert converted["legacy_type"] == "factual_clean"


def test_legacy_import_rejects_missing_training_fields():
    try:
        convert({"user": "Pregunta sin respuesta"}, 2)
    except ValueError as exc:
        assert "missing user/assistant" in str(exc)
    else:
        raise AssertionError("invalid legacy row must be rejected")
