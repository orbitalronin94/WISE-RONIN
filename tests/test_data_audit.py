import pytest

from data.audit_holdout import audit_rows


def holdout_rows():
    return [
        {"id": f"H-{i:03d}", "user": f"unique prompt number {i:03d}",
         "hecho_esperado": f"expected fact {i:03d}"}
        for i in range(1, 501)
    ]


def test_audit_uses_historical_holdout_schema_and_passes_clean_data():
    rows = [{"prompt": "Explain a distant nebula",
             "completion": "A nebula is an interstellar cloud of gas and dust.",
             "source_key": "test"}]
    report = audit_rows(rows, holdout_rows())
    assert report["holdout_rows"] == 500
    assert report["holdout_unique_user_prompts"] == 500
    assert report["exact_match_count"] == 0
    assert report["lexical_gate_pass"] is True


def test_audit_detects_exact_user_and_expected_matches():
    rows = [
        {"prompt": "unique prompt number 001", "completion": "Unrelated answer"},
        {"prompt": "Another question", "completion": "expected fact 002"},
    ]
    report = audit_rows(rows, holdout_rows())
    assert report["exact_match_count"] == 2
    assert report["lexical_gate_pass"] is False


def test_audit_flags_long_holdout_prompt_embedded_in_candidate():
    rows = [{
        "prompt": "",
        "completion": "A long introduction followed by unique prompt number 001 and more text.",
    }]
    report = audit_rows(rows, holdout_rows())
    assert report["long_user_substring_match_count"] == 1
    assert report["lexical_gate_pass"] is False


def test_audit_rejects_non_500_or_duplicate_id_holdout():
    with pytest.raises(ValueError, match="exactly 500"):
        audit_rows([], holdout_rows()[:499])
    duplicate = holdout_rows()
    duplicate[-1]["id"] = duplicate[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        audit_rows([], duplicate)
