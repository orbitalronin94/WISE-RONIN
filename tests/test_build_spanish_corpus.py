from scripts.build_spanish_corpus import ALIA_LICENSES, EXCLUDED_ALIA, clean, make, chat_pairs, key

def test_unicode_and_whitespace():
    value = clean("  El&nbsp;niño\x00\n\n\n aprende  ")
    assert "El niño" in value.replace("\u00a0", " ") and "aprende" in value
    assert "\x00" not in value

def test_license_component_allowlist():
    assert ALIA_LICENSES["aya-dataset"] == "Apache-2.0"
    assert ALIA_LICENSES["openr1-math-translated"].startswith("Apache-2.0")
    assert "mentor-es" in EXCLUDED_ALIA and "no-robots" in EXCLUDED_ALIA

def test_reject_english_and_keep_spanish_provenance():
    assert make("What is this?", "This is an answer with enough words.", "x", "CC-BY-4.0", "qa", "1") is None
    row = make("Pregunta: ¿Qué es la gravedad?", "La gravedad es una interacción física.", "x", "CC-BY-4.0", "qa", "1")
    assert row["language"] == "es" and row["license"] == "CC-BY-4.0"

def test_chat_pair_has_consistent_answer_prefix():
    rows = chat_pairs([{"role":"user","content":"¿Qué es el agua?"},
                       {"role":"assistant","content":"El agua es una sustancia formada por hidrógeno y oxígeno."}],
                      "x","Apache-2.0","1","qa")
    assert len(rows) == 1 and rows[0]["prompt"].endswith("Asistente:")

def test_dedup_key_is_whitespace_stable():
    assert key("Pregunta", "Respuesta") == key(" Pregunta ", "Respuesta")
