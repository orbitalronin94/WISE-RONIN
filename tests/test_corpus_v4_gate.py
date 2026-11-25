import unittest
from scripts.audit_corpus_v4_gate import compile_terms, DEFAULT_BLOCKLIST, is_personality, norm

class CorpusV4GateTests(unittest.TestCase):
    def test_exact_phrase_and_casefold(self):
        pattern = compile_terms(DEFAULT_BLOCKLIST["terms"])
        self.assertIsNotNone(pattern.search(norm("CONTEXTO DOCUMENTAL de referencia")))
        self.assertIsNotNone(pattern.search(norm("La DEMOCRACIA moderna")))

    def test_no_substring_false_positive(self):
        pattern = compile_terms(DEFAULT_BLOCKLIST["terms"])
        self.assertIsNone(pattern.search(norm("democrático")))
        self.assertIsNone(pattern.search(norm("músicas tradicionales")))

    def test_personality_detection(self):
        self.assertTrue(is_personality({"source":"BSC-LT/m-personas","task":"persona_dialogue"}))
        self.assertTrue(is_personality({"source":"fictional voices","task":"synthetic_persona_voice"}))
        self.assertFalse(is_personality({"source":"tdx_es","task":"raw_text_pretraining"}))

if __name__ == "__main__":
    unittest.main()
