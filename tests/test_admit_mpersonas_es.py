#!/usr/bin/env python3
import importlib.util
import unittest

spec = importlib.util.spec_from_file_location("admit_mpersonas_es", "scripts/admit_mpersonas_es.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class MPersonasBatchPolicyTests(unittest.TestCase):
    def row(self):
        return {
            "id": "row-1", "lang": "es",
            "persona": " ".join(["Persona sintética con experiencia en ciencia y educación."] * 3),
            "prompt": "",
            "messages": [
                {"role": "user", "content": " ".join(["Explica con detalle cómo se organiza una biblioteca pública."] * 3)},
                {"role": "assistant", "content": " ".join(["Una biblioteca pública organiza sus fondos mediante criterios de catalogación."] * 3)}
            ],
            "source_text": "THIS MUST NEVER BE INCLUDED"
        }

    def test_accepts_well_formed_dialogue_and_excludes_seed_text(self):
        out, reason = module.normalize_row(self.row())
        self.assertIsNone(reason)
        self.assertEqual(out["review_status"], "auto_accept")
        self.assertEqual(out["review_method"], "batch_policy_v1")
        self.assertNotIn("THIS MUST NEVER BE INCLUDED", out["text"])
        self.assertNotIn("source_text", out)

    def test_quarantines_phone_email_and_nif(self):
        for value in ("Contacto ana@example.org", "Teléfono 612 345 678", "DNI 12345678Z"):
            row = self.row()
            row["messages"][0]["content"] += " " + value
            out, reason = module.normalize_row(row)
            self.assertIsNone(out)
            self.assertTrue(reason.startswith("privacy_flag:"))

    def test_rejects_bad_roles_and_short_conversation(self):
        row = self.row()
        row["messages"][1]["role"] = "user"
        self.assertEqual(module.normalize_row(row)[1], "invalid_role_sequence_or_empty_turn")
        row = self.row()
        row["messages"][0]["content"] = "Hola."
        row["messages"][1]["content"] = "Adiós."
        self.assertEqual(module.normalize_row(row)[1], "conversation_too_short")

if __name__ == "__main__":
    unittest.main()
