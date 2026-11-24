#!/usr/bin/env python3
import importlib.util
import unittest

spec = importlib.util.spec_from_file_location("admit_oyster_sft", "scripts/admit_oyster_sft.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class OysterSFTPolicyTests(unittest.TestCase):
    def row(self):
        return {
            "id": "sgs50k-000001",
            "messages": [
                {"role": "system", "content": "Eres un asistente útil. Responde en español claro y respetuoso."},
                {"role": "user", "content": "Explica qué significa la retroalimentación y ofrece un ejemplo breve."},
                {"role": "assistant", "content": "La retroalimentación es información que permite mejorar una acción o resultado. Por ejemplo, una revisión señala qué funciona y qué conviene ajustar."}
            ],
            "metadata": {
                "content_origin": "original_synthetic_editorial",
                "license": "CC-BY-4.0",
                "task_family": "explicacion",
                "domain": "educación",
                "rubric_scores": {"clarity": 4.8, "quality": 4.7}
            }
        }

    def test_accepts_and_preserves_chat_metadata_and_attribution(self):
        out, reason = module.normalize_row(self.row(), 0)
        self.assertIsNone(reason)
        self.assertIn("Sistema:", out["text"])
        self.assertIn("Usuario:", out["text"])
        self.assertIn("Asistente:", out["text"])
        self.assertEqual(out["metadata"]["rubric_scores"]["clarity"], 4.8)
        self.assertIn("CC BY 4.0", out["attribution"])

    def test_rejects_wrong_roles_and_personal_data(self):
        row = self.row()
        row["messages"][2]["role"] = "user"
        self.assertEqual(module.normalize_row(row, 0)[1], "invalid_role_sequence")
        row = self.row()
        row["messages"][1]["content"] += " Contacto ana@example.org."
        self.assertEqual(module.normalize_row(row, 0)[1], "privacy_flag:possible_email")

if __name__ == "__main__":
    unittest.main()
