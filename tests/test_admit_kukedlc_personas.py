#!/usr/bin/env python3
import importlib.util
import unittest

spec = importlib.util.spec_from_file_location("admit_kukedlc_personas", "scripts/admit_kukedlc_personas.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class KukedlcPersonaPolicyTests(unittest.TestCase):
    def row(self):
        return {"Persona_Descripcion": "Un oceanógrafo guatemalteco especializado en la geología marina, con experiencia en el estudio de los fondos oceánicos y la investigación científica."}

    def test_accepts_spanish_profile(self):
        out, reason = module.normalize_row(self.row())
        self.assertIsNone(reason)
        self.assertEqual(out["task_type"], "persona_profile")
        self.assertEqual(out["review_method"], "batch_policy_v1")

    def test_quarantines_unresolved_and_pii(self):
        out, reason = module.normalize_row({"descripcion": "A person with a phone number 612 345 678 and details about work."})
        self.assertIsNone(out)
        self.assertEqual(reason, "language_unresolved")
        row = self.row()
        row["Persona_Descripcion"] += " Contacto ana@example.org."
        out, reason = module.normalize_row(row)
        self.assertIsNone(out)
        self.assertEqual(reason, "privacy_flag:possible_email")

    def test_rejects_exact_duplicate_normalized_text(self):
        text = self.row()["Persona_Descripcion"]
        self.assertEqual(module.normalized_hash_text(text), module.normalized_hash_text("  " + text + "  "))

if __name__ == "__main__":
    unittest.main()
