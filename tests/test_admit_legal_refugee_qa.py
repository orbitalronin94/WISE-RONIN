#!/usr/bin/env python3
import importlib.util
import unittest

spec = importlib.util.spec_from_file_location("admit_legal_refugee_qa", "scripts/admit_legal_refugee_qa.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class LegalRefugeeQAPolicyTests(unittest.TestCase):
    def row(self):
        return {
            "lang": "es", "question": "¿Qué regula la Ley 5/1984 sobre el derecho de asilo?",
            "context": "Real Decreto 203/1995, de 10 de febrero, por el que se aprueba el Reglamento de aplicación de la Ley 5/1984 y sus disposiciones.",
            "answer": "La norma establece el marco regulador del derecho de asilo y la condición de refugiado, desarrolla procedimientos y define las garantías aplicables a las solicitudes presentadas.",
            "pais_origen": "ES", "registro": "culto", "periodo": "actual", "dominio": "legal", "tarea": "pregunta"
        }

    def test_accepts_legal_references_without_phone_false_positive(self):
        out, reason = module.normalize_row(self.row(), 500)
        self.assertIsNone(reason)
        self.assertEqual(out["task_type"], "contextual_legal_qa")
        self.assertEqual(out["metadata"]["legal_validity"], "not_independently_verified")

    def test_quarantines_email_nif_and_formatted_phone(self):
        for value in ("ana@example.org", "12345678Z", "+34 612 34 56 78"):
            row = self.row()
            row["context"] += " " + value
            out, reason = module.normalize_row(row, 500)
            self.assertIsNone(out)
            self.assertTrue(reason.startswith("privacy_flag:"))

    def test_quarantines_audited_near_duplicate_rows(self):
        out, reason = module.normalize_row(self.row(), 124)
        self.assertIsNone(out)
        self.assertEqual(reason, "near_duplicate_candidate_quarantined")

if __name__ == "__main__":
    unittest.main()
