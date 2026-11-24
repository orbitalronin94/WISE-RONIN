#!/usr/bin/env python3
import importlib.util
import unittest

spec = importlib.util.spec_from_file_location("admit_stratasynth_personas", "scripts/admit_stratasynth_personas.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class StrataSynthPersonaPolicyTests(unittest.TestCase):
    def row(self):
        return {
            "id": "es-00003", "country": "ES", "language": "es",
            "persona": "Un profesional de la agricultura que describe su trayectoria, su entorno y su relación con el trabajo rural de manera detallada.",
            "daily_routine": "Cada mañana organiza las tareas de la finca, revisa los cultivos y coordina las actividades con el equipo antes de continuar con el trabajo del día.",
            "speaking_style": "Habla con frases pausadas, vocabulario concreto y un tono cercano, sin recurrir a tecnicismos innecesarios.",
            "age": 63, "gender": "male", "personality": {"big_five": {"O": 0.6}}
        }

    def test_accepts_only_narrative_fields(self):
        out, reason = module.normalize_row(self.row(), 0)
        self.assertIsNone(reason)
        self.assertEqual(out["task_type"], "synthetic_persona_voice")
        self.assertNotIn("age", out)
        self.assertNotIn("gender", out)
        self.assertNotIn("personality", out)
        self.assertNotIn("63", out["text"])

    def test_rejects_non_spanish_and_privacy_flags(self):
        row = self.row()
        row["country"] = "MX"
        self.assertEqual(module.normalize_row(row, 0)[1], "not_spanish_spain")
        row = self.row()
        row["speaking_style"] += " Contacto ana@example.org."
        self.assertEqual(module.normalize_row(row, 0)[1], "privacy_flag:possible_email")

if __name__ == "__main__":
    unittest.main()
