import hashlib
import unittest

from scripts.build_literal_qa_queue import iter_sqac_rows
from scripts.validate_literal_qa_dataset import validate


class LiteralQuestionDatasetTests(unittest.TestCase):
    def test_sqac_question_is_copied_exactly_and_answer_offset_is_preserved(self):
        payload = {"data": [{"title": "Artículo", "paragraphs": [{
            "context": "La respuesta está aquí.",
            "qas": [{"id": "q-1", "question": "¿Qué está aquí?", "answers": {
                "text": ["La respuesta"], "answer_start": [0]
            }}]
        }]}]}
        row = list(iter_sqac_rows(payload, "train"))[0]
        self.assertEqual(row["question_verbatim"], "¿Qué está aquí?")
        self.assertEqual(row["source_question_sha256"], hashlib.sha256("¿Qué está aquí?".encode("utf-8")).hexdigest())
        self.assertEqual(row["reference_answers_verbatim"], ["La respuesta"])
        self.assertEqual(row["context_verbatim"][0:12], "La respuesta")
        self.assertEqual(row["ronin_answer_manual"], "")

    def test_dev_and_test_are_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_sqac_rows({"data": []}, "dev"))

    def test_manual_row_must_be_reviewed_and_not_copy_reference(self):
        row = {
            "source_dataset": "PlanTL-GOB-ES/SQAC", "source_revision": "abc",
            "source_split": "train", "source_id": "q-1", "title_verbatim": "Artículo",
            "question_verbatim": "¿Qué está aquí?", "source_question_sha256": hashlib.sha256("¿Qué está aquí?".encode("utf-8")).hexdigest(), "context_verbatim": "La respuesta está aquí.",
            "reference_answers_verbatim": ["La respuesta"], "answer_starts": [0],
            "ronin_answer_manual": "Está aquí la respuesta.", "author_status": "manually_reviewed",
        }
        self.assertEqual(len(validate([row])), 1)
        bad = dict(row, ronin_answer_manual="La respuesta")
        with self.assertRaises(ValueError):
            validate([bad])

    def test_source_queue_detects_question_edits(self):
        row = {
            "source_dataset": "PlanTL-GOB-ES/SQAC", "source_revision": "abc",
            "source_split": "train", "source_id": "q-1", "title_verbatim": "Artículo",
            "question_verbatim": "¿Qué está aquí?", "source_question_sha256": hashlib.sha256("¿Qué está aquí?".encode("utf-8")).hexdigest(),
            "context_verbatim": "La respuesta está aquí.", "reference_answers_verbatim": ["La respuesta"],
            "answer_starts": [0], "ronin_answer_manual": "Está aquí la respuesta.", "author_status": "manually_reviewed",
        }
        source = dict(row, ronin_answer_manual="", author_status="pending_manual_authorship")
        self.assertEqual(len(validate([row], source_rows=[source])), 1)
        changed = dict(row, question_verbatim="¿Qué hay aquí?")
        changed["source_question_sha256"] = hashlib.sha256(changed["question_verbatim"].encode("utf-8")).hexdigest()
        with self.assertRaises(ValueError):
            validate([changed], source_rows=[source])

    def test_multiple_reference_spans_stay_one_question_row(self):
        payload = {"data": [{"title": "Artículo", "paragraphs": [{
            "context": "La respuesta está aquí.",
            "qas": [{"id": "q-2", "question": "¿Qué está aquí?", "answers": {
                "text": ["La respuesta", "respuesta"], "answer_start": [0, 3]
            }}]
        }]}]}
        rows = list(iter_sqac_rows(payload, "train"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reference_answers_verbatim"], ["La respuesta", "respuesta"])

    def test_non_train_manual_rows_are_rejected(self):
        row = {
            "source_dataset": "PlanTL-GOB-ES/SQAC", "source_revision": "abc",
            "source_split": "validation", "source_id": "q-1", "title_verbatim": "Artículo",
            "question_verbatim": "¿Qué está aquí?", "source_question_sha256": hashlib.sha256("¿Qué está aquí?".encode("utf-8")).hexdigest(), "context_verbatim": "La respuesta está aquí.",
            "reference_answers_verbatim": ["La respuesta"], "answer_starts": [0],
            "ronin_answer_manual": "Está aquí la respuesta.", "author_status": "manually_reviewed",
        }
        with self.assertRaises(ValueError):
            validate([row])


if __name__ == "__main__":
    unittest.main()
