import hashlib
import unittest

from scripts.build_literal_qa_queue import iter_sqac_rows
from scripts.validate_literal_qa_dataset import validate


def source_fixture():
    payload = {"data": [{"title": "Artículo", "paragraphs": [{
        "context": "La respuesta está aquí.",
        "qas": [{"id": "q-1", "question": "¿Qué está aquí?", "answers": {
            "text": ["La respuesta"], "answer_start": [0]
        }}]
    }]}]}
    return list(iter_sqac_rows(payload, "train"))[0]


def manual_fixture(source=None):
    source = source or source_fixture()
    return {
        "source_dataset": source["source_dataset"],
        "source_revision": source["source_revision"],
        "source_split": source["source_split"],
        "source_id": source["source_id"],
        "title_verbatim": source["title_verbatim"],
        "question_verbatim": source["question_verbatim"],
        "source_question_sha256": source["source_question_sha256"],
        "ronin_answer_manual": "Está aquí la respuesta.",
        "author_status": "manually_reviewed",
    }


class LiteralQuestionDatasetTests(unittest.TestCase):
    def test_sqac_question_is_copied_exactly_and_answer_offset_is_preserved(self):
        row = source_fixture()
        self.assertEqual(row["question_verbatim"], "¿Qué está aquí?")
        self.assertEqual(row["source_question_sha256"], hashlib.sha256("¿Qué está aquí?".encode("utf-8")).hexdigest())
        self.assertEqual(row["reference_answers_verbatim"], ["La respuesta"])
        self.assertEqual(row["context_verbatim"][0:12], "La respuesta")
        self.assertEqual(row["ronin_answer_manual"], "")

    def test_dev_and_test_are_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_sqac_rows({"data": []}, "dev"))

    def test_manual_row_must_be_reviewed_and_not_copy_reference(self):
        source = source_fixture()
        row = manual_fixture(source)
        self.assertEqual(len(validate([row], [source])), 1)
        bad = dict(row, ronin_answer_manual="La respuesta")
        with self.assertRaises(ValueError):
            validate([bad], [source])
        pending = dict(row, author_status="pending_manual_authorship")
        with self.assertRaises(ValueError):
            validate([pending], [source])

    def test_source_queue_detects_question_edits(self):
        source = source_fixture()
        row = manual_fixture(source)
        changed = dict(row, question_verbatim="¿Qué hay aquí?")
        changed["source_question_sha256"] = hashlib.sha256(changed["question_verbatim"].encode("utf-8")).hexdigest()
        with self.assertRaises(ValueError):
            validate([changed], [source])

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

    def test_sqac_list_of_answer_objects_is_supported(self):
        payload = {"data": [{"title": "Artículo", "paragraphs": [{
            "context": "La respuesta está aquí.",
            "qas": [{"id": "q-list", "question": "¿Qué está aquí?", "answers": [
                {"text": "La respuesta", "answer_start": 0}
            ]}]
        }]}]}
        row = list(iter_sqac_rows(payload, "train"))[0]
        self.assertEqual(row["reference_answers_verbatim"], ["La respuesta"])
        self.assertEqual(row["answer_starts"], [0])

    def test_non_train_manual_rows_are_rejected(self):
        source = dict(source_fixture(), source_split="validation")
        row = manual_fixture(source)
        with self.assertRaises(ValueError):
            validate([row], [source])


if __name__ == "__main__":
    unittest.main()
