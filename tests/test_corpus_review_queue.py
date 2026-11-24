import json, subprocess, sys, tempfile, unittest
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"build_corpus_review_queue.py"
class ReviewQueueTests(unittest.TestCase):
 def test_metadata_only_and_field_aware_pii(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); source=root/"in.jsonl"; output=root/"index.jsonl"
   rows=[
    {"id":"12345678Z","question":"¿Qué derechos tiene una persona refugiada en España?","answer":"Debe recibir información clara y asistencia jurídica."},
    {"id":"555 123 456","question":"¿Cómo se tramita una solicitud de asilo?","answer":"La autoridad competente debe resolver conforme a la normativa aplicable."},
    {"question":"Contacte con ana@example.org para más información sobre el procedimiento."}]
   source.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
   subprocess.run([sys.executable,str(SCRIPT),"--input",str(source),"--output",str(output),
    "--source","example/dataset","--revision","deadbeef","--text-fields","question,answer"],
    check=True,capture_output=True,text=True)
   records=[json.loads(x) for x in output.read_text(encoding="utf-8").splitlines()]
   self.assertEqual(len(records),3); self.assertNotIn("question",records[0]); self.assertNotIn("answer",records[0])
   self.assertFalse(records[0]["raw_text_exported"]); self.assertEqual(records[0]["privacy_flags"],[])
   self.assertEqual(records[1]["privacy_flags"],[]); self.assertIn("possible_email_in_text",records[2]["privacy_flags"])
   self.assertEqual(records[0]["source_revision"],"deadbeef"); self.assertEqual(records[0]["task_type"],"question_answer")
if __name__=="__main__": unittest.main()
