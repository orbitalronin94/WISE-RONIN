import json, subprocess, sys, tempfile, unittest
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"build_corpus_review_queue.py"
class ReviewQueueTests(unittest.TestCase):
 def run_index(self,root,rows,extra=None):
  source=root/"in.jsonl"; output=root/"index.jsonl"
  source.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
  args=[sys.executable,str(SCRIPT),"--input",str(source),"--output",str(output),
   "--source","example/dataset","--revision","deadbeef","--text-fields","question,answer"]
  if extra: args.extend(extra)
  proc=subprocess.run(args,capture_output=True,text=True)
  self.assertEqual(proc.returncode,0,proc.stderr)
  return [json.loads(x) for x in output.read_text(encoding="utf-8").splitlines()]
 def test_metadata_only_and_field_aware_pii(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   records=self.run_index(root,[
    {"id":"12345678Z","question":"¿Qué derechos tiene una persona refugiada en España?","answer":"Debe recibir información clara y asistencia jurídica."},
    {"id":"555 123 456","question":"¿Cómo se tramita una solicitud de asilo?","answer":"La autoridad competente debe resolver conforme a la normativa aplicable."},
    {"question":"Contacte con ana@example.org para más información sobre el procedimiento."}])
   self.assertEqual(len(records),3); self.assertNotIn("question",records[0]); self.assertNotIn("answer",records[0])
   self.assertFalse(records[0]["raw_text_exported"]); self.assertEqual(records[0]["privacy_flags"],[])
   self.assertEqual(records[1]["privacy_flags"],[]); self.assertIn("possible_email_in_text",records[2]["privacy_flags"])
   self.assertEqual(records[0]["source_revision"],"deadbeef"); self.assertEqual(records[0]["task_type"],"question_answer")
 def test_v31_exact_duplicate_is_flagged(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); base=root/"base.jsonl"
   base.write_text(json.dumps({"text":"Este texto ya forma parte del corpus base congelado."},ensure_ascii=False)+"\n",encoding="utf-8")
   source=root/"in.jsonl"; output=root/"index.jsonl"
   source.write_text(json.dumps({"question":"Este texto ya forma parte del corpus base congelado."},ensure_ascii=False)+"\n",encoding="utf-8")
   proc=subprocess.run([sys.executable,str(SCRIPT),"--input",str(source),"--output",str(output),
    "--source","example/dataset","--revision","deadbeef","--text-fields","question","--base-jsonl",str(base)],
    capture_output=True,text=True)
   self.assertEqual(proc.returncode,0,proc.stderr)
   record=json.loads(output.read_text(encoding="utf-8").splitlines()[0])
   self.assertIn("exact_duplicate_v3_1",record["privacy_flags"])
 def test_chat_messages_are_hashed_without_exporting_content(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   records=self.run_index(root,[{"id":"sample-1","messages":[
    {"role":"system","content":"Responde en español claro."},
    {"role":"user","content":"¿Qué significa una licencia abierta?"},
    {"role":"assistant","content":"Permite reutilizar una obra bajo condiciones."}]}],
    ["--messages-field","messages"])
   self.assertEqual(records[0]["task_type"],"chat_messages")
   self.assertEqual(records[0]["text_fields"],["messages[system]","messages[user]","messages[assistant]"])
   self.assertNotIn("content",records[0]); self.assertNotIn("messages",records[0])
if __name__=="__main__": unittest.main()
