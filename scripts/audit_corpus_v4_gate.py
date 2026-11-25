#!/usr/bin/env python3
"""Audit frozen WISE-RONIN JSONL and gate new corpus admissions against LUMEN attractors."""
from __future__ import annotations
import argparse, collections, hashlib, json, re, statistics, unicodedata
from pathlib import Path

DEFAULT_BLOCKLIST = {
    "schema": "lumen-attractor-blocklist-v1",
    "terms": ["contexto documental", "democracia", "música", "deformación", "latinoamericana", "estados miembros", "unión europea"]
}

def norm(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()

def compile_terms(terms: list[str]) -> re.Pattern:
    escaped = [re.escape(norm(term)) for term in terms]
    return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.UNICODE)

def model_text(row: dict) -> str:
    if "text" in row:
        return str(row.get("text") or "")
    return "\n".join(str(row.get(k) or "") for k in ("prompt", "completion", "instruction", "input", "output", "answer", "response"))

def is_personality(row: dict) -> bool:
    source = str(row.get("source", "")).casefold()
    task = str(row.get("task", row.get("task_type", ""))).casefold()
    return any(key in source for key in ("persona", "m-personas", "stratasynth")) or "persona" in task or "personality" in task

class ChunkWriter:
    """Write accepted JSONL into immutable chunks capped at 100 MiB by default."""
    def __init__(self, directory: str, max_bytes: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.index, self.handle, self.size, self.rows = 0, None, 0, 0
        self.manifest = []

    def _open(self):
        self.index += 1
        self.path = self.directory / f"v4-block-{self.index:04d}.jsonl"
        self.handle = self.path.open("wb")
        self.size = self.rows = 0
        self.sha = hashlib.sha256()

    def write(self, line: bytes):
        if len(line) > self.max_bytes:
            raise ValueError(f"Single row exceeds block limit ({len(line)} > {self.max_bytes}); split at source-document level first")
        if self.handle is None or (self.size and self.size + len(line) > self.max_bytes):
            self.close_current()
            self._open()
        self.handle.write(line)
        self.sha.update(line)
        self.size += len(line)
        self.rows += 1

    def close_current(self):
        if self.handle is not None:
            self.handle.close()
            self.manifest.append({"file": self.path.name, "bytes": self.size, "rows": self.rows, "sha256": self.sha.hexdigest()})
            self.handle = None

    def close(self):
        self.close_current()
        (self.directory / "blocks-manifest.json").write_text(
            json.dumps({"schema":"wise-ronin-v4-blocks-v1","max_bytes":self.max_bytes,"blocks":self.manifest}, ensure_ascii=False, indent=2)+"\n",
            encoding="utf-8"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL to audit; for admission this must be the incoming delta")
    ap.add_argument("--report", required=True)
    ap.add_argument("--blocklist")
    ap.add_argument("--accepted-output", help="Optional filtered copy; input is never modified")
    ap.add_argument("--chunks-dir", help="Optional 100 MiB block output for accepted rows")
    ap.add_argument("--chunk-mib", type=int, default=100)
    ap.add_argument("--base-rows", type=int, default=0)
    ap.add_argument("--base-personality-rows", type=int, default=0)
    ap.add_argument("--target-multiplier", type=int, default=10)
    ap.add_argument("--minimum-personality-share", type=float, default=0.20)
    ap.add_argument("--require-provenance", action="store_true",
                    help="For new admissions, require source, source_revision, license, language and task")
    args = ap.parse_args()

    block = json.loads(Path(args.blocklist).read_text(encoding="utf-8")) if args.blocklist else DEFAULT_BLOCKLIST
    pattern = compile_terms(block["terms"])
    counters = {name: collections.Counter() for name in (
        "sources","tasks","licenses","languages","blocked_sources","blocked_terms","provenance_rejects"
    )}
    total = blocked = accepted = personality = accepted_personality = provenance_rejected = 0
    chars, lengths = 0, []
    input_sha = hashlib.sha256()
    accepted_file = Path(args.accepted_output).open("wb") if args.accepted_output else None
    chunks = ChunkWriter(args.chunks_dir, args.chunk_mib * 1024 * 1024) if args.chunks_dir else None

    try:
        with open(args.input, "rb") as src:
            for line_no, raw in enumerate(src, 1):
                input_sha.update(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON at line {line_no}: {exc}")
                total += 1
                source = str(row.get("source", "unknown"))
                task = str(row.get("task", row.get("task_type", "UNSPECIFIED")))
                counters["sources"][source] += 1
                counters["tasks"][task] += 1
                counters["licenses"][str(row.get("license", "UNSPECIFIED"))] += 1
                counters["languages"][str(row.get("language", "UNSPECIFIED"))] += 1
                persona = is_personality(row)
                personality += int(persona)

                if args.require_provenance:
                    required = ("source", "source_revision", "license", "language")
                    missing = [key for key in required if not str(row.get(key, "")).strip()]
                    allowed = {"apache-2.0","mit","cc-by-4.0","cc0-1.0","public-domain/per-document-rights-verified"}
                    license_value = str(row.get("license", "")).casefold()
                    if missing or license_value not in allowed or not task.strip() or task == "UNSPECIFIED":
                        provenance_rejected += 1
                        reason = "missing:" + ",".join(missing) if missing else "license_or_task_not_allowlisted"
                        counters["provenance_rejects"][reason] += 1
                        continue

                text = model_text(row)
                chars += len(text)
                lengths.append(len(text))
                matches = sorted(set(m.group(0) for m in pattern.finditer(norm(text))))
                if matches:
                    blocked += 1
                    counters["blocked_sources"][source] += 1
                    counters["blocked_terms"].update(matches)
                    continue

                accepted += 1
                accepted_personality += int(persona)
                encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                if accepted_file:
                    accepted_file.write(encoded)
                if chunks:
                    chunks.write(encoded)
    finally:
        if accepted_file:
            accepted_file.close()
        if chunks:
            chunks.close()

    final_rows = args.base_rows + accepted
    final_personality = args.base_personality_rows + accepted_personality
    share = final_personality / final_rows if final_rows else 0.0
    target_rows = args.base_rows * args.target_multiplier if args.base_rows else None
    report = {
        "schema":"wise-ronin-corpus-v4-gate-audit-v1",
        "input_file":Path(args.input).name,
        "input_sha256":input_sha.hexdigest(),
        "rows_seen":total,
        "rows_blocked_by_lumen_attractors":blocked,
        "rows_rejected_for_provenance":provenance_rejected,
        "rows_accepted_by_gate":accepted,
        "blocked_share":round(blocked/total,6) if total else 0,
        "approx_characters":chars,
        "average_characters_per_row":round(chars/total,2) if total else 0,
        "median_characters_per_row":statistics.median(lengths) if lengths else 0,
        "source_rows":dict(counters["sources"].most_common()),
        "task_rows":dict(counters["tasks"].most_common()),
        "license_rows":dict(counters["licenses"].most_common()),
        "language_rows":dict(counters["languages"].most_common()),
        "blocked_rows_by_source":dict(counters["blocked_sources"].most_common()),
        "blocked_rows_by_term":dict(counters["blocked_terms"].most_common()),
        "provenance_reject_reasons":dict(counters["provenance_rejects"].most_common()),
        "personality":{
            "input_personality_rows":personality,
            "accepted_personality_rows":accepted_personality,
            "final_rows_if_accepted":final_rows,
            "final_personality_rows_if_accepted":final_personality,
            "final_share_if_accepted":round(share,6),
            "minimum_share":args.minimum_personality_share,
            "preferred_share":0.25,
            "share_gate_passed":share >= args.minimum_personality_share
        },
        "growth":{
            "target_multiplier":args.target_multiplier,
            "target_rows":target_rows,
            "final_rows_if_accepted":final_rows,
            "row_target_passed":final_rows >= target_rows if target_rows else None,
            "additional_rows_needed_to_target":max(0,target_rows-args.base_rows) if target_rows else None,
            "additional_personality_rows_needed_at_25pct":max(0,round(target_rows*0.25)-args.base_personality_rows) if target_rows else None
        },
        "blocklist":{"schema":block.get("schema"),"terms":block["terms"]},
        "chunking":{"max_uncompressed_bytes":args.chunk_mib*1024*1024,"chunks_manifest":"blocks-manifest.json" if chunks else None},
        "training_authorized":False,
        "note":"Admission audit only. Never authorizes training or modifies the frozen base corpus."
    }
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
