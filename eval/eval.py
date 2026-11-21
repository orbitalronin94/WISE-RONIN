"""M1 evaluation shell: verify the frozen holdout can be loaded."""
import argparse
import json
from pathlib import Path

def load_holdout(path: Path) -> list[dict]:
    """Load holdout records without modifying or using them for training."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--holdout", default="data/holdout_500.jsonl")
    args = parser.parse_args()
    rows = load_holdout(Path(args.holdout))
    print(json.dumps({"status": "M1_SKELETON_ONLY", "holdout_records": len(rows),
                      "metrics": "not implemented in M1"}, ensure_ascii=False))

if __name__ == "__main__":
    main()
