import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Compare two batch-inference CSV files.")
parser.add_argument("old", type=Path, help="baseline results.csv")
parser.add_argument("new", type=Path, help="candidate results.csv")
parser.add_argument(
    "--output",
    type=Path,
    default=Path("comparison_summary.json"),
)
args = parser.parse_args()
old_path = args.old
new_path = args.new
out_path = args.output


def load_csv(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            result = row.get("result")
            end_num = row.get("end_num")
            rows[row["file_name"]] = {
                "status": str(row.get("status", "")).strip().lower() == "true",
                "message": row.get("message"),
                "result": None if result in (None, "") else float(result),
                "end_num": None if end_num in (None, "") else float(end_num),
            }
    return rows

old_rows = load_csv(old_path)
new_rows = load_csv(new_path)
all_names = sorted(set(old_rows) | set(new_rows))

changed = []
status_changed = []
improved = []
regressed = []
unchanged = []

for name in all_names:
    old = old_rows.get(name)
    new = new_rows.get(name)
    record = {
        "file_name": name,
        "old": old,
        "new": new,
    }
    if old and new:
        old_status = old["status"]
        new_status = new["status"]
        old_result = old["result"]
        new_result = new["result"]
        result_delta = None
        if old_result is not None and new_result is not None:
            result_delta = new_result - old_result
        record["result_delta"] = result_delta
        record["status_changed"] = old_status != new_status
        record["result_changed"] = result_delta is not None and abs(result_delta) > 1e-12

        if old_status != new_status:
            status_changed.append(record)
            if (not old_status) and new_status:
                improved.append(record)
            elif old_status and (not new_status):
                regressed.append(record)
        elif record["result_changed"]:
            changed.append(record)
        else:
            unchanged.append(record)
    else:
        changed.append(record)

summary = {
    "old_success": sum(1 for v in old_rows.values() if v["status"]),
    "old_failed": sum(1 for v in old_rows.values() if not v["status"]),
    "new_success": sum(1 for v in new_rows.values() if v["status"]),
    "new_failed": sum(1 for v in new_rows.values() if not v["status"]),
    "status_changed": status_changed,
    "improved": improved,
    "regressed": regressed,
    "result_changed": changed,
    "unchanged_count": len(unchanged),
}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
