"""Render sealed paper-result CSV files into deterministic LaTeX table fragments.

The renderer never runs evaluation and never opens datasets.  It accepts only the
sealed output of :mod:`assemble_paper_results`, verifies every source hash and the
CSV/summary correspondence, and refuses to overwrite an existing destination.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = "pointer_meter_paper_result_assembly_v1"
OUTPUT_PROTOCOL = "paper_result_latex_tables_v1"
DEFAULT_SOURCE_ROOT = Path(r"C:\pointer_read\paper_final_results_v2")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "paper/submission_mdpi/official/generated_tables"

SOURCE_FILES = {
    "summary": "summary.json",
    "main_table": "main_table.csv",
    "sensitivity_table": "sensitivity_table.csv",
    "component_table": "component_table.csv",
    "paired_group_bootstrap": "paired_group_bootstrap.csv",
}
OUTPUT_FILES = {
    "main_table": "strict-main-table.tex",
    "paired_group_bootstrap": "paired-bootstrap-table.tex",
    "component_table": "component-oof-table.tex",
    "sensitivity_table": "sensitivity-table.tex",
}


class TableRenderError(RuntimeError):
    """Raised when sealed inputs or generated table invariants fail."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TableRenderError(message)


def _reject_constant(value: str) -> Any:
    raise TableRenderError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required artifact is absent: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(
                handle,
                parse_constant=_reject_constant,
                object_pairs_hook=_strict_object,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TableRenderError(f"cannot read JSON artifact {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"required artifact is absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"required artifact is absent: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, f"CSV has no header: {path}")
            require(len(reader.fieldnames) == len(set(reader.fieldnames)), f"duplicate CSV column: {path}")
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise TableRenderError(f"cannot read CSV artifact {path}: {exc}") from exc
    require(bool(rows), f"CSV is empty: {path}")
    return rows


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    require(not isinstance(value, (dict, list)), "nested value cannot be represented in source CSV")
    return str(value)


def _expected_csv_rows(summary: Mapping[str, Any], name: str) -> list[dict[str, str]]:
    key = {
        "main_table": "main_table",
        "sensitivity_table": "sensitivity_table",
        "component_table": "component_table",
        "paired_group_bootstrap": "paired_group_bootstrap",
    }[name]
    raw_rows = summary.get(key)
    require(isinstance(raw_rows, list) and raw_rows, f"summary has no non-empty {key}")
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        require(isinstance(raw, Mapping), f"summary {key} contains a non-object row")
        if name == "paired_group_bootstrap":
            nmae_ci = raw.get("nmae_difference_group_bootstrap_95ci")
            coverage_ci = raw.get("coverage_difference_group_bootstrap_95ci")
            require(isinstance(nmae_ci, list) and len(nmae_ci) == 2, "bootstrap NMAE CI is malformed")
            require(isinstance(coverage_ci, list) and len(coverage_ci) == 2, "bootstrap coverage CI is malformed")
            flat = {k: v for k, v in raw.items() if not isinstance(v, list)}
            flat.update(
                {
                    "nmae_difference_ci95_low": nmae_ci[0],
                    "nmae_difference_ci95_high": nmae_ci[1],
                    "coverage_difference_ci95_low": coverage_ci[0],
                    "coverage_difference_ci95_high": coverage_ci[1],
                }
            )
            raw = flat
        rows.append({str(column): _csv_scalar(value) for column, value in raw.items()})
    return rows


def _assert_csv_matches_summary(
    observed: list[dict[str, str]], expected: list[dict[str, str]], label: str
) -> None:
    require(len(observed) == len(expected), f"{label} row count differs from summary")
    observed_columns = list(observed[0])
    expected_columns: list[str] = []
    for row in expected:
        for column in row:
            if column not in expected_columns:
                expected_columns.append(column)
    require(observed_columns == expected_columns, f"{label} header/order differs from summary")
    normalized = [
        {column: row.get(column, "") for column in expected_columns} for row in expected
    ]
    require(observed == normalized, f"{label} values/order differ from summary")


def validate_source(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    root = root.resolve()
    require(root.is_dir(), f"paper result root is absent: {root}")
    source_paths = {name: root / filename for name, filename in SOURCE_FILES.items()}
    seal_path = root / "seal.json"
    summary = strict_json(source_paths["summary"])
    seal = strict_json(seal_path)
    require(summary.get("protocol") == SOURCE_PROTOCOL, "paper result summary protocol drift")
    require(summary.get("status") == "complete", "paper result summary is incomplete")
    cohort = summary.get("cohort", {})
    require(
        isinstance(cohort, Mapping)
        and int(cohort.get("samples", -1)) == 412
        and int(cohort.get("groups", -1)) == 19,
        "strict paper cohort must be 412 images / 19 groups",
    )
    require(seal.get("protocol") == SOURCE_PROTOCOL, "paper result seal protocol drift")
    require(seal.get("status") == "sealed", "paper result seal is not sealed")
    artifacts = seal.get("artifacts")
    require(isinstance(artifacts, Mapping), "paper result seal has no artifact map")
    require(set(artifacts) == set(SOURCE_FILES), "paper result seal artifact roster drift")
    for name, path in source_paths.items():
        binding = artifacts.get(name)
        require(isinstance(binding, Mapping), f"paper result seal binding is malformed: {name}")
        require(binding.get("sha256") == sha256_file(path), f"paper result hash drift: {name}")
        require(int(binding.get("bytes", -1)) == path.stat().st_size, f"paper result size drift: {name}")
    require(seal.get("bundle_sha256") == canonical_sha256(artifacts), "paper result bundle hash drift")

    csv_rows: dict[str, list[dict[str, str]]] = {}
    for name in SOURCE_FILES:
        if name == "summary":
            continue
        observed = read_csv(source_paths[name])
        _assert_csv_matches_summary(observed, _expected_csv_rows(summary, name), name)
        csv_rows[name] = observed
    require(
        [row.get("method") for row in csv_rows["main_table"]]
        == ["garc", "vdn_official200", "under_pressure_official"],
        "strict main-table method roster/order drift",
    )
    require(
        any(row.get("method") == "enhanced_v5_geometry_head" for row in csv_rows["component_table"]),
        "enhanced V5 component row is absent",
    )
    return summary, csv_rows


_METHOD_NAMES = {
    "garc": "GARC (ours)",
    "vdn_official200": "VDN official-200",
    "under_pressure_official": "Under Pressure (official)",
    "original_transformer": "Original Transformer",
    "enhanced_v5_geometry_head": "Enhanced V5 geometry head",
    "garc_range_fixed_seed_20260720": "GARC range (fixed fold)",
}


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def method_name(value: str) -> str:
    return latex_escape(_METHOD_NAMES.get(value, value))


def number(value: str | float | int | None, digits: int = 6) -> str:
    if value in (None, ""):
        return "--"
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite table value: {value}")
    return f"{parsed:.{digits}f}"


def percent(value: str | float | int | None, digits: int = 2) -> str:
    if value in (None, ""):
        return "--"
    return f"{100.0 * float(value):.{digits}f}\\%"


def _table(caption: str, label: str, columns: str, header: str, rows: Sequence[str]) -> str:
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\caption{" + caption + r"\label{" + label + r"}}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{" + columns + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *(row + r" \\" for row in rows),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_tables(rows: Mapping[str, list[dict[str, str]]]) -> dict[str, str]:
    main_rows = [
        " & ".join(
            [
                method_name(row["method"]),
                latex_escape(row["samples"]),
                latex_escape(row["groups"]),
                percent(row["coverage"]),
                number(row["full_denominator_nmae_failure_penalty_1"]),
                number(row["full_denominator_p95_normalized_error"]),
                number(row["macro_group_nmae"]),
            ]
        )
        for row in rows["main_table"]
    ]
    main = _table(
        "Strict all-component grouped-OOF comparison on the fixed 412-image, "
        "19-group cohort. Failures remain in the denominator with normalized error 1.",
        "tab:garc-strict-main",
        "lrrrrrr",
        "Method & Images & Groups & Coverage & NMAE $\\downarrow$ & P95 $\\downarrow$ & Macro NMAE $\\downarrow$",
        main_rows,
    )

    bootstrap_rows = [
        " & ".join(
            [
                method_name(row["comparator"]),
                number(row["nmae_difference"]),
                f"[{number(row['nmae_difference_ci95_low'])}, {number(row['nmae_difference_ci95_high'])}]",
                number(row.get("garc_relative_nmae_reduction_percent"), 2) + r"\%",
                number(row["coverage_difference"], 4),
                f"[{number(row['coverage_difference_ci95_low'], 4)}, {number(row['coverage_difference_ci95_high'], 4)}]",
            ]
        )
        for row in rows["paired_group_bootstrap"]
    ]
    bootstrap = _table(
        "Paired physical-group bootstrap effects on the fixed 412-image cohort. "
        "Positive NMAE differences favor GARC; coverage differences are GARC minus comparator.",
        "tab:garc-bootstrap",
        "lrrrrr",
        "Comparator & $\\Delta$NMAE & 95\\% CI & Reduction & $\\Delta$Coverage & 95\\% CI",
        bootstrap_rows,
    )

    component_rows = [
        " & ".join(
            [
                method_name(row["method"]),
                latex_escape(row["samples"]),
                latex_escape(row["groups"]),
                percent(row["coverage"]),
                number(row["full_denominator_nmae"]),
                number(row["p95_absolute_progress_error"]),
            ]
        )
        for row in rows["component_table"]
    ]
    component = _table(
        "Authenticated component-level OOF evidence. This table is not the "
        "strict 412-image end-to-end comparison.",
        "tab:v5-component-oof",
        "lrrrrr",
        "Component & Images & Groups & Coverage & NMAE $\\downarrow$ & P95 $\\downarrow$",
        component_rows,
    )

    sensitivity_rows = [
        " & ".join(
            [
                latex_escape(row["family"]),
                method_name(row["method"]),
                latex_escape(row["samples"]),
                latex_escape(row["groups"]),
                percent(row.get("coverage")),
                latex_escape(row["metric_name"]),
                number(row["metric_value"]),
            ]
        )
        for row in rows["sensitivity_table"]
    ]
    sensitivity = _table(
        "Pre-specified sensitivity evidence excluded from the strict main table.",
        "tab:garc-sensitivity",
        "llllrrr",
        "Family & Method & Images & Groups & Coverage & Metric & Value",
        sensitivity_rows,
    )
    return {
        "main_table": main,
        "paired_group_bootstrap": bootstrap,
        "component_table": component,
        "sensitivity_table": sensitivity,
    }


def render(source_root: Path, output_root: Path) -> Path:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    require(not output_root.exists(), f"refusing to overwrite existing table output: {output_root}")
    summary, rows = validate_source(source_root)
    fragments = render_tables(rows)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.tmp.{os.getpid()}")
    require(not staging.exists(), f"staging output already exists: {staging}")
    staging.mkdir()
    try:
        for name, filename in OUTPUT_FILES.items():
            (staging / filename).write_text(fragments[name], encoding="utf-8", newline="\n")
        artifacts = {
            name: {
                "path": filename,
                "sha256": sha256_file(staging / filename),
                "bytes": (staging / filename).stat().st_size,
            }
            for name, filename in OUTPUT_FILES.items()
        }
        manifest = {
            "schema_version": 1,
            "protocol": OUTPUT_PROTOCOL,
            "status": "complete",
            "source": {
                "path": str(source_root),
                "summary_sha256": sha256_file(source_root / "summary.json"),
                "seal_sha256": sha256_file(source_root / "seal.json"),
                "cohort": summary["cohort"],
            },
            "artifacts": artifacts,
            "audit": {
                "source_seal_verified": True,
                "csv_summary_correspondence_verified": True,
                "datasets_opened": 0,
                "inference_started": False,
                "training_started": False,
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        sealed_artifacts = {
            **artifacts,
            "manifest": {
                "path": "manifest.json",
                "sha256": sha256_file(manifest_path),
                "bytes": manifest_path.stat().st_size,
            },
        }
        seal = {
            "schema_version": 1,
            "protocol": OUTPUT_PROTOCOL,
            "status": "sealed",
            "artifacts": sealed_artifacts,
            "bundle_sha256": canonical_sha256(sealed_artifacts),
        }
        (staging / "seal.json").write_text(
            json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, output_root)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return output_root / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        summary, rows = validate_source(args.source_root)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "protocol": summary["protocol"],
                    "cohort": summary["cohort"],
                    "rows": {name: len(value) for name, value in rows.items()},
                    "artifacts_written": 0,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    output = render(args.source_root, args.output_root)
    print(json.dumps({"status": "complete", "manifest": str(output)}, indent=2))


if __name__ == "__main__":
    main()
