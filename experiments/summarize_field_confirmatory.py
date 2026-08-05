"""Verify and summarize the frozen one-shot field confirmatory evaluation.

This command never fits a model, changes a threshold, or chooses a method.  It
reports all frozen-method and comparator results, including unfavorable ones.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.summarize_field_development import (
    _finite,
    compute_metrics,
    paired_physical_group_bootstrap,
    sample_ids_sha256,
    sha256_file,
)


FREEZE_PROTOCOL = "field_confirmatory_one_shot_freeze_v1"
SUMMARY_PROTOCOL = "field_confirmatory_one_shot_summary_v1"
SPLIT_PROTOCOL = "group_disjoint_field_development_confirmatory_split_v1"
SPLIT = "field_confirmatory"
METHOD_ORDER = (
    "pepd_fadr",
    "base_mask",
    "pepd_vector",
    "reference_conditioned_vector",
    "original_transformer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def rows_by_id(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError(f"{label} row {index} has an empty/duplicate sample_id")
        result[sample_id] = row
    return result


def same_number(left: Any, right: Any) -> bool:
    a = _finite(left)
    b = _finite(right)
    return a is not None and b is not None and math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


def flat_prediction(row: Mapping[str, Any]) -> float | None:
    if row.get("status", True) is False:
        return None
    return _finite(row.get("prediction"))


def base_prediction(row: Mapping[str, Any]) -> float | None:
    predictions = row.get("predictions")
    return _finite(predictions.get("ours")) if isinstance(predictions, Mapping) else None


def transformer_prediction(row: Mapping[str, Any]) -> float | None:
    methods = row.get("methods")
    transformer = methods.get("transformer") if isinstance(methods, Mapping) else None
    if not isinstance(transformer, Mapping) or transformer.get("status") is False:
        return None
    return _finite(transformer.get("prediction"))


def bind(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def validate_manifest(
    manifest: Path, freeze: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = freeze.get("dataset")
    if not isinstance(identity, Mapping) or identity.get("split") != SPLIT:
        raise ValueError("freeze lacks the confirmatory dataset identity")
    protocol_path = Path(str(identity["protocol_path"])).resolve()
    if manifest != Path(str(identity["path"])).resolve():
        raise ValueError("manifest path differs from the frozen identity")
    if sha256_file(manifest) != identity.get("sha256"):
        raise ValueError("manifest hash differs from the frozen identity")
    if sha256_file(protocol_path) != identity.get("protocol_sha256"):
        raise ValueError("manifest protocol hash differs from the frozen identity")
    protocol = read_json(protocol_path)
    if (
        protocol.get("protocol") != SPLIT_PROTOCOL
        or protocol.get("split") != SPLIT
        or protocol.get("confirmatory_sealed") is not True
        or protocol.get("group_disjoint") is not True
        or protocol.get("assignment_uses_model_predictions") is not False
        or protocol.get("assignment_uses_ground_truth_reading") is not False
        or protocol.get("manifest_sha256") != identity.get("sha256")
    ):
        raise ValueError("manifest sidecar is not the frozen confirmatory split")
    rows = read_jsonl(manifest)
    mapping = rows_by_id(rows, "manifest")
    groups = sorted({str(row.get("group_id") or "") for row in rows})
    if "" in groups:
        raise ValueError("manifest contains an empty group_id")
    if (
        len(rows) != int(identity["rows"])
        or len(groups) != int(identity["groups"])
        or groups != sorted(identity["group_ids"])
        or sample_ids_sha256(mapping) != identity["sample_ids_sha256"]
    ):
        raise ValueError("manifest rows/groups/sample IDs differ from the freeze")
    for index, row in enumerate(rows, 1):
        if row.get("split") != SPLIT:
            raise ValueError(f"manifest row {index} is not confirmatory")
        if (
            _finite(row.get("ground_truth")) is None
            or _finite(row.get("scale_start")) is None
            or _finite(row.get("scale_end")) is None
            or abs(float(row["scale_end"]) - float(row["scale_start"])) <= 1e-12
        ):
            raise ValueError(f"manifest row {index} has an invalid label/scale")
    return rows, {
        "samples": len(rows),
        "physical_groups": len(groups),
        "sample_ids_sha256": sample_ids_sha256(mapping),
        "group_ids": groups,
    }


def main() -> None:
    args = parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in vars(args).items()
        if name != "output"
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"formal summary is immutable: {output}")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    freeze = read_json(paths["freeze"])
    if (
        freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("status") != "frozen_authorized_not_started"
        or (freeze.get("authorization") or {}).get(
            "user_explicitly_authorized_one_shot_field_blind_test"
        )
        is not True
        or (freeze.get("authorization") or {}).get(
            "no_tuning_or_method_changes_after_result"
        )
        is not True
    ):
        raise ValueError("one-shot confirmatory freeze is invalid")

    manifest_rows, identity = validate_manifest(paths["manifest"], freeze)
    manifest_by_id = rows_by_id(manifest_rows, "manifest")
    prediction_rows = {
        name: read_jsonl(paths[name]) for name in ("raw", "base", "vector", "final")
    }
    mappings = {name: rows_by_id(rows, name) for name, rows in prediction_rows.items()}
    expected_ids = set(manifest_by_id)
    for name, mapping in mappings.items():
        if set(mapping) != expected_ids:
            raise ValueError(f"{name} sample IDs differ from the manifest")
        for sample_id, row in mapping.items():
            source = manifest_by_id[sample_id]
            if (
                row.get("split") != SPLIT
                or row.get("group_id") != source.get("group_id")
                or row.get("dataset") != source.get("dataset")
                or not same_number(row.get("ground_truth"), source.get("ground_truth"))
                or not same_number(row.get("scale_start"), source.get("scale_start"))
                or not same_number(row.get("scale_end"), source.get("scale_end"))
            ):
                raise ValueError(f"{name}[{sample_id}] identity/label differs")

    ordered_ids = [str(row["sample_id"]) for row in manifest_rows]
    predictions: dict[str, list[float | None]] = {
        "pepd_fadr": [flat_prediction(mappings["final"][key]) for key in ordered_ids],
        "base_mask": [base_prediction(mappings["base"][key]) for key in ordered_ids],
        "pepd_vector": [flat_prediction(mappings["vector"][key]) for key in ordered_ids],
        "reference_conditioned_vector": [
            _finite(mappings["final"][key].get("reference_conditioned_prediction"))
            for key in ordered_ids
        ],
        "original_transformer": [
            transformer_prediction(mappings["raw"][key]) for key in ordered_ids
        ],
    }
    evaluation = freeze["evaluation"]
    failure_penalty = float(evaluation["failure_penalty_nmae"])
    bootstrap = evaluation["paired_bootstrap"]
    metrics: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        metrics[method], errors[method], _ = compute_metrics(
            manifest_rows,
            predictions[method],
            failure_penalty=failure_penalty,
        )
    groups = np.asarray([str(row["group_id"]) for row in manifest_rows], dtype=object)
    paired = {}
    for comparator in (
        "original_transformer",
        "base_mask",
        "pepd_vector",
        "reference_conditioned_vector",
    ):
        paired[f"pepd_fadr_vs_{comparator}"] = {
            "candidate": "pepd_fadr",
            "comparator": comparator,
            **paired_physical_group_bootstrap(
                errors["pepd_fadr"],
                errors[comparator],
                groups,
                iterations=int(bootstrap["iterations"]),
                seed=int(bootstrap["seed"]),
            ),
        }

    final_nmae = metrics["pepd_fadr"]["full_denominator_nmae"]
    transformer_nmae = metrics["original_transformer"]["full_denominator_nmae"]
    base_nmae = metrics["base_mask"]["full_denominator_nmae"]
    summary = {
        "schema_version": 1,
        "protocol": SUMMARY_PROTOCOL,
        "status": "complete",
        "scope": {
            "split": SPLIT,
            "one_shot": True,
            "method_or_threshold_selection_performed": False,
            "result_based_retry_performed": False,
            "official200_vdn_included": False,
            "official200_vdn_reason": "independent frozen fair evaluation is tracked separately",
        },
        "identity": identity,
        "freeze": bind(paths["freeze"]),
        "inputs": {name: bind(path) for name, path in sorted(paths.items())},
        "predeclared_statistics": {
            "failure_penalty_nmae": failure_penalty,
            "bootstrap_iterations": int(bootstrap["iterations"]),
            "bootstrap_seed": int(bootstrap["seed"]),
            "bootstrap_unit": "physical meter group_id",
        },
        "methods": metrics,
        "paired_physical_group_bootstrap": paired,
        "headline": {
            "pepd_fadr_nmae": final_nmae,
            "original_transformer_nmae": transformer_nmae,
            "base_mask_nmae": base_nmae,
            "relative_nmae_reduction_vs_original_transformer": (
                (transformer_nmae - final_nmae) / transformer_nmae
                if transformer_nmae > 0
                else None
            ),
            "relative_nmae_reduction_vs_base_mask": (
                (base_nmae - final_nmae) / base_nmae if base_nmae > 0 else None
            ),
        },
        "interpretation_boundary": [
            "All metrics use the complete frozen confirmatory denominator; failed predictions receive NMAE=1.0.",
            "No confirmatory result is used to alter the method, seed, feature set, threshold, or preprocessing.",
            "The support-only legacy VDN cache supplies frozen meter/reference geometry and is not reported as a comparator.",
            "The official200 VDN comparator remains a separately frozen fair-evaluation task.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "samples": identity["samples"],
                "groups": identity["physical_groups"],
                "pepd_fadr_nmae": final_nmae,
                "original_transformer_nmae": transformer_nmae,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
