"""Export and independently validate the public SyncG CNN ledgers.

The release intentionally includes both raw-image and SARN-v2 preprocessing
arms.  The explicit ``preprocessing`` field prevents the SARN-v2 predictions
used for paired model comparisons from being mistaken for raw-CNN results.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


SEEDS: Final[tuple[int, ...]] = (20_262_020, 20_262_021, 20_262_022)
PREPROCESSING_ARMS: Final[tuple[str, ...]] = ("raw", "sarn_v2")
EXPECTED_ROWS_PER_LEDGER: Final[int] = 9_348
EXPECTED_SAMPLES: Final[int] = 1_558
EXPECTED_SCENES: Final[int] = 14
ROBUSTNESS_SEED: Final[int] = 20_260_720


@dataclass(frozen=True, slots=True)
class CnnSource:
    model: str
    source_model_name: str
    family_root: str
    directory_name: str


CNN_SOURCES: Final[tuple[CnnSource, ...]] = (
    CnnSource("ResNet-18", "Direct-ResNet18", "factorial", "resnet18_direct"),
    CnnSource(
        "EfficientNet-B0",
        "EfficientNet-B0",
        "lightweight_baselines",
        "efficientnet_b0",
    ),
    CnnSource(
        "MobileNetV3-Large",
        "MobileNetV3-Large",
        "lightweight_baselines",
        "mobilenet_v3_large",
    ),
)

OUTPUT_FIELDS: Final[tuple[str, ...]] = (
    "model",
    "source_model_name",
    "preprocessing",
    "seed",
    "scene_id",
    "image_id",
    "condition",
    "target",
    "prediction",
    "absolute_error",
    "status",
    "source_method",
    "source_protocol",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source_path(
    root: Path,
    source: CnnSource,
    *,
    preprocessing: str,
    seed: int,
) -> Path:
    return (
        Path(root)
        / source.family_root
        / "evaluation"
        / preprocessing
        / "syncg_scene_holdout"
        / source.directory_name
        / f"seed_{seed}"
        / "predictions.jsonl"
    )


def _load_reference(
    path: Path,
) -> tuple[
    tuple[tuple[str, str], ...],
    dict[tuple[str, str], tuple[float, str]],
]:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8-sig"))
    rows = payload.get("per_sample_condition")
    _require(isinstance(rows, list), "formal SyncG reference rows are missing")
    _require(len(rows) == EXPECTED_ROWS_PER_LEDGER, "formal SyncG row count differs")
    ordered: list[tuple[str, str]] = []
    indexed: dict[tuple[str, str], tuple[float, str]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "formal SyncG row is malformed")
        key = (str(row["sample_id"]), str(row["condition"]))
        target = float(row["normalized_target"])
        scene = str(row["scene_stem"])
        _require(key not in indexed, "formal SyncG keys repeat")
        _require(math.isfinite(target) and 0.0 <= target <= 1.0, "target is invalid")
        indexed[key] = (target, scene)
        ordered.append(key)
    _require(len({key[0] for key in ordered}) == EXPECTED_SAMPLES, "sample count differs")
    _require(len({value[1] for value in indexed.values()}) == EXPECTED_SCENES, "scene count differs")
    return tuple(ordered), indexed


def _load_prediction_ledger(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"CNN prediction ledger is missing: {source}")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            _require(isinstance(row, dict), "CNN prediction row is malformed")
            key = (str(row["sample_id"]), str(row["condition"]))
            _require(key not in indexed, "CNN prediction keys repeat")
            indexed[key] = row
    _require(len(indexed) == EXPECTED_ROWS_PER_LEDGER, "CNN ledger row count differs")
    return indexed


class _GzipCsvWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.path.open("wb")
        compressed = gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
        )
        self._text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self._text, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        self.writer.writeheader()
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow(dict(row))
        self.rows += 1

    def close(self) -> None:
        self._text.flush()
        self._text.close()

    def __enter__(self) -> "_GzipCsvWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def export_public_cnn_data(
    *,
    source_root: Path,
    reference_path: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    ordered_keys, reference = _load_reference(reference_path)
    summaries: dict[str, Any] = {}
    with _GzipCsvWriter(output_path) as output:
        for preprocessing in PREPROCESSING_ARMS:
            summaries[preprocessing] = {}
            for source in CNN_SOURCES:
                summaries[preprocessing][source.model] = {}
                for seed in SEEDS:
                    ledger = _load_prediction_ledger(
                        _source_path(
                            source_root,
                            source,
                            preprocessing=preprocessing,
                            seed=seed,
                        )
                    )
                    _require(set(ledger) == set(ordered_keys), "CNN roster differs")
                    errors: list[float] = []
                    expected_method = (
                        f"{source.source_model_name}_seed_{seed}"
                        if preprocessing == "raw"
                        else f"SARN-v2+{source.source_model_name}_seed_{seed}"
                    )
                    for key in ordered_keys:
                        row = ledger[key]
                        target, scene = reference[key]
                        _require(row.get("status") == "pass", "CNN row did not pass")
                        _require(row.get("failure_code") is None, "CNN failure code is set")
                        _require(row.get("method") == expected_method, "CNN method differs")
                        _require(
                            int(row.get("robustness_seed")) == ROBUSTNESS_SEED,
                            "CNN robustness seed differs",
                        )
                        prediction = float(row["normalized_progress"])
                        _require(
                            math.isfinite(prediction) and 0.0 <= prediction <= 1.0,
                            "CNN prediction is invalid",
                        )
                        error = abs(prediction - target)
                        errors.append(error)
                        output.write(
                            {
                                "model": source.model,
                                "source_model_name": source.source_model_name,
                                "preprocessing": preprocessing,
                                "seed": seed,
                                "scene_id": scene,
                                "image_id": key[0],
                                "condition": key[1],
                                "target": target,
                                "prediction": prediction,
                                "absolute_error": error,
                                "status": "pass",
                                "source_method": row["method"],
                                "source_protocol": row["protocol"],
                            }
                        )
                    summaries[preprocessing][source.model][str(seed)] = {
                        "rows": len(errors),
                        "nmae": float(statistics.fmean(errors)),
                        "acc_at_2_percent": float(
                            statistics.fmean(error <= 0.02 for error in errors)
                        ),
                    }
        rows = output.rows
    expected_rows = (
        len(PREPROCESSING_ARMS)
        * len(CNN_SOURCES)
        * len(SEEDS)
        * EXPECTED_ROWS_PER_LEDGER
    )
    _require(rows == expected_rows, "public CNN Cartesian row count differs")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "rows": rows,
        "source_ledgers": len(PREPROCESSING_ARMS) * len(CNN_SOURCES) * len(SEEDS),
        "samples": EXPECTED_SAMPLES,
        "scenes": EXPECTED_SCENES,
        "conditions": 6,
        "seeds": list(SEEDS),
        "preprocessing": list(PREPROCESSING_ARMS),
        "models": [source.model for source in CNN_SOURCES],
        "all_rows_pass": True,
        "prediction_target_error_recomputed": True,
        "summary": summaries,
        "correction_note": (
            "The initial 84,132-row release contained valid SARN-v2 CNN "
            "predictions but described them as raw. This corrected ledger "
            "retains those rows, adds the 84,132 authoritative raw rows, and "
            "labels preprocessing explicitly."
        ),
    }
    audit_output = Path(audit_path).resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing factorial/ and lightweight_baselines/ evaluation ledgers.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Formal SyncG result containing the paired target and scene roster.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    audit = export_public_cnn_data(
        source_root=args.source_root,
        reference_path=args.reference,
        output_path=args.output,
        audit_path=args.audit,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["OUTPUT_FIELDS", "export_public_cnn_data"]
