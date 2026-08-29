"""Expand the public RF100-VL ledger to every evaluated CNN preprocessing arm."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


SEEDS: Final[tuple[int, ...]] = (20_262_020, 20_262_021, 20_262_022)
PREPROCESSING: Final[tuple[str, ...]] = ("raw", "sarn_v2")
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
SAMPLES: Final[int] = 151
ROWS_PER_SEED: Final[int] = SAMPLES * len(CONDITIONS)
ROBUSTNESS_SEED: Final[int] = 20_260_720
CORE_MODELS: Final[tuple[str, ...]] = (
    "GeoPRR-Net",
    "GeoPRR raw anchor",
    "GeoPRR normalized endpoint",
)
FIELDS: Final[tuple[str, ...]] = (
    "model",
    "seed",
    "preprocessing",
    "group_id",
    "image_id",
    "condition",
    "target",
    "prediction",
    "absolute_error",
    "status",
    "relation_available",
)


@dataclass(frozen=True, slots=True)
class CnnSource:
    display_name: str
    source_name: str
    family: str
    directory: str


CNN_SOURCES: Final[tuple[CnnSource, ...]] = (
    CnnSource("ResNet-18", "Direct-ResNet18", "factorial", "resnet18_direct"),
    CnnSource("EfficientNet-B0", "EfficientNet-B0", "lightweight_baselines", "efficientnet_b0"),
    CnnSource("MobileNetV3-Large", "MobileNetV3-Large", "lightweight_baselines", "mobilenet_v3_large"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_public(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"RF100 CNN ledger is missing: {source}")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sample_id"]), str(row["condition"]))
            _require(key not in output, "duplicate RF100 CNN prediction")
            output[key] = row
    _require(len(output) == ROWS_PER_SEED, "RF100 CNN denominator differs")
    return output


def _source_path(
    root: Path,
    source: CnnSource,
    *,
    preprocessing: str,
    seed: int,
) -> Path:
    return (
        Path(root)
        / source.family
        / "evaluation"
        / preprocessing
        / "rf100"
        / source.directory
        / f"seed_{seed}"
        / "predictions.jsonl"
    )


class _Writer:
    def __init__(self, path: Path) -> None:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = destination.open("wb")
        compressed = gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0)
        self._text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self._text, fieldnames=FIELDS, lineterminator="\n")
        self.writer.writeheader()
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow(dict(row))
        self.rows += 1

    def __enter__(self) -> "_Writer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._text.flush()
        self._text.close()


def export_rf100_public_data(
    *,
    source_root: Path,
    input_path: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    existing = _read_public(input_path)
    core_rows = [row for row in existing if row["model"] in CORE_MODELS]
    old_efficientnet = {
        (row["preprocessing"], int(row["seed"]), row["image_id"], row["condition"]): row
        for row in existing
        if row["model"] in {"Raw EfficientNet-B0", "SARN-v2 + EfficientNet-B0"}
    }
    core_cells: dict[tuple[str, int], dict[tuple[str, str], dict[str, str]]] = {}
    for row in core_rows:
        cell = core_cells.setdefault((row["model"], int(row["seed"])), {})
        key = (row["image_id"], row["condition"])
        _require(key not in cell, "duplicate RF100 core row")
        cell[key] = row
    for model in CORE_MODELS:
        for seed in SEEDS:
            _require(len(core_cells[(model, seed)]) == ROWS_PER_SEED, "RF100 core denominator differs")

    references = {seed: core_cells[("GeoPRR-Net", seed)] for seed in SEEDS}
    reference_keys = set(references[SEEDS[0]])
    _require(len({key[0] for key in reference_keys}) == SAMPLES, "RF100 image count differs")
    for seed in SEEDS:
        _require(set(references[seed]) == reference_keys, "RF100 roster differs across seeds")
    for model in CORE_MODELS:
        for seed in SEEDS:
            _require(set(core_cells[(model, seed)]) == reference_keys, "RF100 core roster differs")

    summaries: dict[str, Any] = {}
    with _Writer(output_path) as writer:
        for model in CORE_MODELS:
            for seed in SEEDS:
                for key in sorted(reference_keys):
                    writer.write(core_cells[(model, seed)][key])

        for preprocessing in PREPROCESSING:
            summaries[preprocessing] = {}
            for source in CNN_SOURCES:
                summaries[preprocessing][source.display_name] = {}
                for seed in SEEDS:
                    ledger = _read_jsonl(
                        _source_path(source_root, source, preprocessing=preprocessing, seed=seed)
                    )
                    _require(set(ledger) == reference_keys, "RF100 CNN roster differs")
                    errors: list[float] = []
                    display = (
                        f"Raw {source.display_name}"
                        if preprocessing == "raw"
                        else f"SARN-v2 + {source.display_name}"
                    )
                    expected_method = (
                        f"{source.source_name}_seed_{seed}"
                        if preprocessing == "raw"
                        else f"SARN-v2+{source.source_name}_seed_{seed}"
                    )
                    for key in sorted(reference_keys):
                        row = ledger[key]
                        reference = references[seed][key]
                        _require(row["method"] == expected_method, "RF100 CNN method differs")
                        _require(int(row["robustness_seed"]) == ROBUSTNESS_SEED, "RF100 robustness seed differs")
                        passed = row["status"] == "pass"
                        prediction = float(row["normalized_progress"])
                        target = float(reference["target"])
                        _require(math.isfinite(prediction), "RF100 CNN prediction is non-finite")
                        error = abs(prediction - target) if passed else 1.0
                        errors.append(error)
                        output_row = {
                            "model": display,
                            "seed": seed,
                            "preprocessing": preprocessing,
                            "group_id": reference["group_id"],
                            "image_id": key[0],
                            "condition": key[1],
                            "target": target,
                            "prediction": prediction,
                            "absolute_error": error,
                            "status": row["status"],
                            "relation_available": False,
                        }
                        if source.display_name == "EfficientNet-B0" and old_efficientnet:
                            old = old_efficientnet[(preprocessing, seed, key[0], key[1])]
                            _require(
                                abs(float(old["prediction"]) - prediction) <= 1e-12
                                and abs(float(old["absolute_error"]) - error) <= 1e-12,
                                "RF100 EfficientNet ledger differs from the prior public rows",
                            )
                        writer.write(output_row)
                    summaries[preprocessing][source.display_name][str(seed)] = {
                        "rows": len(errors),
                        "nmae": statistics.fmean(errors),
                        "acc_at_2_percent": statistics.fmean(error <= 0.02 for error in errors),
                    }
        rows = writer.rows

    expected_rows = (len(CORE_MODELS) + len(PREPROCESSING) * len(CNN_SOURCES)) * len(SEEDS) * ROWS_PER_SEED
    _require(rows == expected_rows, "RF100 public Cartesian denominator differs")
    audit = {
        "schema_version": 1,
        "status": "pass",
        "cohort": "RF100-VL gauge subset",
        "samples": SAMPLES,
        "conditions": len(CONDITIONS),
        "source_groups": len({row["group_id"] for row in core_rows}),
        "seeds": list(SEEDS),
        "rows": rows,
        "methods": list(CORE_MODELS)
        + [
            (f"Raw {source.display_name}" if arm == "raw" else f"SARN-v2 + {source.display_name}")
            for arm in PREPROCESSING
            for source in CNN_SOURCES
        ],
        "preprocessing": list(PREPROCESSING),
        "prior_efficientnet_rows_reproduced": bool(old_efficientnet),
        "prediction_target_error_recomputed": True,
        "summary": summaries,
    }
    destination = Path(audit_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = export_rf100_public_data(
        source_root=args.source_root,
        input_path=args.input,
        output_path=args.output,
        audit_path=args.audit,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export_rf100_public_data"]
