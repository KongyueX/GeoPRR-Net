"""Summarize batch-one end-to-end latency from frozen prediction caches.

``collect_predictions`` handles one image at a time and records wall time from
media decoding through the final reading payload.  Model construction and JSON
serialization are outside that interval.  This script keeps successful and
failed samples in the latency population and binds the report to cache
metadata and file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.make_failure_table import FAILURE_ORDER, _failure_category


LATENCY_PROTOCOL = "frozen_batch1_end_to_end_latency_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=predictions.jsonl")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=predictions.jsonl")
    return name, Path(raw_path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            runtime = row.get("runtime_seconds")
            if (
                not isinstance(runtime, (int, float))
                or not math.isfinite(float(runtime))
                or float(runtime) <= 0.0
            ):
                raise ValueError(f"{path}:{line_number} has invalid runtime_seconds")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _latency_metrics(seconds: list[float]) -> dict[str, float] | None:
    if not seconds:
        return None
    values = np.asarray(seconds, dtype=np.float64) * 1000.0
    mean_ms = float(np.mean(values))
    return {
        "samples": int(values.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "serial_images_per_second": 1000.0 / mean_ms,
    }


def summarize_cache(name: str, path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata_path = path.with_name(path.name + ".meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    signature = metadata.get("signature")
    if not isinstance(signature, dict):
        raise ValueError(f"{metadata_path} has no collection signature")
    if signature.get("reading_backend") != "compare":
        raise ValueError(f"{path} was not collected with the compare backend")
    if signature.get("include_transformer") is not True:
        raise ValueError(f"{path} does not include the complete reading pipeline")

    rows = _read_rows(path)
    all_seconds = [float(row["runtime_seconds"]) for row in rows]
    success_seconds = [
        float(row["runtime_seconds"]) for row in rows if row.get("status") is True
    ]
    failure_seconds = [
        float(row["runtime_seconds"]) for row in rows if row.get("status") is not True
    ]
    by_outcome: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        outcome = "success" if row.get("status") is True else _failure_category(row)
        by_outcome[outcome].append(float(row["runtime_seconds"]))
    outcome_order = ("success", *FAILURE_ORDER)
    return {
        "name": name,
        "source": str(path),
        "source_sha256": _sha256(path),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "batch_size": 1,
        "timed_scope": (
            "media decode through final reading payload; model initialization "
            "and JSON serialization excluded"
        ),
        "device": signature.get("device"),
        "samples": len(rows),
        "successes": len(success_seconds),
        "failures": len(failure_seconds),
        "all": _latency_metrics(all_seconds),
        "successful": _latency_metrics(success_seconds),
        "failed": _latency_metrics(failure_seconds),
        "by_outcome": {
            outcome: metrics
            for outcome in outcome_order
            if (metrics := _latency_metrics(by_outcome.get(outcome, []))) is not None
        },
        "collection_signature": signature,
    }


def _fmt(metrics: dict[str, Any] | None, key: str) -> str:
    if metrics is None:
        return "—"
    return f"{float(metrics[key]):.2f}"


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Batch-one end-to-end latency",
        "",
        (
            "Each frozen cache was collected serially (batch size 1). Timing "
            "starts before image decode and ends after the final reading payload; "
            "model initialization and JSON serialization are excluded. All "
            "samples, including failures and early exits, remain in the headline."
        ),
        "",
        "| Dataset | Device | N | Mean (ms) | Median (ms) | P95 (ms) | Serial FPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        metrics = summary["all"]
        lines.append(
            "| {name} | {device} | {samples} | {mean} | {median} | {p95} | {fps} |".format(
                name=summary["name"],
                device=summary.get("device") or "unknown",
                samples=summary["samples"],
                mean=_fmt(metrics, "mean_ms"),
                median=_fmt(metrics, "median_ms"),
                p95=_fmt(metrics, "p95_ms"),
                fps=_fmt(metrics, "serial_images_per_second"),
            )
        )

    lines.extend(
        [
            "",
            "## Latency by outcome",
            "",
            "| Dataset | Outcome | N | Median (ms) | P95 (ms) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    labels = {
        "success": "Success",
        "meter_not_found": "Meter not found",
        "pointer_not_found": "Pointer not found",
        "reading_backend_failed": "Reading backend",
        "collector_exception": "Collector",
        "other_failure": "Other",
    }
    for summary in summaries:
        for outcome, metrics in summary["by_outcome"].items():
            lines.append(
                "| {name} | {outcome} | {samples} | {median} | {p95} |".format(
                    name=summary["name"],
                    outcome=labels[outcome],
                    samples=metrics["samples"],
                    median=_fmt(metrics, "median_ms"),
                    p95=_fmt(metrics, "p95_ms"),
                )
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_named_path,
        required=True,
        metavar="NAME=PREDICTIONS",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [name for name, _ in args.dataset]
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    summaries = [summarize_cache(name, path) for name, path in args.dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summaries), encoding="utf-8")
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "protocol": LATENCY_PROTOCOL,
                "datasets": summaries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    print(json_path.resolve())


if __name__ == "__main__":
    main()
