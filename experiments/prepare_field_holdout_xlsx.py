"""Extract a prediction-independent field holdout manifest from an XLSX label pack.

The label workbooks used by this project contain one cropped dial image per
row as an embedded drawing.  This module reads the OOXML package directly so
that the original workbook remains untouched, extracts the embedded images,
and freezes a JSONL manifest plus an auditable protocol file before any model
is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import cv2
import numpy as np


WORKBOOK_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
DRAWING_MAIN_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

REQUIRED_HEADERS = (
    "image_name/图片名称",
    "scaleEnd/最大量程",
    "gt_result/读数",
    "source_kind",
    "source_image",
)
PREPARATION_PROTOCOL = "field_holdout_xlsx_preparation_v1"
SPLIT = "confirmatory_external_test"
DEFAULT_DATASET_NAME = "FieldHoldout-Xiangmu2-2026"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    payload = json.dumps(
        sorted(str(item) for item in sample_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def _atomic_text(path: Path, content: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _resolve_member(base_member: str, target: str) -> str:
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(
        posixpath.join(posixpath.dirname(base_member), target)
    )


def _relationships_path(member: str) -> str:
    directory = posixpath.dirname(member)
    filename = posixpath.basename(member)
    return posixpath.join(directory, "_rels", filename + ".rels")


def _read_relationships(
    archive: zipfile.ZipFile,
    source_member: str,
) -> dict[str, dict[str, str]]:
    path = _relationships_path(source_member)
    root = ET.fromstring(archive.read(path))
    result: dict[str, dict[str, str]] = {}
    for relationship in root.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        identifier = relationship.attrib["Id"]
        target = relationship.attrib["Target"]
        result[identifier] = {
            "target": _resolve_member(source_member, target),
            "type": relationship.attrib.get("Type", ""),
        }
    return result


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(path))
    strings: list[str] = []
    for item in root.findall(f"{{{WORKBOOK_NS}}}si"):
        strings.append(
            "".join(
                node.text or ""
                for node in item.iter(f"{{{WORKBOOK_NS}}}t")
            )
        )
    return strings


def _column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference.upper())
    if not match:
        raise ValueError(f"invalid cell reference: {cell_reference!r}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _cell_value(cell: ET.Element, shared: list[str]) -> Any:
    cell_type = cell.attrib.get("t", "n")
    value_node = cell.find(f"{{{WORKBOOK_NS}}}v")
    if cell_type == "inlineStr":
        return "".join(
            node.text or ""
            for node in cell.iter(f"{{{WORKBOOK_NS}}}t")
        )
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared[int(raw)]
    if cell_type == "b":
        return raw == "1"
    if cell_type in {"str", "e"}:
        return raw
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _sheet_paths(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook_member = "xl/workbook.xml"
    workbook = ET.fromstring(archive.read(workbook_member))
    relationships = _read_relationships(archive, workbook_member)
    result: dict[str, str] = {}
    sheets = workbook.find(f"{{{WORKBOOK_NS}}}sheets")
    if sheets is None:
        raise ValueError("workbook has no sheets")
    for sheet in sheets:
        name = sheet.attrib["name"]
        identifier = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
        result[name] = relationships[identifier]["target"]
    return result


def _sheet_rows(
    archive: zipfile.ZipFile,
    member: str,
    shared: list[str],
) -> list[tuple[int, list[Any]]]:
    root = ET.fromstring(archive.read(member))
    sheet_data = root.find(f"{{{WORKBOOK_NS}}}sheetData")
    if sheet_data is None:
        return []
    rows: list[tuple[int, list[Any]]] = []
    for row in sheet_data.findall(f"{{{WORKBOOK_NS}}}row"):
        excel_row = int(row.attrib["r"])
        values: dict[int, Any] = {}
        for cell in row.findall(f"{{{WORKBOOK_NS}}}c"):
            values[_column_index(cell.attrib["r"])] = _cell_value(cell, shared)
        width = max(values, default=-1) + 1
        rows.append((excel_row, [values.get(index) for index in range(width)]))
    return rows


def _records_from_sheet(
    rows: list[tuple[int, list[Any]]],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("label_template is empty")
    _, header_values = rows[0]
    headers = [str(value or "").strip() for value in header_values]
    missing = sorted(set(REQUIRED_HEADERS) - set(headers))
    if missing:
        raise ValueError(f"label_template missing required headers: {missing}")
    records: list[dict[str, Any]] = []
    for excel_row, values in rows[1:]:
        record = {
            header: values[index] if index < len(values) else None
            for index, header in enumerate(headers)
            if header
        }
        record["_excel_row"] = excel_row
        records.append(record)
    return records


def _summary_from_sheet(
    rows: list[tuple[int, list[Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for _, values in rows[1:]:
        if not values or values[0] in (None, ""):
            continue
        result[str(values[0])] = values[1] if len(values) > 1 else None
    return result


def _drawing_media_by_row(
    archive: zipfile.ZipFile,
    sheet_member: str,
) -> dict[int, str]:
    sheet_relationships = _read_relationships(archive, sheet_member)
    drawings = [
        value["target"]
        for value in sheet_relationships.values()
        if value["type"].endswith("/drawing")
    ]
    if len(drawings) != 1:
        raise ValueError(f"expected one drawing part, found {len(drawings)}")
    drawing_member = drawings[0]
    drawing_relationships = _read_relationships(archive, drawing_member)
    root = ET.fromstring(archive.read(drawing_member))
    namespaces = {
        "xdr": DRAWING_NS,
        "a": DRAWING_MAIN_NS,
        "r": OFFICE_REL_NS,
    }
    result: dict[int, str] = {}
    anchors = list(root.findall("xdr:twoCellAnchor", namespaces))
    anchors.extend(root.findall("xdr:oneCellAnchor", namespaces))
    for anchor in anchors:
        row_node = anchor.find("xdr:from/xdr:row", namespaces)
        column_node = anchor.find("xdr:from/xdr:col", namespaces)
        blip = anchor.find(".//a:blip", namespaces)
        if row_node is None or column_node is None or blip is None:
            continue
        if int(column_node.text or "-1") != 1:
            continue
        identifier = blip.attrib.get(f"{{{OFFICE_REL_NS}}}embed")
        if not identifier or identifier not in drawing_relationships:
            raise ValueError("drawing image relationship is missing")
        excel_row = int(row_node.text or "0") + 1
        if excel_row in result:
            raise ValueError(f"multiple images are anchored to Excel row {excel_row}")
        result[excel_row] = drawing_relationships[identifier]["target"]
    return result


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _finite_float(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    return normalized.strip("-") or "unknown"


def infer_meter_id(record: dict[str, Any]) -> str:
    source = str(record.get("source_image") or "").replace("\\", "/")
    parts = [part for part in source.split("/") if part]
    source_kind = str(record.get("source_kind") or "")
    if source_kind == "xiangmu2" and len(parts) >= 2 and parts[0] == "xiangmu2":
        return parts[1]
    if parts:
        return parts[0]
    return source_kind or "unknown_meter"


def infer_session_date(record: dict[str, Any]) -> str | None:
    source = str(record.get("source_image") or "")
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", source)
    return match.group(1) if match else None


def _bbox(value: Any) -> list[int] | None:
    if _missing(value):
        return None
    try:
        items = [int(float(item.strip())) for item in str(value).split(",")]
    except (TypeError, ValueError):
        return None
    return items if len(items) == 4 else None


def _decode_image(payload: bytes, name: str) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None or image.size == 0:
        raise ValueError(f"cannot decode embedded image {name}")
    return image


def _quality_features(image: np.ndarray) -> dict[str, float]:
    height, width = image.shape[:2]
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return {
        "width": int(width),
        "height": int(height),
        "short_side": int(min(width, height)),
        "aspect_ratio": float(width / max(height, 1)),
        "laplacian_variance_256": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "luminance_mean_256": float(np.mean(gray)),
        "luminance_std_256": float(np.std(gray)),
        "bright_pixel_ratio_256": float(np.mean(gray >= 245)),
        "dark_pixel_ratio_256": float(np.mean(gray <= 10)),
    }


def _pixel_sha256(image: np.ndarray) -> str:
    header = json.dumps(
        {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        },
        separators=(",", ":"),
    ).encode("ascii")
    return sha256_bytes(header + b"\0" + image.tobytes(order="C"))


def _perceptual_hash_64(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    transformed = cv2.dct(np.asarray(resized, dtype=np.float32))
    low_frequency = transformed[:8, :8].reshape(-1)
    # Ignore the DC term when choosing the threshold but retain a fixed
    # 64-bit representation for straightforward Hamming-distance audits.
    threshold = float(np.median(low_frequency[1:]))
    bits = low_frequency > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def _freeze_quality_groups(rows: list[dict[str, Any]]) -> dict[str, float]:
    definitions = {
        "natural_blur_q1": ("laplacian_variance_256", 0.25, "low"),
        "natural_low_light_q1": ("luminance_mean_256", 0.25, "low"),
        "natural_low_contrast_q1": ("luminance_std_256", 0.25, "low"),
        "natural_glare_q4": ("bright_pixel_ratio_256", 0.75, "high"),
        "natural_small_crop_q1": ("short_side", 0.25, "low"),
    }
    thresholds = {
        name: float(
            np.quantile(
                [float(row["metadata"]["quality"][field]) for row in rows],
                quantile,
            )
        )
        for name, (field, quantile, _) in definitions.items()
    }
    for row in rows:
        quality = row["metadata"]["quality"]
        groups = []
        for name, (field, _, direction) in definitions.items():
            value = float(quality[field])
            threshold = thresholds[name]
            if (direction == "low" and value <= threshold) or (
                direction == "high" and value >= threshold
            ):
                groups.append(name)
        if len(groups) >= 2:
            groups.append("natural_challenge_2of5")
        quality["groups"] = groups
        quality["challenge_count_5"] = len(
            [name for name in groups if name != "natural_challenge_2of5"]
        )
    return thresholds


def prepare_field_holdout(
    workbook: Path,
    output_dir: Path,
    manifest: Path,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    scale_start: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    workbook = workbook.resolve()
    output_dir = output_dir.resolve()
    manifest = manifest.resolve()
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    if not math.isfinite(scale_start):
        raise ValueError("scale_start must be finite")

    workbook_sha256 = sha256_file(workbook)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    extracted_hashes: list[str] = []
    with zipfile.ZipFile(workbook) as archive:
        shared = _shared_strings(archive)
        sheets = _sheet_paths(archive)
        if "label_template" not in sheets:
            raise ValueError("workbook has no label_template sheet")
        label_member = sheets["label_template"]
        records = _records_from_sheet(
            _sheet_rows(archive, label_member, shared)
        )
        workbook_summary = (
            _summary_from_sheet(_sheet_rows(archive, sheets["summary"], shared))
            if "summary" in sheets
            else {}
        )
        media_by_row = _drawing_media_by_row(archive, label_member)

        for record in records:
            excel_row = int(record["_excel_row"])
            reasons: list[str] = []
            scale_end = _finite_float(record.get("scaleEnd/最大量程"))
            ground_truth = _finite_float(record.get("gt_result/读数"))
            if scale_end is None:
                reasons.append("missing_or_invalid_scale_end")
            elif abs(scale_end - scale_start) <= 1e-12:
                reasons.append("zero_scale_span")
            if ground_truth is None:
                reasons.append("missing_or_invalid_ground_truth")
            elif scale_end is not None:
                low, high = sorted((scale_start, scale_end))
                if ground_truth < low or ground_truth > high:
                    reasons.append("ground_truth_outside_scale")
            media_member = media_by_row.get(excel_row)
            if media_member is None:
                reasons.append("missing_embedded_image")

            if reasons:
                excluded.append(
                    {
                        "excel_row": excel_row,
                        "reasons": sorted(set(reasons)),
                        "source_kind": record.get("source_kind"),
                        "source_image": record.get("source_image"),
                        "scale_end": scale_end,
                        "ground_truth": ground_truth,
                    }
                )
                continue

            assert media_member is not None
            assert scale_end is not None
            assert ground_truth is not None
            payload = archive.read(media_member)
            content_sha256 = sha256_bytes(payload)
            image = _decode_image(payload, media_member)
            extension = Path(media_member).suffix.lower() or ".png"
            sample_id = f"field_xm2_r{excel_row:05d}_{content_sha256[:12]}"
            image_path = image_dir / f"{sample_id}{extension}"
            if image_path.exists():
                if sha256_file(image_path) != content_sha256:
                    raise ValueError(f"existing extracted image hash mismatch: {image_path}")
            else:
                image_path.write_bytes(payload)
            extracted_hashes.append(content_sha256)

            meter_id = infer_meter_id(record)
            height, width = image.shape[:2]
            confidence = _finite_float(record.get("confidence"))
            included.append(
                {
                    "dataset": dataset_name,
                    "split": SPLIT,
                    "sample_id": sample_id,
                    "group_id": meter_id,
                    "meter_id": meter_id,
                    "image_path": str(image_path.resolve()),
                    "ground_truth": ground_truth,
                    "scale_start": float(scale_start),
                    "scale_end": scale_end,
                    "metadata": {
                        "source_workbook": str(workbook),
                        "source_workbook_sha256": workbook_sha256,
                        "source_excel_row": excel_row,
                        "source_kind": record.get("source_kind"),
                        "source_image": record.get("source_image"),
                        "source_crop_name": record.get("image_name/图片名称"),
                        "source_bbox_xyxy": _bbox(record.get("bbox")),
                        "source_detector_confidence": confidence,
                        "source_media_member": media_member,
                        "source_media_sha256": content_sha256,
                        "decoded_pixel_sha256": _pixel_sha256(image),
                        "perceptual_hash_phash64": _perceptual_hash_64(image),
                        "session_date": infer_session_date(record),
                        "group_identity_source": (
                            "source_image directory interpreted as physical "
                            "instrument identifier"
                        ),
                        "dial_bbox": [0, 0, int(width), int(height)],
                        "quality": _quality_features(image),
                    },
                }
            )

    if not included:
        raise ValueError("no eligible field holdout samples")
    quality_thresholds = _freeze_quality_groups(included)

    group_counts = Counter(str(row["group_id"]) for row in included)
    source_kind_counts = Counter(
        str(row["metadata"].get("source_kind") or "") for row in included
    )
    session_counts = Counter(
        str(row["metadata"].get("session_date") or "unknown")
        for row in included
    )
    quality_group_counts: Counter[str] = Counter()
    for row in included:
        quality_group_counts.update(row["metadata"]["quality"]["groups"])

    manifest_content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in included
    )
    _atomic_text(manifest, manifest_content, overwrite=overwrite)
    excluded_path = output_dir / "excluded_rows.jsonl"
    _atomic_text(
        excluded_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in excluded
        ),
        overwrite=overwrite,
    )

    protocol = {
        "schema_version": 1,
        "protocol": PREPARATION_PROTOCOL,
        "dataset": dataset_name,
        "split": SPLIT,
        "confirmatory_holdout": True,
        "selection_uses_model_predictions": False,
        "test_labels_used_for_model_or_threshold_selection": 0,
        "source_workbook": str(workbook),
        "source_workbook_sha256": workbook_sha256,
        "source_summary": workbook_summary,
        "scale_policy": (
            f"scale_start fixed to {float(scale_start)} before evaluation; "
            "scale_end and ground_truth taken from label_template"
        ),
        "inclusion_rule": (
            "include every row with a decodable embedded crop, finite non-zero "
            "scale span, finite ground truth inside the configured scale"
        ),
        "exclusion_rule": (
            "exclude only missing/invalid labels, zero scale span, out-of-range "
            "ground truth, or missing embedded image; no model output is consulted"
        ),
        "source_rows": len(included) + len(excluded),
        "included_rows": len(included),
        "excluded_rows": len(excluded),
        "exclusion_reasons": dict(
            sorted(
                Counter(
                    reason
                    for row in excluded
                    for reason in row["reasons"]
                ).items()
            )
        ),
        "groups": len(group_counts),
        "group_counts": dict(sorted(group_counts.items())),
        "group_identity_policy": (
            "for source_kind=xiangmu2 use the second source_image directory "
            "(for example 02-111); otherwise use the first source directory. "
            "These directory labels are treated as physical instrument IDs "
            "pending dataset-owner provenance confirmation"
        ),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "session_counts": dict(sorted(session_counts.items())),
        "quality_features_use_predictions_or_labels": False,
        "quality_quantile_thresholds": quality_thresholds,
        "quality_group_counts": dict(sorted(quality_group_counts.items())),
        "sample_ids_sha256": sample_ids_sha256(
            row["sample_id"] for row in included
        ),
        "embedded_image_hashes_sha256": sample_ids_sha256(extracted_hashes),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "excluded_rows_file": str(excluded_path),
        "excluded_rows_sha256": sha256_file(excluded_path),
        "evaluation_scope": (
            "reading conditional on the workbook-provided detector crop; "
            "upstream raw-image detector misses are reported separately"
        ),
        "publication_rights_status": (
            "pending dataset-owner confirmation for field-image publication "
            "or controlled reviewer access"
        ),
    }
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    _atomic_text(
        protocol_path,
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--scale-start", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_field_holdout(
        args.workbook,
        args.output_dir,
        args.manifest,
        dataset_name=args.dataset_name,
        scale_start=args.scale_start,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
