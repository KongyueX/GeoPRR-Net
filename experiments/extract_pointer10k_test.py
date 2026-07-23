"""Verify the official Pointer-10K archive and extract only its test split.

The project uses Pointer-10K as an auxiliary, zero-shot pointer-direction
benchmark.  Train and validation images are intentionally not extracted by
this helper so that an accidental fine-tuning run cannot silently contaminate
the external test protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo


POINTER10K_ARCHIVE_SHA256 = (
    "716c521c009721c235a5cdef6538b7dd"
    "bdedd87595ad2ee56c6b63402ecaa3a7"
)
POINTER10K_ARCHIVE_ROOT = PurePosixPath("Database/Done/pointer_10k")
POINTER10K_TEST_ANNOTATION = (
    POINTER10K_ARCHIVE_ROOT / "annotations" / "ann_test_pointer.json"
)
POINTER10K_TEST_IMAGE_PREFIX = (
    POINTER10K_ARCHIVE_ROOT / "images" / "test_pointer"
)
POINTER10K_TEST_ANNOTATION_SHA256 = (
    "b580e5cb9bc7b5898250782489d241f0707dab82f62718a57ff19dff520be698"
)
POINTER10K_TEST_IMAGES = 539
POINTER10K_TEST_POINTERS = 685
POINTER10K_TEST_SINGLE_POINTER_IMAGES = 438


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _safe_destination(root: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive path: {relative}")
    destination = (root / Path(*relative.parts)).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise ValueError(f"archive path escapes output directory: {relative}")
    return destination


def _read_and_validate_annotation(
    archive: ZipFile,
) -> tuple[dict, list[str], dict[str, int | str]]:
    annotation_name = POINTER10K_TEST_ANNOTATION.as_posix()
    try:
        payload = archive.read(annotation_name)
    except KeyError as exc:
        raise FileNotFoundError(
            f"official test annotation is missing from archive: {annotation_name}"
        ) from exc

    annotation_hash = hashlib.sha256(payload).hexdigest()
    if annotation_hash != POINTER10K_TEST_ANNOTATION_SHA256:
        raise ValueError(
            "Pointer-10K test annotation drift: "
            f"expected {POINTER10K_TEST_ANNOTATION_SHA256}, got {annotation_hash}"
        )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Pointer-10K test annotation must be a COCO JSON object")
    images = value.get("images")
    annotations = value.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Pointer-10K test annotation lacks COCO images/annotations")

    pointer_counts = Counter(int(item["image_id"]) for item in annotations)
    image_names = [str(item["file_name"]) for item in images]
    if len(images) != POINTER10K_TEST_IMAGES:
        raise ValueError(
            f"Pointer-10K test has {len(images)} images; "
            f"expected {POINTER10K_TEST_IMAGES}"
        )
    if len(annotations) != POINTER10K_TEST_POINTERS:
        raise ValueError(
            f"Pointer-10K test has {len(annotations)} pointer instances; "
            f"expected {POINTER10K_TEST_POINTERS}"
        )
    single_pointer_images = sum(
        pointer_counts.get(int(image["id"]), 0) == 1 for image in images
    )
    if single_pointer_images != POINTER10K_TEST_SINGLE_POINTER_IMAGES:
        raise ValueError(
            f"Pointer-10K test has {single_pointer_images} single-pointer images; "
            f"expected {POINTER10K_TEST_SINGLE_POINTER_IMAGES}"
        )
    if len(set(image_names)) != len(image_names):
        raise ValueError("Pointer-10K test contains duplicate image file names")

    audit: dict[str, int | str] = {
        "annotation_sha256": annotation_hash,
        "test_images": len(images),
        "test_pointer_instances": len(annotations),
        "single_pointer_images": single_pointer_images,
        "multi_pointer_images": len(images) - single_pointer_images,
    }
    return value, image_names, audit


def _extract_member(
    archive: ZipFile,
    info: ZipInfo,
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == info.file_size:
        if _crc32_file(destination) == info.CRC:
            return "skipped"
    partial = destination.with_name(destination.name + ".part")
    with archive.open(info, "r") as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    extracted_size = partial.stat().st_size
    if extracted_size != info.file_size:
        partial.unlink(missing_ok=True)
        raise IOError(
            f"incomplete extraction for {info.filename}: "
            f"{extracted_size} != {info.file_size}"
        )
    partial.replace(destination)
    return "extracted"


def extract_pointer10k_test(
    archive_path: Path,
    output_root: Path,
    *,
    verify_archive_hash: bool = True,
) -> dict[str, object]:
    archive_path = archive_path.resolve()
    output_root = output_root.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    archive_hash = sha256_file(archive_path)
    if verify_archive_hash and archive_hash != POINTER10K_ARCHIVE_SHA256:
        raise ValueError(
            "Pointer-10K archive identity mismatch: "
            f"expected {POINTER10K_ARCHIVE_SHA256}, got {archive_hash}. "
            "Use --allow-unverified-archive only for a diagnostic copy."
        )

    try:
        with ZipFile(archive_path) as archive:
            _, image_names, annotation_audit = _read_and_validate_annotation(archive)
            by_name = {info.filename: info for info in archive.infolist()}
            required: list[tuple[str, PurePosixPath]] = [
                (
                    POINTER10K_TEST_ANNOTATION.as_posix(),
                    PurePosixPath("annotations/ann_test_pointer.json"),
                )
            ]
            required.extend(
                (
                    (POINTER10K_TEST_IMAGE_PREFIX / image_name).as_posix(),
                    PurePosixPath("images/test_pointer") / image_name,
                )
                for image_name in image_names
            )
            missing = [name for name, _ in required if name not in by_name]
            if missing:
                raise FileNotFoundError(
                    f"Pointer-10K archive misses {len(missing)} official test files; "
                    f"first missing: {missing[0]}"
                )

            counts: Counter[str] = Counter()
            for archive_name, relative in required:
                destination = _safe_destination(output_root, relative)
                counts[_extract_member(archive, by_name[archive_name], destination)] += 1
    except BadZipFile as exc:
        raise ValueError(f"invalid Pointer-10K ZIP archive: {archive_path}") from exc

    audit: dict[str, object] = {
        "protocol": "pointer10k_official_test_extraction_v1",
        "archive": str(archive_path),
        "archive_sha256": archive_hash,
        "archive_identity_verified": bool(
            verify_archive_hash and archive_hash == POINTER10K_ARCHIVE_SHA256
        ),
        "output_root": str(output_root),
        "extraction_scope": "official test images and ann_test_pointer.json only",
        "reused_file_policy": "size and ZIP CRC-32 must both match",
        "train_or_validation_extracted": False,
        "files_extracted": int(counts["extracted"]),
        "files_reused": int(counts["skipped"]),
        **annotation_audit,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "extraction.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/Pointer10K_official/pointer_10k"),
    )
    parser.add_argument(
        "--allow-unverified-archive",
        action="store_true",
        help="diagnostic only: accept a different outer ZIP hash",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = extract_pointer10k_test(
        args.archive,
        args.output,
        verify_archive_hash=not args.allow_unverified_archive,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
