from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from experiments.audit_field_holdout_leakage import (
    AUDIT_PROTOCOL,
    audit_field_holdout_leakage,
    run_cli,
)
from experiments.prepare_field_holdout_xlsx import (
    _perceptual_hash_64,
    _pixel_sha256,
)


def _image(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("failed to encode fixture image")
    return encoded.tobytes()


def _row(
    sample_id: str,
    group_id: str,
    image_path: Path,
    split: str,
    *,
    pixel_hash: str,
    phash: str,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "meter_id": group_id,
        "image_path": str(image_path),
        "split": split,
        # Values below deliberately exist to verify that the auditor does not
        # need them for selection or duplicate matching.
        "ground_truth": 123.456,
        "scale_start": 0.0,
        "scale_end": 999.0,
        "metadata": {
            "decoded_pixel_sha256": pixel_hash,
            "perceptual_hash_phash64": phash,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_xlsx_media(path: Path, payloads: list[bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, payload in enumerate(payloads, 1):
            archive.writestr(f"xl/media/image{index}.png", payload)


def _fixture_paths(root: Path) -> dict[str, Path]:
    data_root = root / "data"
    data_root.mkdir()
    xiangmu1 = data_root / "xiangmu1.xlsx"
    _write_xlsx_media(xiangmu1, [])
    return {
        "combined": root / "combined.jsonl",
        "development": root / "development.jsonl",
        "confirmatory": root / "confirmatory.jsonl",
        "data_root": data_root,
        "xiangmu1": xiangmu1,
    }


def _write_partitioned_fixture(
    paths: dict[str, Path],
    development: list[dict[str, object]],
    confirmatory: list[dict[str, object]],
) -> None:
    _write_jsonl(paths["combined"], development + confirmatory)
    _write_jsonl(paths["development"], development)
    _write_jsonl(paths["confirmatory"], confirmatory)


class AuditFieldHoldoutLeakageTests(unittest.TestCase):
    def test_near_phash_is_reported_but_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _fixture_paths(root)
            development = [
                _row(
                    "dev-1",
                    "meter-dev",
                    root / "dev.png",
                    "field_development",
                    pixel_hash="a" * 64,
                    phash="0000000000000000",
                )
            ]
            confirmatory = [
                _row(
                    "confirm-1",
                    "meter-confirm",
                    root / "confirm.png",
                    "field_confirmatory",
                    pixel_hash="b" * 64,
                    phash="0000000000000001",
                )
            ]
            _write_partitioned_fixture(paths, development, confirmatory)

            audit = audit_field_holdout_leakage(
                paths["combined"],
                paths["development"],
                paths["confirmatory"],
                paths["data_root"],
                paths["xiangmu1"],
            )

            self.assertEqual(audit["protocol"], AUDIT_PROTOCOL)
            self.assertTrue(audit["passed"])
            self.assertFalse(audit["fatal_findings"])
            self.assertEqual(
                audit["development_confirmatory_phash"][
                    "minimum_hamming_distance"
                ],
                1,
            )
            self.assertEqual(
                len(
                    audit["development_confirmatory_phash"][
                        "candidates_le_2"
                    ]
                ),
                1,
            )
            self.assertFalse(
                audit["audit_scope"]["uses_ground_truth_readings_for_selection"]
            )

    def test_exact_pixel_overlap_and_cross_group_hash_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _fixture_paths(root)
            shared_hash = "c" * 64
            development = [
                _row(
                    "dev-1",
                    "meter-dev",
                    root / "dev.png",
                    "field_development",
                    pixel_hash=shared_hash,
                    phash="0000000000000000",
                )
            ]
            confirmatory = [
                _row(
                    "confirm-1",
                    "meter-confirm",
                    root / "confirm.png",
                    "field_confirmatory",
                    pixel_hash=shared_hash,
                    phash="ffffffffffffffff",
                )
            ]
            _write_partitioned_fixture(paths, development, confirmatory)

            audit = audit_field_holdout_leakage(
                paths["combined"],
                paths["development"],
                paths["confirmatory"],
                paths["data_root"],
                paths["xiangmu1"],
            )

            self.assertFalse(audit["passed"])
            codes = {finding["code"] for finding in audit["fatal_findings"]}
            self.assertIn(
                "development_confirmatory_exact_intersection",
                codes,
            )
            self.assertIn(
                "decoded_pixel_hash_crosses_physical_groups",
                codes,
            )
            self.assertEqual(
                audit["development_confirmatory_exact_intersections"][
                    "decoded_pixel_sha256"
                ]["shared_values"],
                1,
            )

    def test_historical_xlsx_exact_match_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _fixture_paths(root)
            historical_image = _image(7)
            other_image = _image(8)
            _write_xlsx_media(paths["xiangmu1"], [_png(historical_image)])
            historical_folder = paths["data_root"] / "legacy" / "session"
            historical_folder.mkdir(parents=True)
            (historical_folder / "img_000123.png").write_bytes(
                _png(other_image)
            )
            development = [
                _row(
                    "dev-1",
                    "meter-dev",
                    root / "dev.png",
                    "field_development",
                    pixel_hash=_pixel_sha256(historical_image),
                    phash=_perceptual_hash_64(historical_image),
                )
            ]
            confirmatory = [
                _row(
                    "confirm-1",
                    "meter-confirm",
                    root / "confirm.png",
                    "field_confirmatory",
                    pixel_hash=_pixel_sha256(other_image),
                    phash=_perceptual_hash_64(other_image),
                )
            ]
            _write_partitioned_fixture(paths, development, confirmatory)

            audit = audit_field_holdout_leakage(
                paths["combined"],
                paths["development"],
                paths["confirmatory"],
                paths["data_root"],
                paths["xiangmu1"],
            )

            self.assertFalse(audit["passed"])
            self.assertEqual(
                audit["historical_scan"]["xiangmu1_xl_media_members"],
                1,
            )
            self.assertEqual(
                audit["historical_scan"]["recursive_img_png_files"],
                1,
            )
            self.assertEqual(
                len(
                    audit["holdout_vs_historical"][
                        "exact_decoded_pixel_matches"
                    ]
                ),
                2,
            )
            self.assertIn(
                "holdout_exactly_matches_historical_data",
                {
                    finding["code"]
                    for finding in audit["fatal_findings"]
                },
            )

    def test_historical_equal_phash_with_different_pixels_is_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _fixture_paths(root)
            historical_image = _image(17)
            _write_xlsx_media(paths["xiangmu1"], [_png(historical_image)])
            historical_phash = _perceptual_hash_64(historical_image)
            development = [
                _row(
                    "dev-1",
                    "meter-dev",
                    root / "dev.png",
                    "field_development",
                    pixel_hash="f" * 64,
                    phash=historical_phash,
                )
            ]
            confirmatory = [
                _row(
                    "confirm-1",
                    "meter-confirm",
                    root / "confirm.png",
                    "field_confirmatory",
                    pixel_hash="e" * 64,
                    phash="ffffffffffffffff",
                )
            ]
            _write_partitioned_fixture(paths, development, confirmatory)

            audit = audit_field_holdout_leakage(
                paths["combined"],
                paths["development"],
                paths["confirmatory"],
                paths["data_root"],
                paths["xiangmu1"],
            )

            self.assertTrue(audit["passed"])
            candidates = audit["holdout_vs_historical"][
                "phash_candidates_le_2"
            ]
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["phash_hamming_distance"], 0)

    def test_partition_identity_drift_is_protocol_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _fixture_paths(root)
            development = [
                _row(
                    "dev-1",
                    "meter-dev",
                    root / "dev.png",
                    "field_development",
                    pixel_hash="d" * 64,
                    phash="0000000000000000",
                )
            ]
            confirmatory = [
                _row(
                    "confirm-1",
                    "meter-confirm",
                    root / "confirm.png",
                    "field_confirmatory",
                    pixel_hash="e" * 64,
                    phash="ffffffffffffffff",
                )
            ]
            _write_partitioned_fixture(paths, development, confirmatory)
            drifted = dict(development[0])
            drifted["image_path"] = str(root / "different.png")
            _write_jsonl(paths["development"], [drifted])

            audit = audit_field_holdout_leakage(
                paths["combined"],
                paths["development"],
                paths["confirmatory"],
                paths["data_root"],
                paths["xiangmu1"],
            )

            self.assertFalse(audit["passed"])
            self.assertFalse(audit["protocol_consistency"]["passed"])
            self.assertIn(
                "partition_identity_drift",
                {
                    issue["code"]
                    for issue in audit["protocol_consistency"]["issues"]
                },
            )

    def test_cli_exit_code_changes_only_for_fatal_findings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _fixture_paths(root)
            development = [
                _row(
                    "dev-1",
                    "meter-dev",
                    root / "dev.png",
                    "field_development",
                    pixel_hash="1" * 64,
                    phash="0000000000000000",
                )
            ]
            confirmatory = [
                _row(
                    "confirm-1",
                    "meter-confirm",
                    root / "confirm.png",
                    "field_confirmatory",
                    pixel_hash="2" * 64,
                    phash="0000000000000003",
                )
            ]
            _write_partitioned_fixture(paths, development, confirmatory)
            output = root / "audit.json"
            arguments = [
                "--combined-manifest",
                str(paths["combined"]),
                "--development-manifest",
                str(paths["development"]),
                "--confirmatory-manifest",
                str(paths["confirmatory"]),
                "--data-root",
                str(paths["data_root"]),
                "--xiangmu1-workbook",
                str(paths["xiangmu1"]),
                "--output",
                str(output),
            ]

            self.assertEqual(run_cli(arguments), 0)
            first = output.read_text(encoding="utf-8")
            self.assertTrue(json.loads(first)["passed"])

            # Preserve the same sample IDs and paths but introduce an exact
            # decoded-pixel intersection. This is fatal; the pHash distance
            # itself remains non-fatal.
            confirmatory[0]["metadata"]["decoded_pixel_sha256"] = "1" * 64
            _write_partitioned_fixture(paths, development, confirmatory)
            self.assertEqual(run_cli(arguments + ["--overwrite"]), 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["passed"])


if __name__ == "__main__":
    unittest.main()
