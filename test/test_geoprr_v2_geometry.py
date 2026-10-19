from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from experiments.geoprr_v2_data import (
    GeoPRRV2Dataset, GeometryConfig, GeometryContractError, crop_and_resize,
    dial_calibration, geometry_targets, load_controlled_pairs, transform_points,
)
from experiments.prepare_geoprr_v2_data import make_split, prepare


def synthetic_row(sample_id: str = "dial", y: float = 0.4, *, arc_degrees: float = 240,
                  num_ticks: int = 9) -> dict:
    bbox = np.asarray([0, 0, 256, 256], dtype=float)
    matrix = np.array([[1.1, 0.07, -0.08], [0.02, 0.9, 0.04], [0.1, -0.03, 1]])
    angles = np.deg2rad(140 + np.linspace(0, arc_degrees, num_ticks))
    plane = np.stack((0.5 + 0.35 * np.cos(angles), 0.5 + 0.35 * np.sin(angles)), -1)
    normal = transform_points(plane, np.linalg.inv(matrix))
    points = normal * (bbox[2:] - bbox[:2]) + bbox[:2]
    return {"sample_id": sample_id, "group_id": "gauge::scene", "image_path": "unused.png",
            "ground_truth": 50 + 80 * y, "scale_start": 50, "scale_end": 130,
            "metadata": {"keypoints": [{"type": "ScaleMark", "all_kp": points.tolist()},
                                        {"type": "Pointer", "origin_kp": [131, 120], "outside_kp": [190, 30]}],
                         "dial_bbox": bbox.tolist(), "homography": matrix.tolist(),
                         "long_num": num_ticks, "long_interval_degree": arc_degrees / (num_ticks - 1),
                         "pointer_angle": arc_degrees * y, "scene_name": "scene.hdr"}}


class GeoPRRV2GeometryTests(unittest.TestCase):
    def test_quantized_short_arc_uses_annotated_angular_spacing(self) -> None:
        row = synthetic_row(arc_degrees=60, num_ticks=7)
        row["metadata"].update(dial_bbox=[633, 213, 1286, 866], homography=np.eye(3).tolist())
        theta = np.deg2rad(240 + np.arange(7) * 10)
        pixels = np.stack((960 + 245 * np.cos(theta), 540 + 245 * np.sin(theta)), -1)
        row["metadata"]["keypoints"][0]["all_kp"] = pixels.astype(int).tolist()
        result = dial_calibration(row, np.eye(3), GeometryConfig())
        np.testing.assert_allclose(result["angles"], np.deg2rad(np.arange(7) * 10))
        self.assertLess(result["audit"]["anchor_progress_error"], 0.01)
        self.assertLess(result["audit"]["arc_relative_error"], 0.01)

    def test_released_short_arc_endpoint_meets_readout_tolerance(self) -> None:
        row = synthetic_row("sync_8352")
        row.update(ground_truth=49.389767597011655, scale_start=49, scale_end=211)
        row["metadata"].update(
            dial_bbox=[698, 300, 1221, 802], long_num=7, long_interval_degree=10,
            pointer_angle=0.14435836926357526,
            homography=[[0.6032142104434299, 1.0417101309537347, -0.2698913129319288],
                        [-0.9641700187690594, 0.6746588886232333, 0.6890913256038186],
                        [1.1783093019381194e-7, 0.12251612559867543, 1]],
        )
        row["metadata"]["keypoints"][0]["all_kp"] = [
            [1046, 385], [1074, 401], [1098, 421], [1119, 445], [1135, 472], [1146, 502], [1151, 533],
        ]
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        config = GeometryConfig()
        _, resize = crop_and_resize(image, row["metadata"]["dial_bbox"], 768)
        target = geometry_targets(dial_calibration(row, resize, config), 768, 192, config)
        self.assertLess(target["geometry_audit"]["gt_readout_error"], config.max_readout_error)

    def test_metric_calibration_recovers_cross_zero_ordered_arc(self) -> None:
        result = dial_calibration(synthetic_row(), np.eye(3), GeometryConfig())
        np.testing.assert_allclose(result["angles"], np.deg2rad(np.arange(9) * 30), atol=1e-12)
        self.assertLess(result["audit"]["radial_error"], 1e-12)
        targets = geometry_targets(result, 256, 64, GeometryConfig())
        self.assertAlmostEqual(float(targets["point_target"].sum()), 1.0, places=6)
        self.assertLess(targets["geometry_audit"]["gt_readout_error"], 0.001)

    def test_wrong_homography_coordinate_convention_fails(self) -> None:
        row = synthetic_row()
        row["metadata"]["homography"] = [[1, 0, 30], [0, 1, 20], [0, 0, 1]]
        with self.assertRaisesRegex(GeometryContractError, "circle audit"):
            dial_calibration(row, np.eye(3), GeometryConfig())

    def test_visible_pointer_tip_and_elevated_pivot_do_not_define_virtual_point(self) -> None:
        row = synthetic_row()
        before = geometry_targets(dial_calibration(row, np.eye(3), GeometryConfig()), 256, 64, GeometryConfig())
        row["metadata"]["keypoints"][1].update(origin_kp=[-100, 900], outside_kp=[1500, -800])
        after = geometry_targets(dial_calibration(row, np.eye(3), GeometryConfig()), 256, 64, GeometryConfig())
        torch.testing.assert_close(before["point_target"], after["point_target"])
        torch.testing.assert_close(before["field_target"], after["field_target"])

    def test_physical_reading_and_angle_mismatch_fails(self) -> None:
        row = synthetic_row()
        row["metadata"]["pointer_angle"] += 10
        with self.assertRaisesRegex(GeometryContractError, "disagree"):
            dial_calibration(row, np.eye(3), GeometryConfig())

    def test_short_arc_endpoint_bias_requires_finer_spatial_grid(self) -> None:
        row = synthetic_row(y=0, arc_degrees=120, num_ticks=7)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        config = GeometryConfig()
        _, resize256 = crop_and_resize(image, row["metadata"]["dial_bbox"], 256)
        coarse = dial_calibration(row, resize256, config)
        with self.assertRaisesRegex(GeometryContractError, "discretization"):
            geometry_targets(coarse, 256, 64, config)
        _, resize640 = crop_and_resize(image, row["metadata"]["dial_bbox"], 640)
        fine = dial_calibration(row, resize640, config)
        targets = geometry_targets(fine, 640, 160, config)
        self.assertLess(targets["geometry_audit"]["gt_readout_error"], config.max_readout_error)
        # U is layout-only even when the endpoint changes to the other end.
        row["ground_truth"] = row["scale_end"]
        row["metadata"]["pointer_angle"] = 120
        other = geometry_targets(dial_calibration(row, resize640, config), 640, 160, config)
        torch.testing.assert_close(targets["field_target"], other["field_target"])
        for y in (0.001, 0.01, 0.132785, 0.662749, 0.99, 0.999):
            row["ground_truth"] = row["scale_start"] + y * (row["scale_end"] - row["scale_start"])
            row["metadata"]["pointer_angle"] = 120 * y
            target = geometry_targets(dial_calibration(row, resize640, config), 640, 160, config)
            self.assertLess(target["geometry_audit"]["gt_readout_error"], config.max_readout_error)

    def test_crop_resize_uses_half_pixel_centers(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        _, matrix = crop_and_resize(image, [10, 20, 90, 60], 256)
        points = transform_points(np.array([[9.5, 19.5], [89.5, 59.5]]), matrix)
        np.testing.assert_allclose(points, [[-0.5, -0.5], [255.5, 255.5]], atol=1e-12)

    def test_known_image_warp_preserves_virtual_dial_point(self) -> None:
        config = GeometryConfig()
        row = synthetic_row()
        original = dial_calibration(row, np.eye(3), config)
        warp = np.array([[0.94, -0.04, 5], [0.03, 0.97, 2], [0.0002, -0.0001, 1]])
        warped = dial_calibration(row, warp, config)
        targets = geometry_targets(original, 256, 64, config)
        warped_targets = geometry_targets(warped, 256, 64, config)
        expected = transform_points(targets["virtual_point_xy"].numpy()[None], warp)[0]
        np.testing.assert_allclose(warped_targets["virtual_point_xy"], expected, atol=2e-5)

    def test_projective_camera_warp_transports_canonical_point_mass(self) -> None:
        row = synthetic_row(y=0.4)
        warp640 = np.array([[0.88, 0.20, 18], [-0.1, 1.05, 10], [0.001, -0.0003, 1]])
        errors = []
        for side, nominal_sigma in ((640, 1.0), (1280, 2.0)):
            # Both grids represent the same physical canonical kernel.  A
            # finer grid reduces the unavoidable bilinear resampling error.
            config = GeometryConfig(point_sigma=nominal_sigma)
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            _, resize = crop_and_resize(image, row["metadata"]["dial_bbox"], side)
            scale = side / 640
            refinement = np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1]])
            warp = refinement @ warp640 @ np.linalg.inv(refinement)
            a = geometry_targets(dial_calibration(row, resize, config), side, side // 4, config)
            b = geometry_targets(dial_calibration(row, warp @ resize, config), side, side // 4, config)
            output_to_roi = np.array([[4, 0, 1.5], [0, 4, 1.5], [0, 0, 1]])
            b_to_a = np.linalg.inv(output_to_roi) @ np.linalg.inv(warp) @ output_to_roi
            yy, xx = np.indices((side // 4, side // 4), dtype=np.float64)
            coordinates = transform_points(np.stack((xx, yy), axis=-1), b_to_a)
            grid = torch.from_numpy((2 * (coordinates + 0.5) / (side // 4) - 1).astype(np.float32))[None]
            transported = F.grid_sample(a["point_target"][None], grid, align_corners=False)[0]
            denominator = b_to_a[2, 0] * xx + b_to_a[2, 1] * yy + b_to_a[2, 2]
            jacobian = abs(np.linalg.det(b_to_a)) / np.abs(denominator) ** 3
            transported *= torch.from_numpy(jacobian.astype(np.float32))[None]
            transported /= transported.sum()
            errors.append(float(0.5 * (transported - b["point_target"]).abs().sum()))
        # The default subpixel kernel admits at most 10% total-variation
        # discretization error here; doubled sampling must reduce it below 3.5%.
        self.assertLess(errors[0], 0.10)
        self.assertLess(errors[1], 0.035)
        self.assertLess(errors[1], errors[0] * 0.45)

    def test_pair_factor_audit_rejects_unrecorded_appearance_change(self) -> None:
        a, b = synthetic_row("a", 0.2), synthetic_row("b", 0.8)
        fa = {"g": {"scale": 9}, "r": 0.2, "a": {"font": "one"}, "c": {"camera": [0, 0, 6]}}
        fb = dict(fa, r=0.8, a={"font": "two"})
        for row, factors in ((a, fa), (b, fb)):
            row["metadata"].update(layout_id="layout", factors=factors)
        pair = {"sample_a": "a", "sample_b": "b", "layout_id": "layout", "kind": "state", "factors_a": fa, "factors_b": fb}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.jsonl"
            path.write_text(json.dumps(pair) + "\n")
            with self.assertRaisesRegex(ValueError, "Non-intervened|non-intervened"):
                load_controlled_pairs(path, [a, b])

    def test_controlled_pair_shares_extra_photo_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.full((256, 256, 3), 180, dtype=np.uint8)
            cv2.imwrite(str(root / "dial.png"), image)
            rows = [synthetic_row("a", 0.2), synthetic_row("b", 0.8)]
            factors = [{"g": {"scale": 9}, "r": y, "a": {"font": "one"}, "c": {"camera": [0, 0, 6]}} for y in (0.2, 0.8)]
            for row, value in zip(rows, factors):
                row["image_path"] = str(root / "dial.png")
                row["metadata"].update(layout_id="layout", factors=value)
            pair = {"sample_a": "a", "sample_b": "b", "layout_id": "layout", "kind": "state", "factors_a": factors[0], "factors_b": factors[1]}
            path = root / "pairs.jsonl"
            path.write_text(json.dumps(pair) + "\n")
            sample = GeoPRRV2Dataset(rows, augment=True, require_geometry=False, pair_manifest=path)[0]
            torch.testing.assert_close(sample["image"], sample["pair_image"])
            self.assertTrue(sample["pair_valid"])
            self.assertNotEqual(float(sample["y"]), float(sample["pair_y"]))

    def test_scene_split_keeps_all_gauges_in_same_scene_together(self) -> None:
        rows = []
        for scene in range(5):
            for gauge in range(2):
                row = synthetic_row(f"{scene}_{gauge}")
                row["metadata"]["scene_name"] = f"scene{scene}.hdr"
                row["group_id"] = f"gauge{gauge}::scene{scene}"
                rows.append(row)
        split = make_split(rows, validation_fraction=0.4)
        by_id = {row["sample_id"]: row for row in rows}
        train_scenes = {by_id[sid]["metadata"]["scene_name"] for sid in split["train_sample_ids"]}
        validation_scenes = {by_id[sid]["metadata"]["scene_name"] for sid in split["validation_sample_ids"]}
        self.assertFalse(train_scenes & validation_scenes)

    def test_formal_prepare_refuses_incomplete_upload_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "annotations" / "train").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "upload incomplete"):
                prepare(root, output)
            self.assertFalse(output.exists())

    def test_source_image_resize_cannot_silently_change_annotation_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resized.png"
            cv2.imwrite(str(path), np.zeros((128, 128, 3), dtype=np.uint8))
            row = synthetic_row()
            row["image_path"] = str(path)
            row["metadata"].update(image_width=256, image_height=256)
            with self.assertRaisesRegex(GeometryContractError, "dimensions disagree"):
                GeoPRRV2Dataset([row], require_geometry=True)[0]


if __name__ == "__main__":
    unittest.main()
