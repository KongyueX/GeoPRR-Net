from __future__ import annotations

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.geoprr_v2_data import GeoPRRV2Dataset, load_controlled_pairs


def state_rows() -> list[dict]:
    return [{"sample_id": f"s{index}", "y": state,
             "metadata": {"layout_id": "layout", "factors": {
                 "g": {"ticks": 9}, "r": state, "a": {"font": "one"},
                 "c": {"position": [0, 0, 6]}}}}
            for index, state in enumerate((0.1, 0.3, 0.6, 0.9))]


def state_edge(rows: list[dict], a: int, b: int) -> dict:
    return {"sample_a": rows[a]["sample_id"], "sample_b": rows[b]["sample_id"],
            "layout_id": "layout", "kind": "state",
            "factors_a": rows[a]["metadata"]["factors"],
            "factors_b": rows[b]["metadata"]["factors"]}


def write_pairs(root: Path, pairs: list[dict]) -> Path:
    path = root / "pairs.jsonl"
    path.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
    return path


def write_snapshot(path: Path, factors: dict, *, layout_id: str = "layout") -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"layout_id": layout_id,
                   "fixed_factors": {key: factors[key] for key in ("g", "a", "c")}}, handle)


class MultiRecipientDataTests(unittest.TestCase):
    def test_eight_cycle_edges_expand_items_without_copying_unique_rows(self) -> None:
        rows = state_rows()
        edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_pairs(Path(temporary), [state_edge(rows, a, b) for a, b in edges])
            dataset = GeoPRRV2Dataset(rows, image_size=32, require_geometry=False, pair_manifest=path)
            self.assertEqual((len(dataset.rows), len(dataset)), (4, 8))
            self.assertEqual((dataset.unique_image_count, dataset.item_count, dataset.controlled_edge_count), (4, 8, 8))
            self.assertEqual(len(dataset.pairs["s0"]), 2)
            def sample(index: int, photometric: tuple | None) -> dict:
                return {"sample_id": rows[index]["sample_id"], "image": index, "y": rows[index]["y"]}
            with patch.object(dataset, "_sample", side_effect=sample):
                samples = [dataset[index] for index in range(len(dataset))]
            self.assertEqual([(s["sample_id"], s["pair_sample_id"]) for s in samples], [
                ("s0", "s1"), ("s0", "s3"), ("s1", "s0"), ("s1", "s2"),
                ("s2", "s1"), ("s2", "s3"), ("s3", "s2"), ("s3", "s0")])
            self.assertTrue(all(s["pair_valid"] and not s["pair_same_state"] for s in samples))

    def test_duplicate_directed_edge_is_rejected(self) -> None:
        rows = state_rows()
        with tempfile.TemporaryDirectory() as temporary:
            edge = state_edge(rows, 0, 1)
            path = write_pairs(Path(temporary), [edge, edge])
            with self.assertRaisesRegex(ValueError, "duplicate controlled pair edge"):
                load_controlled_pairs(path, rows)

    def test_unpaired_and_single_partner_items_retain_row_order(self) -> None:
        rows = state_rows()
        with tempfile.TemporaryDirectory() as temporary:
            path = write_pairs(Path(temporary), [state_edge(rows, 1, 3)])
            paired = GeoPRRV2Dataset(rows, image_size=32, require_geometry=False, pair_manifest=path)
            unpaired = GeoPRRV2Dataset(rows, image_size=32, require_geometry=False)
            for dataset in (paired, unpaired):
                self.assertEqual(len(dataset), 4)
                self.assertEqual([item[0] for item in dataset.items], [0, 1, 2, 3])
            self.assertEqual(paired.controlled_edge_count, 1)
            self.assertEqual(unpaired.controlled_edge_count, 0)
            self.assertEqual([p["sample_b"] if p is not None else None for _, p in paired.items],
                             [None, "s3", None, None])

    def test_snapshot_references_match_full_records_and_are_read_once(self) -> None:
        rows = state_rows()
        full_rows = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot.json.gz"
            write_snapshot(snapshot, rows[0]["metadata"]["factors"])
            for row in rows:
                row["metadata"]["factors"] = {"reference": str(snapshot), "r": row["y"]}
            edges = [state_edge(rows, 0, 1), state_edge(full_rows, 1, 2), state_edge(rows, 2, 3)]
            path = write_pairs(root, edges)
            with patch("experiments.geoprr_v2_data.gzip.open", wraps=gzip.open) as opened:
                pairs = load_controlled_pairs(path, rows)
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(sum(map(len, pairs.values())), 3)
            self.assertEqual(rows[0]["metadata"]["factors"], {"reference": str(snapshot), "r": 0.1})

    def test_snapshot_non_intervened_factor_change_is_rejected(self) -> None:
        rows = state_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (0, 1):
                factors = copy.deepcopy(rows[index]["metadata"]["factors"])
                if index == 1:
                    factors["a"]["font"] = "two"
                snapshot = root / f"snapshot{index}.json.gz"
                write_snapshot(snapshot, factors)
                rows[index]["metadata"]["factors"] = {"reference": str(snapshot), "r": rows[index]["y"]}
            path = write_pairs(root, [state_edge(rows, 0, 1)])
            with self.assertRaisesRegex(ValueError, "non-intervened factors changed"):
                load_controlled_pairs(path, rows)

    def test_snapshot_layout_must_match_pair_and_row(self) -> None:
        rows = state_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot.json.gz"
            write_snapshot(snapshot, rows[0]["metadata"]["factors"], layout_id="other")
            rows[0]["metadata"]["factors"] = {"reference": str(snapshot), "r": rows[0]["y"]}
            path = write_pairs(root, [state_edge(rows, 0, 1)])
            with self.assertRaisesRegex(ValueError, "snapshot layout must match"):
                load_controlled_pairs(path, rows)


if __name__ == "__main__":
    unittest.main()
