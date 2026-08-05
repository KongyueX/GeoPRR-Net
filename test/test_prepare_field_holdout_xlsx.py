from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from experiments.prepare_field_holdout_xlsx import (
    PREPARATION_PROTOCOL,
    infer_meter_id,
    prepare_field_holdout,
)


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XDR_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _inline_cell(reference: str, value: str) -> str:
    return (
        f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>'
    )


def _number_cell(reference: str, value: float) -> str:
    return f'<c r="{reference}"><v>{value}</v></c>'


def _png_payload(value: int) -> bytes:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("failed to build test PNG")
    return encoded.tobytes()


def _write_fixture(path: Path) -> None:
    headers = [
        "image_name/图片名称",
        "image",
        "scaleEnd/最大量程",
        "gt_result/读数",
        "source_kind",
        "source_image",
        "bbox",
        "confidence",
    ]
    header_cells = "".join(
        _inline_cell(f"{chr(ord('A') + index)}1", value)
        for index, value in enumerate(headers)
    )
    valid_cells = "".join(
        [
            _inline_cell("A2", "valid.bmp"),
            _number_cell("C2", 1.6),
            _number_cell("D2", 0.45),
            _inline_cell("E2", "xiangmu2"),
            _inline_cell(
                "F2",
                "xiangmu2/02-111/2-111-2026-06-23_13-47-08.jpg",
            ),
            _inline_cell("G2", "1,2,20,22"),
            _number_cell("H2", 0.91),
        ]
    )
    invalid_cells = "".join(
        [
            _inline_cell("A3", "invalid.bmp"),
            _inline_cell("E3", "xiangmu2"),
            _inline_cell(
                "F3",
                "xiangmu2/06-151/6-151-2026-06-23_13-47-08.jpg",
            ),
        ]
    )
    sheet1 = (
        f'<worksheet xmlns="{MAIN_NS}"><sheetData>'
        f'<row r="1">{header_cells}</row>'
        f'<row r="2">{valid_cells}</row>'
        f'<row r="3">{invalid_cells}</row>'
        "</sheetData></worksheet>"
    )
    sheet2 = (
        f'<worksheet xmlns="{MAIN_NS}"><sheetData>'
        '<row r="1">'
        f'{_inline_cell("A1", "item")}{_inline_cell("B1", "value")}'
        "</row>"
        '<row r="2">'
        f'{_inline_cell("A2", "raw_images")}{_number_cell("B2", 2)}'
        "</row>"
        "</sheetData></worksheet>"
    )
    workbook = (
        f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}"><sheets>'
        '<sheet name="label_template" sheetId="1" r:id="rId1"/>'
        '<sheet name="summary" sheetId="2" r:id="rId2"/>'
        "</sheets></workbook>"
    )
    workbook_rels = (
        f'<Relationships xmlns="{PACKAGE_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/>'
        "</Relationships>"
    )
    sheet1_rels = (
        f'<Relationships xmlns="{PACKAGE_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
        'Target="../drawings/drawing1.xml"/>'
        "</Relationships>"
    )
    drawing = (
        f'<xdr:wsDr xmlns:xdr="{XDR_NS}" xmlns:a="{DRAWING_NS}" xmlns:r="{REL_NS}">'
        '<xdr:twoCellAnchor><xdr:from><xdr:col>1</xdr:col><xdr:row>1</xdr:row>'
        '</xdr:from><xdr:pic><xdr:blipFill><a:blip r:embed="rId1"/>'
        '</xdr:blipFill></xdr:pic></xdr:twoCellAnchor>'
        '<xdr:twoCellAnchor><xdr:from><xdr:col>1</xdr:col><xdr:row>2</xdr:row>'
        '</xdr:from><xdr:pic><xdr:blipFill><a:blip r:embed="rId2"/>'
        '</xdr:blipFill></xdr:pic></xdr:twoCellAnchor>'
        "</xdr:wsDr>"
    )
    drawing_rels = (
        f'<Relationships xmlns="{PACKAGE_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        'Target="../media/image1.png"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        'Target="../media/image2.png"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet1)
        archive.writestr("xl/worksheets/sheet2.xml", sheet2)
        archive.writestr(
            "xl/worksheets/_rels/sheet1.xml.rels",
            sheet1_rels,
        )
        archive.writestr("xl/drawings/drawing1.xml", drawing)
        archive.writestr(
            "xl/drawings/_rels/drawing1.xml.rels",
            drawing_rels,
        )
        archive.writestr("xl/media/image1.png", _png_payload(80))
        archive.writestr("xl/media/image2.png", _png_payload(120))


class PrepareFieldHoldoutTests(unittest.TestCase):
    def test_meter_id_is_derived_from_physical_instrument_folder(self) -> None:
        self.assertEqual(
            infer_meter_id(
                {
                    "source_kind": "xiangmu2",
                    "source_image": "xiangmu2/07-153/example.jpg",
                }
            ),
            "07-153",
        )

    def test_prepare_extracts_only_prediction_independent_eligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbook = root / "labels.xlsx"
            _write_fixture(workbook)
            output_dir = root / "extracted"
            manifest = root / "manifest.jsonl"
            protocol = prepare_field_holdout(
                workbook,
                output_dir,
                manifest,
                dataset_name="SyntheticFieldFixture",
            )

            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group_id"], "02-111")
            self.assertEqual(rows[0]["scale_start"], 0.0)
            self.assertEqual(rows[0]["scale_end"], 1.6)
            self.assertEqual(rows[0]["ground_truth"], 0.45)
            self.assertEqual(rows[0]["metadata"]["dial_bbox"], [0, 0, 32, 24])
            self.assertTrue(Path(rows[0]["image_path"]).is_file())

            self.assertEqual(protocol["protocol"], PREPARATION_PROTOCOL)
            self.assertTrue(protocol["confirmatory_holdout"])
            self.assertFalse(protocol["selection_uses_model_predictions"])
            self.assertEqual(protocol["included_rows"], 1)
            self.assertEqual(protocol["excluded_rows"], 1)
            self.assertEqual(
                protocol["exclusion_reasons"],
                {
                    "missing_or_invalid_ground_truth": 1,
                    "missing_or_invalid_scale_end": 1,
                },
            )
            self.assertEqual(protocol["source_summary"]["raw_images"], 2)
            self.assertTrue(
                manifest.with_name(manifest.name + ".protocol.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
