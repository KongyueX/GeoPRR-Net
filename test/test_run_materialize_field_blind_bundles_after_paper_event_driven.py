from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments"
    / "run_materialize_field_blind_bundles_after_paper_event_driven.ps1"
)
POWERSHELL7 = Path(r"C:\Program Files\PowerShell\7\pwsh.exe")
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


class BundleMaterializationEventWrapperTests(unittest.TestCase):
    def test_powershell_ast_is_valid(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            "[void][System.Management.Automation.Language.Parser]::ParseFile("
            f"'{SCRIPT}', [ref]$tokens, [ref]$errors); "
            "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
        )
        result = subprocess.run(
            [str(POWERSHELL7), "-NoLogo", "-NoProfile", "-Command", command],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrapper_is_event_driven_metadata_only_and_sends_no_notification(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("[switch]$PreflightOnly", text)
        self.assertIn("[switch]$StartMaterialization", text)
        self.assertIn("$PaperWaiterProcessId = 25424", text)
        self.assertIn('"2026-08-06T18:58:48.7736440Z"', text)
        self.assertIn("Get-AuthenticatedPaperWaiter", text)
        self.assertIn("Start-Sleep -Seconds 3", text)
        self.assertNotIn("Get-ChildItem", text)
        self.assertNotIn("Get-FileHash", text)
        self.assertNotIn("send_feishu_progress", text.casefold())
        self.assertNotIn("freeze-authorization", text.casefold())

    @unittest.skipUnless(
        POWERSHELL7.exists() and PYTHON.exists(),
        "PowerShell 7 or project Python absent",
    )
    def test_preflight_only_reports_exact_missing_and_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / "spec.json"
            garc = root / "garc.json"
            paper = root / "paper.json"
            seal = root / "seal.json"
            output = root / "output"
            result = subprocess.run(
                [
                    str(POWERSHELL7),
                    "-NoLogo",
                    "-NoProfile",
                    "-File",
                    str(SCRIPT),
                    "-ProjectRoot",
                    str(ROOT),
                    "-Python",
                    str(PYTHON),
                    "-PaperWaiterProcessId",
                    "2147483646",
                    "-Spec",
                    str(spec),
                    "-GarcSummary",
                    str(garc),
                    "-PaperSummary",
                    str(paper),
                    "-PaperSeal",
                    str(seal),
                    "-OutputRoot",
                    str(output),
                    "-PreflightOnly",
                ],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["status"], "preflight_only_complete")
            self.assertFalse(value["scientific_preflight"]["ready"])
            self.assertEqual(
                [row["requirement"] for row in value["scientific_preflight"]["missing"]],
                [
                    "materialization_spec",
                    "garc_summary",
                    "paper_results_summary",
                    "paper_results_seal",
                ],
            )
            self.assertFalse(value["formal_materialization_started"])
            self.assertEqual(value["feishu_messages_sent"], 0)
            self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless(
        POWERSHELL7.exists() and PYTHON.exists(),
        "PowerShell 7 or project Python absent",
    )
    def test_start_refuses_missing_spec_before_waiting_for_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / "spec.json"
            output = root / "output"
            result = subprocess.run(
                [
                    str(POWERSHELL7),
                    "-NoLogo",
                    "-NoProfile",
                    "-File",
                    str(SCRIPT),
                    "-ProjectRoot",
                    str(ROOT),
                    "-Python",
                    str(PYTHON),
                    "-PaperWaiterProcessId",
                    "25424",
                    "-Spec",
                    str(spec),
                    "-GarcSummary",
                    str(root / "garc.json"),
                    "-PaperSummary",
                    str(root / "paper.json"),
                    "-PaperSeal",
                    str(root / "seal.json"),
                    "-OutputRoot",
                    str(output),
                    "-StartMaterialization",
                ],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("materialization spec is absent", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
