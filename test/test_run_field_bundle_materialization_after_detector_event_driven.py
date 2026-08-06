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
    / "run_field_bundle_materialization_after_detector_event_driven.ps1"
)
POWERSHELL7 = Path(r"C:\Program Files\PowerShell\7\pwsh.exe")
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


@unittest.skipUnless(POWERSHELL7.exists(), "PowerShell 7 is absent")
class FieldBundleAfterDetectorEventWrapperTests(unittest.TestCase):
    def test_powershell_ast_is_valid(self) -> None:
        command = (
            "$tokens=$null; $errors=$null; "
            "[void][System.Management.Automation.Language.Parser]::ParseFile("
            f"'{SCRIPT}', [ref]$tokens, [ref]$errors); "
            "if ($errors.Count) { "
            "$errors | ForEach-Object { Write-Error $_ }; exit 1 }"
        )
        result = subprocess.run(
            [str(POWERSHELL7), "-NoLogo", "-NoProfile", "-Command", command],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_formal_wait_identity_is_explicit_and_fail_closed(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("[int]$WaitForPid = 0", text)
        self.assertIn('[string]$WaitForStartedAtUtc = ""', text)
        self.assertIn("[string]$WaitForCommandFragment", text)
        self.assertNotIn("24440", text)
        self.assertNotIn("2026-08-06T20:08:29.6954450Z", text)

        self.assertIn("Get-CimInstance Win32_Process", text)
        self.assertIn('$CommandLine.IndexOf(', text)
        self.assertIn("$Cim.CreationDate.ToUniversalTime()", text)
        self.assertIn("frozen identity", text)
        self.assertIn("Get-Process -Id $WaitForPid", text)
        self.assertIn("Get-AuthenticatedDetectorProcess -RequirePresent", text)

        authenticate_index = text.index(
            "$Detector = Get-AuthenticatedDetectorProcess -RequirePresent"
        )
        wait_index = text.index("$Detector.Process.WaitForExit()")
        frontend_index = text.index("$null = Test-FrontendAuthority")
        materialize_index = text.index(
            '$CurrentStage = "materialize-five-method-bundles"'
        )
        self.assertLess(authenticate_index, wait_index)
        self.assertLess(wait_index, frontend_index)
        self.assertLess(frontend_index, materialize_index)

    def test_chain_is_event_driven_and_does_not_launch_blind_inference(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        folded = text.casefold()
        self.assertIn(".waitforexit()", folded)
        self.assertNotIn("start-sleep", folded)
        self.assertNotIn("wait-event", folded)
        self.assertNotIn("register-objectevent", folded)
        self.assertNotIn("filesystemwatcher", folded)
        self.assertNotIn("run_field_blind_multimethod_final_once", folded)
        self.assertNotIn("run_garc_field_blind_final_once", folded)
        self.assertNotIn("field_blind_multimethod.py", folded)
        self.assertNotIn('"run-once"', folded)
        self.assertNotIn('"score-once"', folded)
        self.assertIn("field_manifest_opened = $false", text)
        self.assertIn("field_images_opened = $false", text)
        self.assertIn("field_labels_opened = $false", text)
        self.assertIn("blind_inference_started = $false", text)

    def test_notifications_are_only_stage_complete_or_anomaly(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '"syncg-public-frontend-to-five-method-bundle-materialization-v1"',
            text,
        )
        self.assertIn(
            '"five-method-public-bundles-materialized-complete-v1"', text
        )
        self.assertIn(
            '"five-method-bundle-$CurrentStage-unexpected-stop-v1"', text
        )
        self.assertEqual(text.count("Send-StageEvent `"), 3)
        self.assertNotIn("epoch", text.casefold())
        self.assertNotIn("heartbeat", text.casefold())
        self.assertNotIn("暂无变化", text)
        self.assertGreaterEqual(text.count('-Eta "'), 3)

    @unittest.skipUnless(PYTHON.exists(), "Project Python is absent")
    def test_preflight_only_is_zero_side_effect_and_reports_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builder_root = root / "builder"
            materialized_root = root / "materialized"
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
                    "-WaitForPid",
                    "0",
                    "-GarcSummary",
                    str(root / "missing-garc.json"),
                    "-PaperSummary",
                    str(root / "missing-paper.json"),
                    "-PaperSeal",
                    str(root / "missing-seal.json"),
                    "-FrontendPlan",
                    str(root / "missing-frontend.json"),
                    "-BuilderOutputRoot",
                    str(builder_root),
                    "-MaterializedOutputRoot",
                    str(materialized_root),
                    "-PreflightOnly",
                ],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["status"], "preflight_only_complete")
            self.assertFalse(
                value["detector_process"]["authenticated_and_running"]
            )
            self.assertEqual(
                value["scientific_preflight"]["missing"],
                ["garc_summary", "paper_summary", "paper_seal", "frontend_plan"],
            )
            self.assertEqual(value["audit"]["writes"], 0)
            self.assertFalse(value["audit"]["waited_for_process"])
            self.assertFalse(value["audit"]["formal_chain_started"])
            self.assertFalse(value["audit"]["blind_inference_started"])
            self.assertEqual(value["audit"]["feishu_messages_sent"], 0)
            self.assertFalse(builder_root.exists())
            self.assertFalse(materialized_root.exists())
            self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless(PYTHON.exists(), "Project Python is absent")
    def test_nonzero_pid_requires_explicit_start_time_even_in_preflight(self) -> None:
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
                "-WaitForPid",
                str(2_147_483_646),
                "-PreflightOnly",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WaitForStartedAtUtc", result.stderr)


if __name__ == "__main__":
    unittest.main()
