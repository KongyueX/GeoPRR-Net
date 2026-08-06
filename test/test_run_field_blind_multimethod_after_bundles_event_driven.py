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
    / "run_field_blind_multimethod_after_bundles_event_driven.ps1"
)
POWERSHELL7 = Path(r"C:\Program Files\PowerShell\7\pwsh.exe")
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


@unittest.skipUnless(POWERSHELL7.exists(), "PowerShell 7 is absent")
class FieldBlindAfterBundlesEventChainTests(unittest.TestCase):
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

    def test_restricted_and_one_shot_paths_have_no_defaults(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for parameter in (
            "MaterializedRoot",
            "MethodRoster",
            "FrontendPlan",
            "DatasetIdentity",
            "FrozenManifest",
            "FrozenLabels",
            "AuthorizationProtocol",
            "RunRoot",
            "PaperResults",
            "PaperTablesRoot",
        ):
            self.assertIn(f'[string]${parameter} = ""', text)
        self.assertIn("Assert-FormalParameters", text)
        self.assertIn("must be supplied explicitly for -StartFormal", text)

    def test_pid_start_and_exact_command_are_authenticated_before_wait(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("[int]$WaitForPid = 0", text)
        self.assertIn('[string]$WaitForStartedAtUtc = ""', text)
        self.assertIn('[string]$WaitForCommandFragment = ""', text)
        self.assertIn('[string]$WaitForCommandLineSha256 = ""', text)
        self.assertIn("Get-CimInstance Win32_Process", text)
        self.assertIn("Get-Sha256Text $CommandLine", text)
        self.assertIn("$Cim.CreationDate.ToUniversalTime()", text)
        self.assertIn("$MaterializationWrapper", text)
        self.assertIn("PID is not the canonical five-method materialization wrapper", text)
        self.assertIn("$Process.StartTime.ToUniversalTime()", text)
        self.assertIn("PID was reused before wait handle acquisition", text)
        self.assertIn(
            "Get-AuthenticatedMaterializerProcess -RequirePresent", text
        )
        authenticate = text.index(
            "$MaterializerProcess = Get-AuthenticatedMaterializerProcess -RequirePresent"
        )
        wait = text.index("$MaterializerProcess.Process.WaitForExit()")
        public = text.index('"public-materialization"')
        runtime = text.index('"runtime"')
        dataset = text.index('"dataset-bindings"')
        freeze = text.index('"-FreezeAuthorization"')
        inference = text.index('"-StartInference"')
        scoring = text.index('"-StartScoring"')
        self.assertLess(authenticate, wait)
        self.assertLess(wait, public)
        self.assertLess(public, runtime)
        self.assertLess(runtime, dataset)
        self.assertLess(dataset, freeze)
        self.assertLess(freeze, inference)
        self.assertLess(inference, scoring)

    def test_formal_paths_are_absolute_and_outputs_cannot_enter_frozen_roots(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Assert-ExplicitAbsolutePath", text)
        self.assertIn("Assert-FormalOutputIsolation", text)
        self.assertIn("must not write inside frozen", text)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
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
                    "-MaterializedRoot",
                    str(root / "frozen"),
                    "-MethodRoster",
                    str(root / "frozen" / "method_roster.json"),
                    "-FrontendPlan",
                    str(root / "frontend.json"),
                    "-DatasetIdentity",
                    str(root / "identity.json"),
                    "-FrozenManifest",
                    str(root / "manifest.jsonl"),
                    "-FrozenLabels",
                    str(root / "labels.jsonl"),
                    "-AuthorizationProtocol",
                    str(root / "frozen" / "must-not-write.json"),
                    "-RunRoot",
                    str(root / "run"),
                    "-PaperResults",
                    str(root / "paper.json"),
                    "-PaperTablesRoot",
                    str(root / "tables"),
                    "-StartFormal",
                ],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not write inside frozen", result.stderr)
            self.assertEqual(list(root.iterdir()), [])

    def test_final_chain_reverifies_prediction_and_score_seals(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        prediction = text.index('-Label "joint prediction seal"')
        scoring = text.index('"-StartScoring"')
        score_seal = text.index('-Label "completed blind score seal"')
        complete = text.index('five-method-final-blind-score-complete-v1')
        self.assertLess(prediction, scoring)
        self.assertLess(scoring, score_seal)
        self.assertLess(score_seal, complete)
        self.assertIn("same_shared_roi_for_all_methods", text)
        self.assertIn("method_or_threshold_selection_after_result", text)
        self.assertIn("summary_sha256", text)
        self.assertIn("denominator drift for role", text)

    def test_event_chain_has_no_polling_or_periodic_notifications(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        folded = text.casefold()
        self.assertIn(".waitforexit()", folded)
        self.assertNotIn("start-sleep", folded)
        self.assertNotIn("while (", folded)
        self.assertNotIn("heartbeat", folded)
        self.assertNotIn("epoch", folded)
        self.assertNotIn("暂无变化", text)
        self.assertEqual(text.count("Send-StageEvent `"), 4)
        for event in (
            "five-method-bundles-to-final-blind-inference-v1",
            "five-method-final-blind-inference-complete-to-scoring-v1",
            "five-method-final-blind-score-complete-v1",
            "five-method-final-blind-$CurrentStage-unexpected-stop-v1",
        ):
            self.assertIn(event, text)

    def test_failure_is_terminal_and_never_auto_retries(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        catch = text.rindex("} catch {")
        anomaly = text.index(
            'five-method-final-blind-$CurrentStage-unexpected-stop-v1', catch
        )
        rethrow = text.index("throw $OriginalError", anomaly)
        self.assertLess(catch, anomaly)
        self.assertLess(anomaly, rethrow)
        terminal = text[catch:]
        self.assertNotIn("Start-Sleep", terminal)
        self.assertNotIn("WaitForExit", terminal)
        self.assertNotIn("Invoke-BlindWrapper", terminal)

    @unittest.skipUnless(PYTHON.exists(), "Project Python is absent")
    def test_preflight_only_has_zero_side_effect_and_opens_no_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                    "-DatasetIdentity",
                    str(root / "do-not-open-identity.json"),
                    "-FrozenManifest",
                    str(root / "do-not-open-manifest.jsonl"),
                    "-FrozenLabels",
                    str(root / "do-not-open-labels.jsonl"),
                    "-AuthorizationProtocol",
                    str(root / "do-not-create-authorization.json"),
                    "-RunRoot",
                    str(root / "do-not-create-run"),
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
            self.assertEqual(value["audit"]["writes"], 0)
            self.assertFalse(value["audit"]["waited_for_process"])
            self.assertFalse(value["audit"]["public_materialization_opened"])
            self.assertFalse(value["audit"]["dataset_identity_opened"])
            self.assertFalse(value["audit"]["field_manifest_opened"])
            self.assertFalse(value["audit"]["field_manifest_hashed"])
            self.assertFalse(value["audit"]["field_images_opened"])
            self.assertFalse(value["audit"]["field_labels_opened"])
            self.assertFalse(value["audit"]["field_labels_hashed"])
            self.assertFalse(value["audit"]["inference_started"])
            self.assertFalse(value["audit"]["scoring_started"])
            self.assertEqual(value["audit"]["feishu_messages_sent"], 0)
            self.assertEqual(list(root.iterdir()), [])

    @unittest.skipUnless(PYTHON.exists(), "Project Python is absent")
    def test_nonzero_pid_requires_exact_identity_in_preflight(self) -> None:
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
