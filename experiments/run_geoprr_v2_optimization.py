"""Run the fixed MobileNet refinement and source-OOF routing comparison."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (20262020, 20262021, 20262022)
FOLD_SEED = 20262020


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_folds(rows: list[dict]) -> list[list[dict]]:
    by_scene = defaultdict(list)
    for row in rows:
        if row["dataset"].lower() != "syncg" or row["split"] != "train":
            raise ValueError("OOF folds require SyncG source-training rows")
        by_scene[Path(row["metadata"]["scene_name"]).stem].append(row)
    scenes = list(sorted(by_scene))
    random.Random(FOLD_SEED).shuffle(scenes)
    scenes.sort(key=lambda scene: len(by_scene[scene]), reverse=True)
    folds: list[list[dict]] = [[], [], []]
    for scene in scenes:
        min(folds, key=len).extend(by_scene[scene])
    return folds


def process_running(pid: int) -> bool:
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def compare_validation(candidate: dict, baseline: dict) -> dict:
    if candidate["samples"] != 1506 or baseline["samples"] != 1506:
        raise ValueError("validation decision requires the fixed 1506-image source cohort")
    if candidate.get("smoke_only") or baseline.get("smoke_only"):
        raise ValueError("diagnostic runs cannot decide whether to continue OOF training")
    improved = candidate["nmae"] < baseline["nmae"]
    return {"decision": "proceed_with_oof" if improved else "return_to_original_mobilenet",
            "improved": improved, "rule": "strictly lower source-validation NMAE",
            "test_used_for_decision": False, "candidate": candidate, "baseline": baseline}


class OptimizationRun:
    def __init__(self, home: Path, output: Path, seeds: list[int]) -> None:
        self.home, self.output, self.seeds = home, output, seeds
        self.repo = Path(__file__).resolve().parents[1]
        self.manifests = home / "runs/geoprr_v2/manifests"
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = output / "queue_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"steps": {}}
        self.state.update(protocol="geoprr_v2_refinement_source_oof_v1", seeds=seeds,
                          pipeline_pid=os.getpid(), status="running")
        self.save()

    def save(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.state_path, self.state)

    def step(self, name: str, command: list[str], expected: Path,
             environment_overrides: dict[str, str] | None = None) -> None:
        previous = self.state["steps"].get(name, {})
        if previous.get("status") == "complete" and expected.exists():
            return
        if previous.get("status") == "running":
            pid = previous.get("pid")
            if pid and not process_running(pid) and expected.exists():
                previous.update(status="complete", exit_code=None, finished_at=datetime.now(timezone.utc).isoformat())
                self.save()
                return
            if pid and process_running(pid):
                actual = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
                if actual != previous["command"]:
                    raise RuntimeError(f"{name}: PID {pid} no longer belongs to the recorded command")
                previous["adopted_by"] = os.getpid()
                self.save()
                print(json.dumps({"event": "adopt_running_step", "step": name, "pid": pid}), flush=True)
                while process_running(pid):
                    time.sleep(5)
                previous.update(status="complete" if expected.exists() else "failed",
                                finished_at=datetime.now(timezone.utc).isoformat(), exit_code=None)
                self.save()
                if previous["status"] != "complete":
                    raise RuntimeError(f"{name} exited without its completion output; see {previous['log']}")
                return
        log_path = self.output / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"status": "running", "command": command, "log": str(log_path),
                  "expected": str(expected), "started_at": datetime.now(timezone.utc).isoformat()}
        if environment_overrides:
            record["environment_overrides"] = environment_overrides
        self.state["steps"][name] = record
        self.save()
        print(json.dumps({"event": "start", "step": name, "log": str(log_path)}), flush=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=self.repo, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, **environment_overrides} if environment_overrides else None)
            record["pid"] = process.pid
            self.save()
            code = process.wait()
        record.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat(),
                      status="complete" if code == 0 and expected.exists() else "failed")
        self.save()
        if record["status"] != "complete":
            raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")
        print(json.dumps({"event": "complete", "step": name}), flush=True)

    def prepare(self) -> None:
        train = read_jsonl(self.manifests / "train.jsonl")
        validation = read_jsonl(self.manifests / "validation.jsonl")
        test = read_jsonl(self.manifests / "test.jsonl")
        if (len(train), len(validation), len(test)) != (14494, 1506, 4000):
            raise ValueError("the fixed experiment requires 14494 training, 1506 validation and 4000 test images")
        if any(row["dataset"].lower() != "syncg" or row["split"] != "test" for row in test):
            raise ValueError("the test cohort must contain official SyncG test rows")
        source_ids = {row["sample_id"] for row in train + validation}
        official_source_ids = {row["sample_id"] for row in read_jsonl(self.manifests / "source_train.jsonl")}
        if source_ids != official_source_ids or len(source_ids) != 16000:
            raise ValueError("training/validation do not partition the fixed official source roster")
        if source_ids & {row["sample_id"] for row in test}:
            raise ValueError("source training/validation and test image IDs overlap")
        identities = {name: [row["sample_id"] for row in population]
                      for name, population in (("train", train), ("validation", validation), ("test", test))}
        identity_path = self.output / "manifests/source_ids.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identities:
            raise ValueError("fixed source/test sample identities changed during this run")
        write_json(identity_path, identities)
        from experiments.train_geoprr_v2 import source_split_check
        source_split_check(train, validation)
        folds = source_folds(train)
        for index, heldout in enumerate(folds):
            fitting = [row for other, fold in enumerate(folds) if other != index for row in fold]
            source_split_check(fitting, heldout)
            for role, rows in (("train", fitting), ("heldout", heldout)):
                path = self.output / "manifests" / f"fold{index}_{role}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        write_json(self.output / "protocol.json", {
            "protocol": self.state["protocol"], "seeds": self.seeds, "fold_seed": FOLD_SEED,
            "folds": 3, "training_samples": len(train), "validation_samples": len(validation),
            "source_train_manifest": str(self.manifests / "train.jsonl"),
            "source_validation_manifest": str(self.manifests / "validation.jsonl"),
            "official_test_manifest": str(self.manifests / "test.jsonl"),
            "source_fold_sizes": [len(fold) for fold in folds],
            "router_training": "three-fold source-training held-out predictions",
            "selection": "source-validation NMAE; no test-based configuration selection",
            "main_configuration": "detail_ce", "epochs_per_reader": 65,
            "continuation_rule": "first detail_ce seed must improve original MobileNetV4 source-validation NMAE before OOF/routing",
            "probe_configurations": ["global_ce", "spatial_ce", "dynamic", "wide"],
            "conditions": ["native"], "external_data": "not used for fitting or selection",
        })

    def reader(self, label: str, seed: int, fold: int | None = None) -> Path:
        if label == "baseline" and seed == 20262020:
            path = self.home / "runs/geoprr_v2/mobilenetv4_conv_medium/field/seed_20262020"
            if json.loads((path / "run.json").read_text())["config"]["training"]["seed"] != seed:
                raise ValueError("historical baseline seed differs")
            return path / "best.pt"
        config = (self.repo / "experiments/configs/geoprr_v2_mobilenetv4.json" if label == "baseline"
                  else self.repo / f"experiments/configs/geoprr_v2_opt_{label}.json")
        tag = f"{label}_s{seed}" + (f"_fold{fold}" if fold is not None else "")
        directory = self.output / "readers" / tag
        complete_path = directory / "completed.json"
        expected_config = json.loads(config.read_text())
        expected_config["training"]["seed"] = seed
        done = False
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if complete["config"] != expected_config:
                raise ValueError(f"completed reader configuration differs: {tag}")
            train_path = (self.manifests / "train.jsonl" if fold is None
                          else self.output / "manifests" / f"fold{fold}_train.jsonl")
            identities = {"train": [r["sample_id"] for r in read_jsonl(train_path)],
                          "validation": [r["sample_id"] for r in read_jsonl(self.manifests / "validation.jsonl")]}
            if complete["sample_ids"] != identities:
                raise ValueError(f"completed reader data identities differ: {tag}")
            done = complete["epochs"] == 65 and not complete["smoke_only"]
        command = [sys.executable, "-m", "experiments.train_geoprr_v2", "--config", str(config),
                   "--train-manifest", str(self.manifests / "train.jsonl" if fold is None
                                           else self.output / "manifests" / f"fold{fold}_train.jsonl"),
                   "--val-manifest", str(self.manifests / "validation.jsonl"),
                   "--seed", str(seed), "--output", str(directory)]
        if not done:
            if (directory / "latest.pt").exists():
                command += ["--resume", str(directory / "latest.pt")]
            self.step(f"train_{tag}", command, complete_path)
            complete = json.loads(complete_path.read_text())
            if complete["config"] != expected_config or complete["epochs"] != 65 or complete["smoke_only"]:
                raise ValueError(f"reader did not finish the expected formal configuration: {tag}")
            actual_train = self.manifests / "train.jsonl" if fold is None else self.output / "manifests" / f"fold{fold}_train.jsonl"
            identities = {"train": [r["sample_id"] for r in read_jsonl(actual_train)],
                          "validation": [r["sample_id"] for r in read_jsonl(self.manifests / "validation.jsonl")]}
            if complete["sample_ids"] != identities:
                raise ValueError(f"reader completion identities differ: {tag}")
        elif self.state["steps"].get(f"train_{tag}", {}).get("status") == "running":
            self.state["steps"][f"train_{tag}"].update(status="complete", exit_code=None)
            self.save()
        return directory / "best.pt"

    def validation_gate(self, checkpoint: Path) -> bool:
        reference_dir = self.home / "runs/geoprr_v2/mobilenetv4_conv_medium/field/seed_20262020"
        reference_run = json.loads((reference_dir / "run.json").read_text())
        candidate_run = json.loads((checkpoint.parent / "run.json").read_text())
        if (reference_run["manifests"]["validation"] != candidate_run["manifests"]["validation"]
                or reference_run["config"]["training"]["seed"] != candidate_run["config"]["training"]["seed"]):
            raise ValueError("validation gate readers must use the same validation cohort and seed")
        reference_ids = reference_run.get("sample_ids", {}).get("validation")
        if reference_ids is None:
            reference_ids = [row["sample_id"] for row in read_jsonl(Path(reference_run["manifests"]["validation"]))]
        if reference_ids != candidate_run["sample_ids"]["validation"]:
            raise ValueError("baseline and candidate validation sample identities differ")
        decision = compare_validation(json.loads((checkpoint.parent / "best_validation.json").read_text()),
                                      json.loads((reference_dir / "best_validation.json").read_text()))
        decision.update(candidate_checkpoint=str(checkpoint), baseline_checkpoint=str(reference_dir / "best.pt"),
                        validation_sample_ids=candidate_run["sample_ids"]["validation"],
                        observed_at=datetime.now(timezone.utc).isoformat())
        write_json(self.output / "validation_decision.json", decision)
        self.state["validation_decision"] = decision["decision"]
        self.save()
        return decision["improved"]

    def evaluate(self, label: str, seed: int, checkpoint: Path, mode: str = "native",
                 router: Path | None = None, modal_radius: int = 0) -> None:
        tag = "router" if router else f"modal_r{modal_radius}" if modal_radius else mode
        directory = self.output / "evaluation" / f"{label}_s{seed}_{tag}"
        command = [sys.executable, "-m", "experiments.evaluate_geoprr_v2", "--checkpoint", str(checkpoint),
                   "--manifest", str(self.manifests / "test.jsonl"), "--output", str(directory)]
        command += ["--router", str(router)] if router else ["--expert-mode", mode]
        if modal_radius:
            command += ["--modal-radius", str(modal_radius)]
        self.step(f"test_{label}_s{seed}_{tag}", command, directory / "summary.json")

    def routed_reader(self, seed: int, checkpoint: Path) -> Path:
        caches = self.output / "caches" / f"s{seed}"
        cache_paths = []
        for fold in range(3):
            fold_checkpoint = self.reader("detail_ce", seed, fold)
            cache = caches / f"fold{fold}.npz"
            self.step(f"cache_s{seed}_fold{fold}", [sys.executable, "-m", "experiments.cache_geoprr_v2_experts",
                      "--checkpoint", str(fold_checkpoint), "--manifest", str(self.output / "manifests" / f"fold{fold}_heldout.jsonl"),
                      "--output", str(cache), "--role", "oof_train", "--fold-id", str(fold)], cache)
            cache_paths.append(cache)
        validation_cache = caches / "validation.npz"
        self.step(f"cache_s{seed}_validation", [sys.executable, "-m", "experiments.cache_geoprr_v2_experts",
                  "--checkpoint", str(checkpoint), "--manifest", str(self.manifests / "validation.jsonl"),
                  "--output", str(validation_cache), "--role", "validation", "--fold-id", "full"], validation_cache)
        directory = self.output / "routers" / f"s{seed}"
        command = [sys.executable, "-m", "experiments.train_geoprr_v2_router", "--validation-cache", str(validation_cache),
                   "--seed", str(seed), "--output", str(directory)]
        for cache in cache_paths:
            command += ["--train-cache", str(cache)]
        self.step(f"router_s{seed}", command, directory / "router.pt")
        return directory / "router.pt"

    def run(self) -> None:
        self.prepare()
        first = self.seeds[0]
        repaired = self.reader("global_ce", first)
        self.evaluate("global_ce", first, repaired)
        for seed in self.seeds:
            main = self.reader("detail_ce", seed)
            if seed == first and not self.validation_gate(main):
                self.state.update(status="validation_rejected", deferred_work=["OOF", "routing", "remaining_previous_matrix"],
                                  next_direction="optimize severe errors from the original MobileNetV4")
                self.save()
                from experiments.export_geoprr_v2_optimization import export
                export(self.output, self.output / "exports")
                print(json.dumps({"event": "validation_rejected", "next": self.state["next_direction"]}), flush=True)
                return
            router = self.routed_reader(seed, main)
            for mode in ("field", "direct", "mean"):
                self.evaluate("detail_ce", seed, main, mode)
            self.evaluate("detail_ce", seed, main, router=router)
            for label in ("baseline", "raw_spatial", "multitask"):
                self.evaluate(label, seed, self.reader(label, seed))
            if seed == first:
                for label in ("spatial_ce", "dynamic", "wide"):
                    self.evaluate(label, seed, self.reader(label, seed))
        export_dir = self.output / "exports"
        self.step("export", [sys.executable, "-m", "experiments.export_geoprr_v2_optimization",
                  "--run", str(self.output), "--output", str(export_dir)], export_dir / "summary.json")
        self.state["status"] = "complete"
        self.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("GEOPRR_HOME", "/root/autodl-tmp/geoprrnet")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.home / "runs/geoprr_v2_optimization"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(args.home, output, list(SEEDS))
        try:
            run.run()
        except Exception:
            run.state["status"] = "failed"
            run.save()
            raise


if __name__ == "__main__":
    main()
