"""Run the fixed MobileNet refinement and source-OOF routing comparison."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (20262020, 20262021, 20262022)
FOLD_SEED = 20262020


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


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

    def step(self, name: str, command: list[str], expected: Path) -> None:
        previous = self.state["steps"].get(name, {})
        if previous.get("status") == "complete" and expected.exists():
            return
        if previous.get("status") == "running":
            pid = previous.get("pid")
            if pid and Path(f"/proc/{pid}/cmdline").exists():
                raise RuntimeError(f"{name} still has a process ({pid}); inspect it before resuming")
        log_path = self.output / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"status": "running", "command": command, "log": str(log_path),
                  "expected": str(expected), "started_at": datetime.now(timezone.utc).isoformat()}
        self.state["steps"][name] = record
        self.save()
        print(json.dumps({"event": "start", "step": name, "log": str(log_path)}), flush=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=self.repo, stdout=log, stderr=subprocess.STDOUT)
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
        return directory / "best.pt"

    def evaluate(self, label: str, seed: int, checkpoint: Path, mode: str = "native",
                 router: Path | None = None) -> None:
        mode = "router" if router else mode
        directory = self.output / "evaluation" / f"{label}_s{seed}_{mode}"
        command = [sys.executable, "-m", "experiments.evaluate_geoprr_v2", "--checkpoint", str(checkpoint),
                   "--manifest", str(self.manifests / "test.jsonl"), "--output", str(directory)]
        command += ["--router", str(router)] if router else ["--expert-mode", mode]
        self.step(f"test_{label}_s{seed}_{mode}", command, directory / "summary.json")

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
