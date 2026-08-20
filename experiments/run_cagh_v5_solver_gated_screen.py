"""Run the single preregistered public inner-screen for solver-gated CAGH-V5.

This runner is deliberately separate from the frozen V2 outer-holdout runner.
P1 imports the already audited P0 inner-train PEPD terminal byte-for-byte,
hardens the parent publication/resume state machine, and then trains/evaluates
only the preregistered 579/73 group split inside the old public fit partition.
It has no command that can evaluate the old 1,625-image common holdout or any
field/sealed/confirmatory data.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import ctypes
import gc
import functools
import hashlib
import inspect
import json
import math
import os
import platform
import random
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from experiments.cagh_v5_canonical_pepd_pair import (
    CanonicalPEPDPairDataset,
    tensor_receipt,
)
from experiments.cagh_v5_solver_gated_residual import (
    ARM_NAMES,
    FULL_GATED,
    NO_MASK_RESIDUAL,
    NO_PEPD_RESIDUAL,
    SOLVER_CORE,
    CAGHV5SolverGatedResidual,
    solver_gated_progress_loss,
)
from experiments.cagh_v5_source_ablation_heads import NO_PEPD_DIRECTION
from experiments.cagh_v5_unified_model import (
    CAGHV5UnifiedOutputs,
    CAGHV5UnifiedModel,
    cagh_v5_unified_loss,
)
from experiments.probabilistic_pivot_direction import (
    IMAGENET_WEIGHTS,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
    probabilistic_direction_loss,
    soft_pivot_coordinates,
    transform_pivot_direction,
)
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    CanonicalTightROIDataset,
    PhotoAugmentation,
    PublicRecord,
)
from experiments.vdn_baseline import (
    VDNSample,
    grouped_train_val_split,
    load_syncg_manifest,
    seed_worker,
    sha256_file,
    syncg_sample_ids_sha256,
)


PROTOCOL = "cagh_v5_solver_gated_inner_screen_v2"
PARENT_TRAINING_ARM = "solver_core_parent"
DEFAULT_OUTPUT_ROOT = Path(r"C:\pointer_read\cagh_v5_solver_gated_screen_v2")
PROTOCOL_PATH = DEFAULT_OUTPUT_ROOT / "protocol.json"
P0_ARCHIVE_ROOT = Path(
    r"C:\pointer_read\archives\cagh_v5_solver_gated_p0_protocol_6aa11c6976def03f"
)
P0_ROOT_SNAPSHOT = P0_ARCHIVE_ROOT / "p0_root"
P0_SOURCE_SNAPSHOT = P0_ARCHIVE_ROOT / "source_snapshot"
P0_PROTOCOL_PATH = P0_ROOT_SNAPSHOT / "protocol.json"
P0_PROTOCOL_FREEZE_RECEIPT = P0_ROOT_SNAPSHOT / "protocol_freeze_receipt.json"
P0_PREFLIGHT_PATH = P0_ROOT_SNAPSHOT / "preflight.json"
P0_PREFLIGHT_RECEIPT = P0_ROOT_SNAPSHOT / "preflight_receipt.json"
P0_PEPD_RUN = P0_ROOT_SNAPSHOT / "runs" / "pepd_inner_seed_20261910"
P0_PEPD_JOURNAL = P0_PEPD_RUN / "journal.pt"
P0_PEPD_TERMINAL = P0_PEPD_RUN / "terminal.pt"
P0_PEPD_SUMMARY = P0_PEPD_RUN / "summary.json"
P0_ARCHIVE_RECEIPT = P0_ARCHIVE_ROOT / "archive_receipt.json"
P0_PROTOCOL_ID = "cagh_v5_solver_gated_inner_screen_v1"
EXPECTED_P0_PROTOCOL_CANONICAL_SHA256 = (
    "6aa11c6976def03fe569dd1c53bf4737e8af6cc3f179e6cec29c7739734e45b7"
)
EXPECTED_P0_PROTOCOL_FILE_SHA256 = (
    "a337cc14d3ebd6abac3a754c1d76214bdb73e89d2fb9d98c457ce195c0462f08"
)
EXPECTED_P0_PROTOCOL_FREEZE_RECEIPT_SHA256 = (
    "3cd0e1395375ba50f9b7c1c23d75e61cdceb46ebc83721109d83066ea2528da2"
)
EXPECTED_P0_PREFLIGHT_SHA256 = (
    "2c3985c972ae66512882ec195c8e8a8181be436c99c6315704448b27f8534546"
)
EXPECTED_P0_PREFLIGHT_RECEIPT_SHA256 = (
    "954b09f4345b8d2e6e413dc9d7df2246368056241d868a831003afb46fbb9fe8"
)
EXPECTED_P0_PEPD_JOURNAL_SHA256 = (
    "980155f5a43e6ba58c89c3ecc3d5faa90bdc93eef503886dfd8670e3d07c0168"
)
EXPECTED_P0_PEPD_TERMINAL_SHA256 = (
    "33805aae9584d9c286469f349d4e2747050e2882f1ffdc47cc697d1c11100620"
)
EXPECTED_P0_PEPD_SUMMARY_SHA256 = (
    "87c626ad446b65f1003fe10cd9c2d50f90fc97285cd1e6fca057ac49eddef91d"
)
EXPECTED_P0_PEPD_MODEL_STATE_SHA256 = (
    "5273f90dc965b8632a97724abc89af6aa1aaa66c49f16871034ed96b489db320"
)
EXPECTED_P0_SOURCE_MANIFEST_SHA256 = (
    "55bc6c0e72d8c7f66130f4386e97b8b16a87e3ab52a789fd482567054221289b"
)
EXPECTED_P0_ARCHIVE_TREE_SHA256 = (
    "8d2172e4b63123b699a3c65ee4f716eeb8720aeafb3ab0300881630b6085eb5e"
)
EXPECTED_P0_ARCHIVE_CANONICAL_TREE_SHA256 = (
    "1e9bddcc1f1fe828f7cb0f91c0513732ed11b11741ce13d5b179fb6043e329f5"
)
EXPECTED_P0_ARCHIVE_RECEIPT_SHA256 = (
    "0bffbe2156274889425f36df2f114d496c5aab55287fe807ed783b72f7a90c10"
)
EXPECTED_P0_ARCHIVE_ENTRY_COUNT = 64
EXPECTED_P0_ARCHIVE_ENTRY_ROSTER_SHA256 = (
    "c89b196b30b640f23b8b028de79793fe0478672cc7932c059d9408ff2a5de021"
)
PUBLIC_MANIFEST = PROJECT_ROOT / "artifacts/manifests/syncg_train.jsonl"
PUBLIC_MANIFEST_PROTOCOL = PUBLIC_MANIFEST.with_name(
    PUBLIC_MANIFEST.name + ".protocol.json"
)
CONTENT_INVENTORY = (
    PROJECT_ROOT / "artifacts/protocols/vdn_phase2_syncg_train_content_inventory_v1.json"
)
V2_PROTOCOL = PROJECT_ROOT / "experiments/cagh_v5_source_ablation_protocol.json"
V2_SOURCE_STATISTICS = Path(
    r"C:\pointer_read\cagh_v5_unified_terminal_v2\source_matrix_statistics.json"
)
EXPECTED_PUBLIC_MANIFEST_SHA256 = (
    "429e4bc24515b4bf7a1d6fb638210e653872d287ba1edc0c1d6e193cb99d28ca"
)
EXPECTED_PUBLIC_MANIFEST_PROTOCOL_SHA256 = (
    "315e17ac8aba46d00f84a0060dba145d7e036fa096bd26423dbd34a170600c59"
)
EXPECTED_CONTENT_INVENTORY_SHA256 = (
    "16cfbfbd19c635f34696a06059fdb7a1e096f99b1a48341ae03b3244f31db92e"
)
EXPECTED_V2_PROTOCOL_SHA256 = (
    "e07823d5b8a84b23e36de6c65e0e1fd7938b7f59370a9064568e43f74f44f916"
)
EXPECTED_V2_SOURCE_STATISTICS_SHA256 = (
    "ddd3c7f0a5e1337cfd7896aff401356019c4a035444baf85f22c17f82909badb"
)
EXPECTED_IMAGENET_FILENAME = "resnet18-f37072fd.pth"
EXPECTED_IMAGENET_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)

OUTER_SPLIT_SEED = 20260720
INNER_SPLIT_SALT = "cagh_v5_gate_screen_v1"
PEPD_SEED = 20261910
PARENT_SEED = 20261920
GAIN_SEED = 20261930
EVALUATION_SEED = 20261820
STRESS_EPOCH = 314
PARENT_GRADIENT_POLICY = (
    "all trainable gradients required finite; clip error_if_nonfinite"
)
GAIN_GRADIENT_POLICY = (
    "all enabled gain gradients required finite; clip error_if_nonfinite"
)
GPU_PROCESS_FILTER_POLICY = (
    "TCC/non-WDDM: every compute-app row; WDDM: only explicit graphics "
    "allowlist rows with N/A memory are ignored; always exclude current pid"
)
WDDM_GRAPHICS_ALLOWLIST = (
    "applicationframehost.exe",
    "chatgpt.exe",
    "chrome.exe",
    "code.exe",
    "dwm.exe",
    "explorer.exe",
    "flclash.exe",
    "lockapp.exe",
    "msedge.exe",
    "msedgewebview2.exe",
    "nvidia overlay.exe",
    "searchapp.exe",
    "shellexperiencehost.exe",
    "startmenuexperiencehost.exe",
    "systemsettings.exe",
    "textinputhost.exe",
    "todesk.exe",
)
EXPECTED_PUBLIC = (16_000, 725)
EXPECTED_OUTER_FIT = (14_375, 652)
EXPECTED_OUTER_HOLDOUT = (1_625, 73)
EXPECTED_INNER_TRAIN = (12_847, 579)
EXPECTED_INNER_SCREEN = (1_528, 73)
EXPECTED_HASHES = {
    "outer_fit_ids_sha256": "b38e477fdf8667bc012023731756aa4fa275fd6e931a87bc065fec9d8ef88979",
    "outer_holdout_ids_sha256": "7550cf807f6669723c8a58cf80c7e1046af4aaea8bacfebc781dcb70d899fca2",
    "outer_fit_groups_sha256": "e936e02292f7b550679026499ee106db974f48ba25a0278131b0335c4141b87e",
    "outer_holdout_groups_sha256": "16e950f77e2f7929a6dc9a0abd06fbc454e6dba5a5f40c6f4003c0c0a2b67bbc",
    "inner_train_ids_sha256": "1b4db208a48604774531e79318229f331e646116feace074607805574772ba77",
    "inner_screen_ids_sha256": "07e5ff780eff9eff3a29104dd4e998e09d788f86eb32d8df87bbc9fb7a5ff570",
    "inner_train_groups_sha256": "efe0cf5e928f87139e104d0f343dbca4392c6d25b6965a9f2bdf94005d2fd5ee",
    "inner_screen_groups_sha256": "6554a27680cff75ff39607e37f33580bc2cb10a22f4f954919382276fc308bae",
}
FORBIDDEN_TOKENS = {"field", "test", "sealed", "confirmatory"}
LABEL_TOKENS = {
    "target",
    "ground_truth",
    "gt",
    "error",
    "scale",
    "reading",
    "bbox",
    "annotation",
    "label",
}
SOURCE_ROOTS = (
    "experiments/run_cagh_v5_solver_gated_screen.py",
    "experiments/cagh_v5_solver_gated_residual.py",
    "experiments/cagh_v5_canonical_pepd_pair.py",
    "experiments/aggregate_cagh_v5_solver_gated_screen.py",
)
TEST_SOURCE_PATHS = (
    "test/test_cagh_net.py",
    "test/test_cagh_v5_unified_model.py",
    "test/test_cagh_v5_solver_gated_residual.py",
    "test/test_cagh_v5_canonical_pepd_pair.py",
    "test/test_run_cagh_v5_solver_gated_screen.py",
    "test/test_aggregate_cagh_v5_solver_gated_screen.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_finite_json(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite_json(nested, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _require_finite_json(nested, location=f"{location}[{index}]")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"non-finite JSON number: {location}")


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    _require_finite_json(value, location=str(path))
    return value


def strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"blank JSONL row: {path}:{line_number}")
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            _require(isinstance(value, dict), f"non-object JSONL row: {path}:{line_number}")
            _require_finite_json(value, location=f"{path}:{line_number}")
            rows.append(value)
    _require(bool(rows), f"empty JSONL: {path}")
    return rows


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(canonical_json_bytes(dict(value)) + b"\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a small JSON marker once without overwrite semantics."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(dict(row)) + b"\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def torch_new(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a torch payload once; a crash leaves explicit forensic evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        torch.save(dict(value), handle)
        handle.flush()
        os.fsync(handle.fileno())


def copy_file_new(source: Path, destination: Path) -> None:
    """Copy bytes into a new file without overwrite or hard-link semantics."""

    source_path = Path(source)
    target = Path(destination)
    _require(source_path.is_file() and not _is_reparse(source_path),
             f"copy source is not a regular pinned file: {source_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    _require(sha256_file(target) == sha256_file(source_path),
             f"copied byte hash drift: {target}")


def _is_reparse(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _guard_output_root(path: Path, *, allow_create: bool) -> Path:
    candidate = Path(path)
    _require(candidate.is_absolute(), "output root must be absolute")
    _require(candidate.drive.casefold() == "c:", "all large artifacts must be on C:")
    _require(
        not ({part.casefold() for part in candidate.parts} & FORBIDDEN_TOKENS),
        "restricted namespace in output root",
    )
    existing = candidate
    while not existing.exists():
        _require(existing != existing.parent, "output root has no existing ancestor")
        existing = existing.parent
    for parent in (existing, *existing.parents):
        _require(not parent.is_symlink() and not _is_reparse(parent), f"reparse output ancestor: {parent}")
    resolved = candidate.resolve(strict=False)
    _require(resolved != Path(resolved.anchor), "drive-root output forbidden")
    if not allow_create:
        _require(resolved.is_dir(), f"output root missing: {resolved}")
    return resolved


def _guard_public_path(path: Path, root: Path, *, label: str) -> Path:
    value = Path(path)
    _require(value.is_absolute(), f"{label} must be absolute")
    _require(
        not ({part.casefold() for part in value.parts} & FORBIDDEN_TOKENS),
        f"{label} entered restricted namespace",
    )
    declared_root = Path(root)
    _require(declared_root.is_absolute(), f"{label} public root must be absolute")
    lexical_value = Path(os.path.abspath(os.fspath(value)))
    lexical_root = Path(os.path.abspath(os.fspath(declared_root)))
    _require(lexical_value.is_relative_to(lexical_root), f"{label} escaped public root")
    current = lexical_value
    while True:
        _require(current.exists(), f"{label} lexical component missing")
        _require(not current.is_symlink() and not _is_reparse(current),
                 f"{label} reparse component")
        if current == lexical_root:
            break
        _require(current != current.parent, f"{label} lexical path escaped public root")
        current = current.parent
    # The declared public root itself must not sit below a hidden junction.
    for ancestor in lexical_root.parents:
        _require(not ancestor.is_symlink() and not _is_reparse(ancestor),
                 f"{label} public-root ancestor is reparse")
    resolved = lexical_value.resolve(strict=True)
    public_root = lexical_root.resolve(strict=True)
    _require(resolved.is_relative_to(public_root), f"{label} escaped public root")
    return resolved


def _module_source_path(module: str) -> Path | None:
    candidate = PROJECT_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if candidate.is_file():
        return candidate.resolve()
    package = PROJECT_ROOT.joinpath(*module.split("."), "__init__.py")
    return package.resolve() if package.is_file() else None


def _local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
            if node.module in ("experiments", "utils"):
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        for module in modules:
            if module.startswith(("experiments", "utils")):
                source = _module_source_path(module)
                if source is not None:
                    found.add(source)
    return found


def source_manifest() -> dict[str, str]:
    pending = [PROJECT_ROOT / value for value in SOURCE_ROOTS]
    visited: set[Path] = set()
    while pending:
        path = pending.pop().resolve(strict=True)
        _require(path.is_relative_to(PROJECT_ROOT), f"source escaped project: {path}")
        if path in visited:
            continue
        visited.add(path)
        pending.extend(sorted(_local_imports(path) - visited))
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in sorted(visited, key=lambda item: item.as_posix().casefold())
    }


def environment_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "opencv": str(cv2.__version__),
        "numpy": str(np.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _canonical_imagenet_initialization() -> Path:
    _require(
        Path(IMAGENET_WEIGHTS.url).name == EXPECTED_IMAGENET_FILENAME,
        "torchvision ImageNet weight identity drift",
    )
    path = (
        Path(torch.hub.get_dir()) / "checkpoints" / EXPECTED_IMAGENET_FILENAME
    ).resolve()
    _require(path.is_file(), f"missing pinned ImageNet initialization: {path}")
    _require(
        sha256_file(path) == EXPECTED_IMAGENET_SHA256,
        "pinned ImageNet initialization SHA256 drift",
    )
    return path


def _expected_test_sources() -> dict[str, str]:
    return {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in TEST_SOURCE_PATHS
    }


def _verify_static_external_artifacts() -> None:
    expected = {
        PUBLIC_MANIFEST: EXPECTED_PUBLIC_MANIFEST_SHA256,
        PUBLIC_MANIFEST_PROTOCOL: EXPECTED_PUBLIC_MANIFEST_PROTOCOL_SHA256,
        CONTENT_INVENTORY: EXPECTED_CONTENT_INVENTORY_SHA256,
        V2_PROTOCOL: EXPECTED_V2_PROTOCOL_SHA256,
        V2_SOURCE_STATISTICS: EXPECTED_V2_SOURCE_STATISTICS_SHA256,
    }
    for path, digest in expected.items():
        _require(path.is_file(), f"missing pinned external artifact: {path}")
        _require(
            sha256_file(path) == digest,
            f"pinned external artifact SHA256 drift: {path}",
        )
    _canonical_imagenet_initialization()


def configure_determinism(seed: int) -> None:
    _require(os.environ.get("PYTHONHASHSEED") == "20260807", "PYTHONHASHSEED drift")
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG drift",
    )
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _ids_hash(values: Sequence[str]) -> str:
    return canonical_sha256(sorted(map(str, values)))


@dataclass(frozen=True)
class ScreenDiscovery:
    inner_train: tuple[PublicRecord, ...]
    inner_screen: tuple[PublicRecord, ...]
    identity: Mapping[str, Any]


def _weighted_records(values: Sequence[VDNSample], partition: str) -> tuple[PublicRecord, ...]:
    counts = Counter(str(value.group_id) for value in values)
    return tuple(
        PublicRecord(
            sample=value,
            group_weight=len(values) / (len(counts) * counts[str(value.group_id)]),
            partition=partition,
        )
        for value in values
    )


def _derive_inner(samples: Sequence[VDNSample]) -> ScreenDiscovery:
    ids = [str(value.sample_id) for value in samples]
    groups = sorted({str(value.group_id) for value in samples})
    _require((len(ids), len(groups)) == EXPECTED_OUTER_FIT, "fit-only inventory drift")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{INNER_SPLIT_SALT}:{group}".encode("utf-8")
        ).digest(),
    )
    screen_groups = set(ordered[: EXPECTED_INNER_SCREEN[1]])
    inner_train = [value for value in samples if str(value.group_id) not in screen_groups]
    inner_screen = [value for value in samples if str(value.group_id) in screen_groups]
    train_ids = [str(value.sample_id) for value in inner_train]
    screen_ids = [str(value.sample_id) for value in inner_screen]
    train_groups = [str(value.group_id) for value in inner_train]
    selected_groups = [str(value.group_id) for value in inner_screen]
    identity = {
        "outer_fit_samples": len(samples),
        "outer_fit_groups": len(groups),
        "inner_train_samples": len(inner_train),
        "inner_train_groups": len(set(train_groups)),
        "inner_screen_samples": len(inner_screen),
        "inner_screen_groups": len(set(selected_groups)),
        "sample_overlap": len(set(train_ids) & set(screen_ids)),
        "group_overlap": len(set(train_groups) & set(selected_groups)),
        "outer_fit_ids_sha256": _ids_hash(ids),
        "outer_fit_groups_sha256": _ids_hash(groups),
        "inner_train_ids_sha256": _ids_hash(train_ids),
        "inner_screen_ids_sha256": _ids_hash(screen_ids),
        "inner_train_groups_sha256": _ids_hash(list(set(train_groups))),
        "inner_screen_groups_sha256": _ids_hash(list(set(selected_groups))),
        "split_salt": INNER_SPLIT_SALT,
        "split_hash_encoding": "sha256(utf8(salt + ':' + group_id))",
    }
    expected = {
        "outer_fit_ids_sha256": EXPECTED_HASHES["outer_fit_ids_sha256"],
        "outer_fit_groups_sha256": EXPECTED_HASHES["outer_fit_groups_sha256"],
        "inner_train_ids_sha256": EXPECTED_HASHES["inner_train_ids_sha256"],
        "inner_screen_ids_sha256": EXPECTED_HASHES["inner_screen_ids_sha256"],
        "inner_train_groups_sha256": EXPECTED_HASHES["inner_train_groups_sha256"],
        "inner_screen_groups_sha256": EXPECTED_HASHES["inner_screen_groups_sha256"],
    }
    _require((len(inner_train), len(set(train_groups))) == EXPECTED_INNER_TRAIN, "inner-train count drift")
    _require((len(inner_screen), len(set(selected_groups))) == EXPECTED_INNER_SCREEN, "inner-screen count drift")
    _require(identity["sample_overlap"] == identity["group_overlap"] == 0, "inner split overlap")
    for name, value in expected.items():
        _require(identity[name] == value, f"{name} drift")
    return ScreenDiscovery(
        inner_train=_weighted_records(inner_train, "inner_train"),
        inner_screen=_weighted_records(inner_screen, "inner_screen"),
        identity=identity,
    )


def augmentation_config() -> dict[str, Any]:
    return {
        "brightness_probability": 0.8,
        "brightness_delta": 0.16,
        "contrast_probability": 0.8,
        "contrast_min": 0.65,
        "contrast_max": 1.4,
        "gamma_probability": 0.45,
        "gamma_min": 0.65,
        "gamma_max": 1.55,
        "blur_probability": 0.35,
        "blur_sigma_max": 1.6,
        "noise_probability": 0.3,
        "noise_sigma_max": 10.0,
        "jpeg_probability": 0.35,
        "jpeg_quality_min": 50,
        "jpeg_quality_max": 94,
        "perspective_probability": 0.35,
        "perspective_fraction_max": 0.035,
        "boundary_trim_probability": 0.35,
        "boundary_trim_fraction_max": 0.06,
    }


def promotion_contract() -> dict[str, Any]:
    return {
        "bootstrap": {
            "unit": "group_cluster",
            "iterations": 20_000,
            "seed": 20261822,
            "difference": "full_gated_minus_comparison; negative favors full_gated",
            "combined": "same sampled group carries all clean and stress rows; image-micro",
        },
        "full_vs_solver_core": {
            "stress_relative_nmae_improvement_min": 0.05,
            "stress_delta_ci_upper_max": 0.0,
            "clean_delta_point_max": 0.0005,
            "clean_noninferiority_margin": 0.001,
            "combined_delta_point_max": 0.0,
            "combined_delta_ci_upper_max": 0.0,
            "stress_group_macro_relative_improvement_min": 0.03,
            "clean_group_macro_delta_max": 0.001,
            "coverage_drop_max_each_condition": 0.001,
            "failure_increase_max_each_condition": 1,
            "clean_p95_ratio_max": 1.05,
            "clean_p99_ratio_max": 1.10,
            "stress_p95_ratio_max": 1.0,
            "stress_p99_ratio_max": 1.0,
        },
        "component_contribution": {
            "comparisons": [NO_PEPD_RESIDUAL, NO_MASK_RESIDUAL],
            "condition_delta_point_max": 0.0005,
            "combined_delta_point_max": 0.0,
            "combined_delta_ci_upper_max": 0.0,
            "formal_p_family": "two combined superiority tests",
            "holm_adjusted_alpha": 0.05,
            "terminal_tanh_gains_must_be_finite_and_positive": True,
        },
        "decision": "all gates pass => promote; any gate fails => stop route without modification",
        "retry_or_variant_authorized": False,
    }


def _expected_materialization_contract() -> dict[str, Any]:
    v2_protocol = strict_json(V2_PROTOCOL)
    return {
        "canonical_roi_contract": v2_protocol["materialization"]["crop_contract"],
        "canonical_roi_contract_sha256": v2_protocol["materialization"]["crop_contract_sha256"],
        "public_16000_roi_root_sha256": v2_protocol["materialization"]["roi_sha256"],
        "public_16000_target_root_sha256": v2_protocol["materialization"]["target_sha256"],
        "shared_native_roi_contract_sha256": v2_protocol["materialization"]["shared_native_roi_contract_sha256"],
        "crop_expansion": 1.0,
        "letterbox": False,
        "padding": False,
        "black_border": False,
        "training_augmentation_border": "reflect_101 only after canonical native ROI",
        "pepd_heatmap_coordinate_interval": [0.0, 63.0],
        "pepd_normalized_homography": "F_second @ inverse(F_first), first-to-second",
        "pepd_equivariance_pivot_normalizer": 63.0,
        "pepd_equivariance_ray_length_normalized": 0.25,
    }


def _expected_model_contract() -> dict[str, Any]:
    return {
        "name": "solver-anchored reliability-gated residual CAGH-V5",
        "progress_bins": 72,
        "anchor_pivot": [0.5, 0.5],
        "anchor_evidence": "raw keypoint_angle_evidence from predicted reference geometry",
        "legacy_pepd_evidence_enabled": False,
        "legacy_mask_evidence_enabled": False,
        "residual_standardization": "z(E)=(E-mean(E))/sqrt(mean((E-mean(E))^2)+1e-6)",
        "residual_formula": "R_branch=s(K)*(z(E_branch)-z(K))",
        "fusion_formula": "K+tanh(a_p)*stopgrad(g_p*R_p)+tanh(a_m)*stopgrad(g_m*R_m)",
        "signed_gain_range": [-1.0, 1.0],
        "gain_initialization": 0.0,
        "gate_range": [0.0, 1.0],
        "absolute_logit_clip": None,
        "description_limit": "relative-RMS residual with bounded signed gain; not a convex mixture or absolute bound",
        "nan_policy": "optional nonfinite branch gate=0 and anchor exact; nonfinite anchor/fusion => uniform invalid",
        "arms": list(ARM_NAMES),
        "runtime_signature": str(inspect.signature(CAGHV5SolverGatedResidual.forward)),
        "runtime_model_inputs": ["image", "final_to_isotropic", "crop_affine", "arm"],
        "runtime_forbidden": sorted(LABEL_TOKENS | {"sample_id", "group_id"}),
    }


def _solver_parent_loss_contract() -> dict[str, Any]:
    return {
        "final_posterior": "log_softmax(raw keypoint_angle_evidence)",
        "final_soft_ce": 1.0,
        "final_expected_smooth_l1": 2.0,
        "tip_heatmap_ce_each": 0.25,
        "tail_heatmap_ce_each": 0.25,
        "keypoint_coordinate": 0.5,
        "keypoint_direction": 0.25,
        "dense_vector": 0.10,
        "mask_support_bce": 0.25,
        "auxiliary_angle_evidence": 0.05,
        "dense_tick": 0.50,
        "reference_endpoint_consistency": 1.0,
        "reference_geometry_angle_arc": 0.25,
        "pepd_probability_progress": 0.0,
        "legacy_fused_final_loss": False,
    }


def _expected_training_contract() -> dict[str, Any]:
    p0_pepd_training = {
        "amp": True,
        "batch_size": 24,
        "bin_loss_weight": 0.2,
        "epochs": 60,
        "equivariance_pivot_weight": 1.0,
        "equivariance_weight": 0.5,
        "eta_min": 3e-6,
        "imagenet_initialization": str(
            Path.home() / ".cache/torch/hub/checkpoints" / EXPECTED_IMAGENET_FILENAME
        ),
        "imagenet_initialization_sha256": EXPECTED_IMAGENET_SHA256,
        "learning_rate": 0.0003,
        "optimizer": "AdamW",
        "paired_supervision_weight": 1.0,
        "pivot_loss_weight": 1.0,
        "scheduler": "CosineAnnealingLR terminal60",
        "seed": PEPD_SEED,
        "soft_target_sigma_bins": 1.25,
        "validation_during_training": False,
        "vector_loss_weight": 0.5,
        "weight_decay": 0.0001,
    }
    return {
        "pepd": {
            "mode": "imported_frozen_p0",
            "source_protocol": P0_PROTOCOL_ID,
            "source_protocol_canonical_sha256": EXPECTED_P0_PROTOCOL_CANONICAL_SHA256,
            "source_training_contract": p0_pepd_training,
            "seed": PEPD_SEED,
            "terminal_epoch": 60,
            "checkpoint_selection": "none; P0 fixed terminal epoch60",
            "terminal_sha256": EXPECTED_P0_PEPD_TERMINAL_SHA256,
            "model_state_sha256": EXPECTED_P0_PEPD_MODEL_STATE_SHA256,
            "training_in_p1_authorized": False,
            "import_receipt_required": True,
        },
        "parent": {
            "seed": PARENT_SEED,
            "arm": PARENT_TRAINING_ARM,
            "runtime_core_arm": SOLVER_CORE,
            "anchor": "raw keypoint_angle_evidence",
            "legacy_pepd_evidence_enabled": False,
            "legacy_mask_evidence_enabled": False,
            "mask_support_auxiliary_supervision": True,
            "evidence_refiner_trainable": False,
            "gradient_policy": PARENT_GRADIENT_POLICY,
            "terminal_state_all_finite_required": True,
            "loss": _solver_parent_loss_contract(),
            "epochs": 18,
            "batch_size": 16,
            "learning_rate": 0.0004,
            "weight_decay": 0.0001,
            "optimizer": "AdamW",
            "optimizer_param_group": {
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
            },
            "checkpoint_selection": "none; terminal epoch18 only",
            "outer_or_screen_validation_during_training": False,
        },
        "gains": {
            "seed": GAIN_SEED,
            "trained_arms": [FULL_GATED, NO_PEPD_RESIDUAL, NO_MASK_RESIDUAL],
            "fixed_control": SOLVER_CORE,
            "fresh_independent_initialization_each_arm": True,
            "warm_start_from_full": False,
            "epochs": 4,
            "batch_size": 16,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "optimizer": "AdamW",
            "optimizer_param_group": {
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
            },
            "gradient_policy": GAIN_GRADIENT_POLICY,
            "terminal_state_all_finite_required": True,
            "optimizer_state_all_finite_required": True,
            "scheduler": None,
            "checkpoint_selection": "none; terminal epoch4 only",
        },
        "augmentation": augmentation_config(),
        "epoch_reseed_policy": "base_seed + epoch*1000003; loader seed base_seed+epoch",
        "gradient_clip_norm": 5.0,
        "early_stopping": False,
        "best_checkpoint": False,
    }


def _expected_screen_contract() -> dict[str, Any]:
    return {
        "cohort": "inner_screen only",
        "conditions": ["clean", "frozen_stress"],
        "seed": EVALUATION_SEED,
        "stress_epoch": STRESS_EPOCH,
        "failure_penalty": 1.0,
        "all_arms_one_shared_batch_pass": True,
        "prediction_bundle_label_free": True,
        "score_bundle_separate": True,
        "outer_holdout_images_read": 0,
    }


def _archive_tree_sha256_before_receipt() -> tuple[int, int, str]:
    files = sorted(
        (
            path
            for path in P0_ARCHIVE_ROOT.rglob("*")
            if path.is_file() and path != P0_ARCHIVE_RECEIPT
        ),
        key=lambda path: path.relative_to(P0_ARCHIVE_ROOT).as_posix(),
    )
    lines: list[str] = []
    total = 0
    for path in files:
        relative = path.relative_to(P0_ARCHIVE_ROOT).as_posix()
        size = int(path.stat().st_size)
        total += size
        lines.append(f"{relative}\t{size}\t{sha256_file(path)}")
    return len(files), total, hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _archive_entry_roster_sha256() -> tuple[int, str]:
    """Enumerate every archive entry without following links or reparse points."""

    rows: list[str] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda value: value.name)
        for entry in entries:
            path = Path(entry.path)
            # pathlib.lstat() preserves NTFS inode/link metadata; DirEntry.stat
            # can report st_nlink=0 on this Windows/Python combination.
            metadata = path.lstat()
            relative = path.relative_to(P0_ARCHIVE_ROOT).as_posix()
            attributes = getattr(metadata, "st_file_attributes", 0)
            _require(
                not entry.is_symlink()
                and not bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)),
                f"P0 archive link/reparse entry: {relative}",
            )
            if stat.S_ISDIR(metadata.st_mode):
                rows.append(f"{relative}\tdir")
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                _require(int(metadata.st_nlink) == 1,
                         f"P0 archive hard-linked file: {relative}")
                rows.append(f"{relative}\tfile")
            else:
                raise ValueError(f"P0 archive non-regular entry: {relative}")

    visit(P0_ARCHIVE_ROOT)
    rows.sort()
    return len(rows), hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=1)
def _verify_p0_archive_closure() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected_files = {
        P0_PROTOCOL_PATH: EXPECTED_P0_PROTOCOL_FILE_SHA256,
        P0_ROOT_SNAPSHOT / "protocol_candidate.json": EXPECTED_P0_PROTOCOL_FILE_SHA256,
        P0_PROTOCOL_FREEZE_RECEIPT: EXPECTED_P0_PROTOCOL_FREEZE_RECEIPT_SHA256,
        P0_PREFLIGHT_PATH: EXPECTED_P0_PREFLIGHT_SHA256,
        P0_PREFLIGHT_RECEIPT: EXPECTED_P0_PREFLIGHT_RECEIPT_SHA256,
        P0_PEPD_JOURNAL: EXPECTED_P0_PEPD_JOURNAL_SHA256,
        P0_PEPD_TERMINAL: EXPECTED_P0_PEPD_TERMINAL_SHA256,
        P0_PEPD_SUMMARY: EXPECTED_P0_PEPD_SUMMARY_SHA256,
        P0_ARCHIVE_RECEIPT: EXPECTED_P0_ARCHIVE_RECEIPT_SHA256,
    }
    _require(P0_ARCHIVE_ROOT.is_dir(), "P0 archive is missing")
    for parent in (P0_ARCHIVE_ROOT, *P0_ARCHIVE_ROOT.parents):
        _require(not _is_reparse(parent), f"P0 archive reparse component: {parent}")
        if parent == parent.parent:
            break
    entry_count, entry_roster_sha256 = _archive_entry_roster_sha256()
    _require(
        entry_count == EXPECTED_P0_ARCHIVE_ENTRY_COUNT
        and entry_roster_sha256 == EXPECTED_P0_ARCHIVE_ENTRY_ROSTER_SHA256,
        "P0 archive entry roster drift",
    )
    for path, expected in expected_files.items():
        _require(
            path.is_file()
            and not path.is_symlink()
            and not _is_reparse(path)
            and int(path.stat().st_nlink) == 1,
            f"P0 evidence file missing/not independent regular file: {path}",
        )
        _require(sha256_file(path) == expected, f"P0 evidence hash drift: {path.name}")
    p0_protocol = strict_json(P0_PROTOCOL_PATH)
    _require(
        p0_protocol.get("protocol") == P0_PROTOCOL_ID
        and p0_protocol.get("status") == "frozen_public_inner_screen"
        and canonical_sha256(p0_protocol) == EXPECTED_P0_PROTOCOL_CANONICAL_SHA256,
        "P0 protocol identity/canonical drift",
    )
    _require(
        p0_protocol.get("source_manifest", {}).get("manifest_sha256")
        == EXPECTED_P0_SOURCE_MANIFEST_SHA256,
        "P0 source manifest drift",
    )
    for path_key, hash_key in (
        ("fit_manifest", "fit_manifest_sha256"),
        ("fit_manifest_protocol", "fit_manifest_protocol_sha256"),
        ("inner_rosters", "inner_rosters_sha256"),
        ("inner_train_manifest", "inner_train_manifest_sha256"),
        ("inner_train_manifest_protocol", "inner_train_manifest_protocol_sha256"),
        ("inner_screen_manifest", "inner_screen_manifest_sha256"),
        ("inner_screen_manifest_protocol", "inner_screen_manifest_protocol_sha256"),
    ):
        archived_input = P0_ROOT_SNAPSHOT / "inputs" / Path(
            str(p0_protocol["inputs"][path_key])
        ).name
        _require(
            archived_input.is_file()
            and not _is_reparse(archived_input)
            and sha256_file(archived_input) == p0_protocol["inputs"][hash_key],
            f"P0 archived input byte drift: {path_key}",
        )
    source_pins = {
        **dict(p0_protocol["source_manifest"]["sha256"]),
        **dict(p0_protocol["test_sources"]),
    }
    actual_snapshot = {
        path.relative_to(P0_SOURCE_SNAPSHOT).as_posix(): sha256_file(path)
        for path in P0_SOURCE_SNAPSHOT.rglob("*")
        if path.is_file()
    }
    _require(actual_snapshot == source_pins, "P0 source/test byte snapshot drift")
    run_directories = sorted(
        path.name for path in (P0_ROOT_SNAPSHOT / "runs").iterdir() if path.is_dir()
    )
    _require(
        run_directories == [f"pepd_inner_seed_{PEPD_SEED}"],
        "P0 archive contains parent/gain/screen run directories",
    )
    _require(not list(P0_ROOT_SNAPSHOT.rglob("*.tmp.*")), "P0 archive contains tmp files")
    lock_files = [path for path in (P0_ROOT_SNAPSHOT / "locks").rglob("*") if path.is_file()]
    _require(not lock_files, "P0 archive contains active/orphan locks")
    receipt = strict_json(P0_ARCHIVE_RECEIPT)
    file_count, byte_count, tree_sha256 = _archive_tree_sha256_before_receipt()
    _require(
        receipt.get("status") == "p0_pepd_evidence_archived"
        and receipt.get("p0_protocol_canonical_sha256")
        == EXPECTED_P0_PROTOCOL_CANONICAL_SHA256
        and receipt.get("p0_source_manifest_sha256") == EXPECTED_P0_SOURCE_MANIFEST_SHA256
        and receipt.get("pepd_terminal_sha256") == EXPECTED_P0_PEPD_TERMINAL_SHA256
        and receipt.get("pepd_model_state_sha256") == EXPECTED_P0_PEPD_MODEL_STATE_SHA256
        and int(receipt.get("archive_files_before_receipt", -1)) == file_count
        and int(receipt.get("archive_bytes_before_receipt", -1)) == byte_count
        and receipt.get("archive_tree_sha256_before_receipt")
        == EXPECTED_P0_ARCHIVE_TREE_SHA256
        and tree_sha256 == EXPECTED_P0_ARCHIVE_CANONICAL_TREE_SHA256
        and int(receipt.get("parent_gain_screen_stage_directories", -1)) == 0,
        "P0 archive receipt/tree drift",
    )
    summary = strict_json(P0_PEPD_SUMMARY)
    _require(
        summary.get("status") == "complete"
        and summary.get("role") == "inner_train_pepd"
        and int(summary.get("seed", -1)) == PEPD_SEED
        and int(summary.get("terminal_epoch", -1)) == 60
        and summary.get("terminal_sha256") == EXPECTED_P0_PEPD_TERMINAL_SHA256
        and summary.get("model_state_sha256") == EXPECTED_P0_PEPD_MODEL_STATE_SHA256
        and summary.get("protocol_snapshot_sha256")
        == EXPECTED_P0_PROTOCOL_CANONICAL_SHA256
        and summary.get("inner_screen_images_read") == 0
        and summary.get("outer_holdout_images_read") == 0,
        "P0 PEPD summary drift",
    )
    return p0_protocol, summary


def _load_imported_p0_protocol(root: Path) -> Mapping[str, Any]:
    """Load only the imported P0 protocol; never traverse archived screen inputs."""

    path = root / "imports/pepd_p0/p0_protocol.json"
    _require(
        path.is_file()
        and not path.is_symlink()
        and not _is_reparse(path)
        and int(path.stat().st_nlink) == 1
        and sha256_file(path) == EXPECTED_P0_PROTOCOL_FILE_SHA256,
        "imported P0 protocol byte/regular-file drift",
    )
    value = strict_json(path)
    _require(
        value.get("protocol") == P0_PROTOCOL_ID
        and value.get("status") == "frozen_public_inner_screen"
        and canonical_sha256(value) == EXPECTED_P0_PROTOCOL_CANONICAL_SHA256
        and value.get("source_manifest", {}).get("manifest_sha256")
        == EXPECTED_P0_SOURCE_MANIFEST_SHA256,
        "imported P0 protocol identity drift",
    )
    return value


def _expected_migration_contract(root: Path) -> dict[str, Any]:
    return {
        "kind": "infrastructure_only_p0_to_p1",
        "reason": "parent resume/publication/provenance state-machine hardening before parent training",
        "algorithm_or_hyperparameter_changed": False,
        "p0_protocol": P0_PROTOCOL_ID,
        "p0_protocol_path": str(P0_PROTOCOL_PATH.resolve()),
        "p0_protocol_file_sha256": EXPECTED_P0_PROTOCOL_FILE_SHA256,
        "p0_protocol_canonical_sha256": EXPECTED_P0_PROTOCOL_CANONICAL_SHA256,
        "p0_source_manifest_sha256": EXPECTED_P0_SOURCE_MANIFEST_SHA256,
        "p0_archive": str(P0_ARCHIVE_ROOT.resolve()),
        "p0_archive_receipt_sha256": EXPECTED_P0_ARCHIVE_RECEIPT_SHA256,
        "p0_archive_receipt_tree_sha256": EXPECTED_P0_ARCHIVE_TREE_SHA256,
        "p0_archive_canonical_tree_sha256": EXPECTED_P0_ARCHIVE_CANONICAL_TREE_SHA256,
        "p0_archive_entry_count": EXPECTED_P0_ARCHIVE_ENTRY_COUNT,
        "p0_archive_entry_roster_sha256": EXPECTED_P0_ARCHIVE_ENTRY_ROSTER_SHA256,
        "p0_parent_gain_screen_stage_directories": 0,
        "p0_pepd_terminal_sha256": EXPECTED_P0_PEPD_TERMINAL_SHA256,
        "p0_pepd_model_state_sha256": EXPECTED_P0_PEPD_MODEL_STATE_SHA256,
        "p1_import_directory": str((root / "imports/pepd_p0").resolve(strict=False)),
        "p1_import_receipt": str(
            (root / "imports/pepd_p0/pepd_import_complete.json").resolve(strict=False)
        ),
        "copy_policy": "new-only byte-identical files; no hard links; no PEPD training in P1",
        "claim_limit": "P1 is infrastructure migration after P0 PEPD; scientific contracts unchanged",
    }


def _validate_protocol_migration(
    protocol: Mapping[str, Any], root: Path, *, p0: Mapping[str, Any]
) -> None:
    _require(
        dict(protocol["infrastructure_migration"]) == _expected_migration_contract(root),
        "P0-to-P1 migration contract drift",
    )
    _require(dict(protocol["materialization"]) == dict(p0["materialization"]),
             "migration changed materialization")
    _require(dict(protocol["model"]) == dict(p0["model"]), "migration changed model")
    p1_parent = dict(protocol["training"]["parent"])
    _require(p1_parent.pop("optimizer") == "AdamW", "migration parent optimizer drift")
    _require(
        p1_parent.pop("optimizer_param_group") == {
            "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
            "maximize": False, "foreach": None, "capturable": False,
            "differentiable": False, "fused": None,
            "decoupled_weight_decay": True,
        },
        "migration parent optimizer parameter contract drift",
    )
    _require(p1_parent == p0["training"]["parent"],
             "migration changed parent scientific training contract")
    p1_gains = dict(protocol["training"]["gains"])
    _require(
        p1_gains.pop("optimizer_param_group") == {
            "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
            "maximize": False, "foreach": None, "capturable": False,
            "differentiable": False, "fused": None,
            "decoupled_weight_decay": True,
        }
        and p1_gains.pop("gradient_policy") == GAIN_GRADIENT_POLICY
        and p1_gains.pop("terminal_state_all_finite_required") is True
        and p1_gains.pop("optimizer_state_all_finite_required") is True,
        "migration gain optimizer/finite-state contract drift",
    )
    _require(p1_gains == p0["training"]["gains"],
             "migration changed gain scientific training contract")
    for key in (
        "augmentation", "epoch_reseed_policy", "gradient_clip_norm",
        "early_stopping", "best_checkpoint",
    ):
        _require(protocol["training"][key] == p0["training"][key],
                 f"migration changed training.{key}")
    _require(
        protocol["training"]["pepd"]["source_training_contract"]
        == p0["training"]["pepd"],
        "migration changed P0 PEPD training contract",
    )
    _require(protocol["screen_evaluation"] == p0["screen_evaluation"],
             "migration changed screen contract")
    _require(protocol["promotion"] == p0["promotion"], "migration changed promotion gates")
    for key in (
        "source_public_manifest_sha256", "source_public_protocol_sha256",
        "source_content_inventory_sha256", "fit_manifest_sha256",
        "fit_manifest_protocol_sha256", "inner_rosters_sha256",
        "inner_train_manifest_sha256", "inner_train_manifest_protocol_sha256",
        "inner_screen_manifest_sha256", "inner_screen_manifest_protocol_sha256",
    ):
        _require(protocol["inputs"][key] == p0["inputs"][key],
                 f"migration changed input bytes: {key}")
    _require(protocol["inputs"]["outer_identity"] == p0["inputs"]["outer_identity"],
             "migration changed outer identity")
    _require(protocol["inputs"]["inner_identity"] == p0["inputs"]["inner_identity"],
             "migration changed inner identity")


def _pepd_import_file_contract() -> dict[str, tuple[Path, str]]:
    return {
        "p0_protocol.json": (P0_PROTOCOL_PATH, EXPECTED_P0_PROTOCOL_FILE_SHA256),
        "p0_protocol_freeze_receipt.json": (
            P0_PROTOCOL_FREEZE_RECEIPT,
            EXPECTED_P0_PROTOCOL_FREEZE_RECEIPT_SHA256,
        ),
        "p0_preflight.json": (P0_PREFLIGHT_PATH, EXPECTED_P0_PREFLIGHT_SHA256),
        "p0_preflight_receipt.json": (
            P0_PREFLIGHT_RECEIPT,
            EXPECTED_P0_PREFLIGHT_RECEIPT_SHA256,
        ),
        "journal.pt": (P0_PEPD_JOURNAL, EXPECTED_P0_PEPD_JOURNAL_SHA256),
        "terminal.pt": (P0_PEPD_TERMINAL, EXPECTED_P0_PEPD_TERMINAL_SHA256),
        "summary.json": (P0_PEPD_SUMMARY, EXPECTED_P0_PEPD_SUMMARY_SHA256),
        "p0_archive_receipt.json": (
            P0_ARCHIVE_RECEIPT,
            EXPECTED_P0_ARCHIVE_RECEIPT_SHA256,
        ),
    }


def _nested_tensors_finite(value: Any, *, label: str) -> None:
    if torch.is_tensor(value):
        _require(bool(torch.isfinite(value).all()), f"non-finite tensor: {label}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _nested_tensors_finite(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _nested_tensors_finite(child, label=f"{label}[{index}]")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"non-finite scalar: {label}")


def _cpu_nested_state(value: Any) -> Any:
    """Clone checkpoint tensor trees to CPU before validation and publication."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_nested_state(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_nested_state(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_nested_state(child) for child in value)
    return value


def _validate_adamw_optimizer_state(
    optimizer_state: Any,
    *,
    model_state: Mapping[str, torch.Tensor],
    trainable: Sequence[str],
    global_step: int,
    learning_rate: float,
    weight_decay: float,
    param_group_contract: Mapping[str, Any],
    label: str,
) -> None:
    """Fail closed on AdamW resume identity, parameter order, and moments."""

    _nested_tensors_finite(optimizer_state, label=label)
    _require(
        isinstance(optimizer_state, Mapping)
        and set(optimizer_state) == {"state", "param_groups"},
        f"{label}: optimizer state schema drift",
    )
    param_groups = optimizer_state["param_groups"]
    _require(
        isinstance(param_groups, list) and len(param_groups) == 1,
        f"{label}: optimizer param-group count drift",
    )
    param_group = param_groups[0]
    expected_group_keys = {
        "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
        "foreach", "capturable", "differentiable", "fused",
        "decoupled_weight_decay", "params",
    }
    _require(
        isinstance(param_group, Mapping)
        and set(param_group) == expected_group_keys
        and float(param_group["lr"]) == float(learning_rate)
        and tuple(param_group["betas"]) == tuple(param_group_contract["betas"])
        and float(param_group["eps"]) == float(param_group_contract["eps"])
        and float(param_group["weight_decay"]) == float(weight_decay)
        and param_group["amsgrad"] is param_group_contract["amsgrad"]
        and param_group["maximize"] is param_group_contract["maximize"]
        and param_group["foreach"] is param_group_contract["foreach"]
        and param_group["capturable"] is param_group_contract["capturable"]
        and param_group["differentiable"] is param_group_contract["differentiable"]
        and param_group["fused"] is param_group_contract["fused"]
        and param_group["decoupled_weight_decay"]
        is param_group_contract["decoupled_weight_decay"]
        and param_group["params"] == list(range(len(trainable))),
        f"{label}: optimizer contract/order drift",
    )
    state = optimizer_state["state"]
    _require(
        isinstance(state, Mapping)
        and set(state) == set(range(len(trainable))),
        f"{label}: optimizer/trainable state coverage drift",
    )
    _require(int(global_step) > 0, f"{label}: optimizer global step is not positive")
    for parameter_id, parameter_name in enumerate(trainable):
        state_name = parameter_name
        if state_name not in model_state and state_name.startswith("parent."):
            state_name = state_name.removeprefix("parent.")
        _require(
            state_name in model_state and torch.is_tensor(model_state[state_name]),
            f"{label}: missing trainable model tensor: {parameter_name}",
        )
        parameter = model_state[state_name]
        _require(
            parameter.is_floating_point(),
            f"{label}: trainable parameter is not floating point: {parameter_name}",
        )
        parameter_state = state[parameter_id]
        _require(
            isinstance(parameter_state, Mapping)
            and set(parameter_state) == {"step", "exp_avg", "exp_avg_sq"},
            f"{label}: AdamW state schema drift: {parameter_name}",
        )
        step = parameter_state["step"]
        _require(
            torch.is_tensor(step)
            and step.dtype == torch.float32
            and step.ndim == 0
            and step.device.type == "cpu"
            and bool(torch.isfinite(step))
            and float(step.item()) == float(global_step),
            f"{label}: AdamW step drift: {parameter_name}",
        )
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            _require(
                torch.is_tensor(moment)
                and moment.shape == parameter.shape
                and moment.dtype == parameter.dtype
                and moment.device.type == "cpu"
                and bool(torch.isfinite(moment).all()),
                f"{label}: AdamW {moment_name} drift: {parameter_name}",
            )


def _validate_loaded_adamw_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    module: torch.nn.Module,
    trainable: Sequence[str],
    global_step: int,
    label: str,
) -> None:
    """Verify PyTorch restored CPU steps and device-local moments exactly."""

    named = dict(module.named_parameters())
    expected_parameters = []
    for name in trainable:
        _require(name in named, f"{label}: missing loaded parameter: {name}")
        expected_parameters.append(named[name])
    actual_parameters = (
        optimizer.param_groups[0]["params"] if len(optimizer.param_groups) == 1 else []
    )
    _require(
        len(actual_parameters) == len(expected_parameters)
        and all(
            actual is expected
            for actual, expected in zip(actual_parameters, expected_parameters, strict=True)
        ),
        f"{label}: loaded optimizer parameter order drift",
    )
    _require(
        {id(parameter) for parameter in optimizer.state}
        == {id(parameter) for parameter in expected_parameters},
        f"{label}: loaded optimizer state coverage drift",
    )
    for name, parameter in zip(trainable, expected_parameters, strict=True):
        state = optimizer.state[parameter]
        _require(
            set(state) == {"step", "exp_avg", "exp_avg_sq"},
            f"{label}: loaded AdamW schema drift: {name}",
        )
        step = state["step"]
        _require(
            torch.is_tensor(step)
            and step.dtype == torch.float32
            and step.ndim == 0
            and step.device.type == "cpu"
            and bool(torch.isfinite(step))
            and float(step.item()) == float(global_step),
            f"{label}: loaded AdamW step drift: {name}",
        )
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state[moment_name]
            _require(
                torch.is_tensor(moment)
                and moment.shape == parameter.shape
                and moment.dtype == parameter.dtype
                and moment.device == parameter.device
                and bool(torch.isfinite(moment).all()),
                f"{label}: loaded AdamW {moment_name} drift: {name}",
            )


def _nested_equal(first: Any, second: Any) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return torch.is_tensor(first) and torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (
            isinstance(first, Mapping)
            and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_nested_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second)
            and all(_nested_equal(left, right) for left, right in zip(first, second))
        )
    return first == second


def import_pepd_p0(protocol: Mapping[str, Any], *, root: Path) -> Path:
    """Publish the audited P0 PEPD bytes as a frozen P1 upstream input."""

    _verify_p0_archive_closure()
    _require(not (root / "preflight.json").exists(), "PEPD import must precede P1 preflight")
    _require(not (root / "runs").exists(), "PEPD import found P1 run artifacts")
    _require(not (root / "locks").exists(), "PEPD import found P1 locks")
    directory = root / "imports" / "pepd_p0"
    _require(not directory.exists(), "P0 PEPD import already exists")
    directory.mkdir(parents=True, exist_ok=False)
    contract = _pepd_import_file_contract()
    for name, (source, expected) in contract.items():
        _require(sha256_file(source) == expected, f"P0 import source drift: {name}")
        copy_file_new(source, directory / name)
        _require(sha256_file(directory / name) == expected, f"P0 import copy drift: {name}")
    receipt_path = directory / "pepd_import_complete.json"
    imported = {name: expected for name, (_, expected) in contract.items()}
    atomic_json_new(
        receipt_path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "p0_pepd_import_complete",
            "protocol_snapshot_sha256": canonical_sha256(protocol),
            "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
            "p0_protocol": P0_PROTOCOL_ID,
            "p0_protocol_canonical_sha256": EXPECTED_P0_PROTOCOL_CANONICAL_SHA256,
            "p0_source_manifest_sha256": EXPECTED_P0_SOURCE_MANIFEST_SHA256,
            "p0_archive_receipt_sha256": EXPECTED_P0_ARCHIVE_RECEIPT_SHA256,
            "p0_archive_receipt_tree_sha256": EXPECTED_P0_ARCHIVE_TREE_SHA256,
            "p0_archive_canonical_tree_sha256": EXPECTED_P0_ARCHIVE_CANONICAL_TREE_SHA256,
            "p0_archive_entry_count": EXPECTED_P0_ARCHIVE_ENTRY_COUNT,
            "p0_archive_entry_roster_sha256": EXPECTED_P0_ARCHIVE_ENTRY_ROSTER_SHA256,
            "pepd_seed": PEPD_SEED,
            "pepd_terminal_epoch": 60,
            "pepd_terminal_sha256": EXPECTED_P0_PEPD_TERMINAL_SHA256,
            "pepd_model_state_sha256": EXPECTED_P0_PEPD_MODEL_STATE_SHA256,
            "imported_files": imported,
            "copy_policy": "new-only byte-identical files; no hard links; no PEPD training in P1",
            "optimizer_steps": 0,
            "images_read": 0,
            "annotations_read": 0,
        },
    )
    _validated_pepd_import(protocol, None, root)
    return receipt_path


def _validated_pepd_import(
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery | None,
    root: Path,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    directory = root / "imports" / "pepd_p0"
    contract = _pepd_import_file_contract()
    expected_names = set(contract) | {"pepd_import_complete.json"}
    _require(directory.is_dir() and not _is_reparse(directory), "P0 PEPD import missing")
    entries = list(directory.iterdir())
    _require(
        len(entries) == len(expected_names)
        and {path.name for path in entries} == expected_names,
        "P0 PEPD import file closure drift",
    )
    for path in entries:
        _require(
            path.is_file()
            and not path.is_symlink()
            and not _is_reparse(path)
            and int(path.stat().st_nlink) == 1,
            f"P0 PEPD import artifact is not an independent regular file: {path.name}",
        )
    for name, (_, expected) in contract.items():
        path = directory / name
        _require(sha256_file(path) == expected,
                 f"P0 PEPD imported byte drift: {name}")
    receipt_path = directory / "pepd_import_complete.json"
    receipt = strict_json(receipt_path)
    expected_receipt_keys = {
        "schema_version", "protocol", "status", "protocol_snapshot_sha256",
        "source_manifest_sha256", "p0_protocol", "p0_protocol_canonical_sha256",
        "p0_source_manifest_sha256", "p0_archive_receipt_sha256",
        "p0_archive_receipt_tree_sha256", "p0_archive_canonical_tree_sha256",
        "p0_archive_entry_count", "p0_archive_entry_roster_sha256",
        "pepd_seed", "pepd_terminal_epoch",
        "pepd_terminal_sha256", "pepd_model_state_sha256", "imported_files",
        "copy_policy", "optimizer_steps", "images_read", "annotations_read",
    }
    _require(set(receipt) == expected_receipt_keys, "P0 PEPD import receipt schema drift")
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("protocol") == PROTOCOL
        and receipt.get("status") == "p0_pepd_import_complete"
        and receipt.get("protocol_snapshot_sha256") == canonical_sha256(protocol)
        and receipt.get("source_manifest_sha256")
        == protocol["source_manifest"]["manifest_sha256"]
        and receipt.get("p0_protocol") == P0_PROTOCOL_ID
        and receipt.get("p0_protocol_canonical_sha256")
        == EXPECTED_P0_PROTOCOL_CANONICAL_SHA256
        and receipt.get("p0_source_manifest_sha256") == EXPECTED_P0_SOURCE_MANIFEST_SHA256
        and receipt.get("p0_archive_receipt_sha256") == EXPECTED_P0_ARCHIVE_RECEIPT_SHA256
        and receipt.get("p0_archive_receipt_tree_sha256") == EXPECTED_P0_ARCHIVE_TREE_SHA256
        and receipt.get("p0_archive_canonical_tree_sha256")
        == EXPECTED_P0_ARCHIVE_CANONICAL_TREE_SHA256
        and receipt.get("p0_archive_entry_count") == EXPECTED_P0_ARCHIVE_ENTRY_COUNT
        and receipt.get("p0_archive_entry_roster_sha256")
        == EXPECTED_P0_ARCHIVE_ENTRY_ROSTER_SHA256
        and int(receipt.get("pepd_seed", -1)) == PEPD_SEED
        and int(receipt.get("pepd_terminal_epoch", -1)) == 60
        and receipt.get("pepd_terminal_sha256") == EXPECTED_P0_PEPD_TERMINAL_SHA256
        and receipt.get("pepd_model_state_sha256") == EXPECTED_P0_PEPD_MODEL_STATE_SHA256
        and dict(receipt.get("imported_files") or {})
        == {name: expected for name, (_, expected) in contract.items()}
        and receipt.get("copy_policy")
        == "new-only byte-identical files; no hard links; no PEPD training in P1"
        and receipt.get("optimizer_steps") == 0
        and receipt.get("images_read") == 0
        and receipt.get("annotations_read") == 0,
        "P0 PEPD import receipt drift",
    )
    imported_protocol = _load_imported_p0_protocol(root)
    summary = strict_json(directory / "summary.json")
    terminal = directory / "terminal.pt"
    journal = directory / "journal.pt"
    payload = torch.load(terminal, map_location="cpu", weights_only=False)
    journal_payload = torch.load(journal, map_location="cpu", weights_only=False)
    expected_payload_keys = {
        "schema_version", "protocol", "role", "seed", "terminal_epoch",
        "completed_epoch", "global_step", "checkpoint_selection",
        "protocol_snapshot_sha256", "source_manifest_sha256",
        "inner_train_ids_sha256", "inner_screen_images_read",
        "outer_holdout_images_read", "model_state", "optimizer_state",
        "scheduler_state", "scaler_state", "history",
    }
    _require(set(payload) == expected_payload_keys, "P0 PEPD terminal schema drift")
    _require(_nested_equal(payload, journal_payload), "P0 PEPD journal/terminal semantic drift")
    history = payload["history"]
    batches = math.ceil(EXPECTED_INNER_TRAIN[0] / 24)
    _require(
        payload.get("protocol") == P0_PROTOCOL_ID
        and payload.get("role") == "inner_train_pepd"
        and int(payload.get("seed", -1)) == PEPD_SEED
        and int(payload.get("terminal_epoch", -1)) == 60
        and int(payload.get("completed_epoch", -1)) == 60
        and payload.get("checkpoint_selection") == "none; fixed terminal"
        and payload.get("protocol_snapshot_sha256") == EXPECTED_P0_PROTOCOL_CANONICAL_SHA256
        and payload.get("source_manifest_sha256") == EXPECTED_P0_SOURCE_MANIFEST_SHA256
        and payload.get("inner_train_ids_sha256") == EXPECTED_HASHES["inner_train_ids_sha256"]
        and payload.get("inner_screen_images_read") == 0
        and payload.get("outer_holdout_images_read") == 0
        and isinstance(history, list)
        and len(history) == 60,
        "P0 PEPD terminal identity/history drift",
    )
    cumulative = 0
    for epoch, row in enumerate(history, start=1):
        _require(
            row.get("epoch") == epoch
            and row.get("terminal_epoch") == 60
            and row.get("samples") == EXPECTED_INNER_TRAIN[0]
            and int(row.get("optimizer_steps", -1))
            + int(row.get("skipped_optimizer_steps", -1)) == batches,
            f"P0 PEPD history drift: epoch{epoch}",
        )
        cumulative += int(row["optimizer_steps"])
        _require(row.get("global_step") == cumulative, f"P0 PEPD global step drift: epoch{epoch}")
        _nested_tensors_finite(row, label=f"P0 PEPD history epoch{epoch}")
    _require(payload.get("global_step") == cumulative, "P0 PEPD terminal global step drift")
    _require_tensor_mapping_finite(payload["model_state"], label="P0 imported PEPD model state")
    _nested_tensors_finite(payload["optimizer_state"], label="P0 imported PEPD optimizer")
    _nested_tensors_finite(payload["scheduler_state"], label="P0 imported PEPD scheduler")
    _nested_tensors_finite(payload["scaler_state"], label="P0 imported PEPD scaler")
    _require(
        _tensor_mapping_sha256(payload["model_state"])
        == EXPECTED_P0_PEPD_MODEL_STATE_SHA256,
        "P0 imported PEPD model-state hash drift",
    )
    _require(
        summary.get("terminal_sha256") == EXPECTED_P0_PEPD_TERMINAL_SHA256
        and summary.get("model_state_sha256") == EXPECTED_P0_PEPD_MODEL_STATE_SHA256
        and summary.get("history") == history,
        "P0 imported PEPD summary drift",
    )
    if discovery is not None:
        _require(
            discovery.identity["inner_train_ids_sha256"]
            == payload["inner_train_ids_sha256"],
            "P0 imported PEPD/current roster drift",
        )
    return terminal, payload, summary, receipt


def _validate_protocol_semantics(
    protocol: Mapping[str, Any], root: Path, *, scope: str | None = None
) -> None:
    _verify_static_external_artifacts()
    expected_top = {
        "schema_version", "protocol", "status", "adaptive_source_evidence",
        "outer_common_holdout_evaluation_authorized",
        "external_or_field_evaluation_authorized", "output_root", "inputs",
        "materialization", "adaptive_lineage", "model", "training",
        "screen_evaluation", "promotion", "source_manifest", "test_sources",
        "environment", "infrastructure_migration",
    }
    _require(set(protocol) == expected_top, "protocol top-level schema drift")
    _require(
        protocol.get("schema_version") == 1
        and protocol.get("protocol") == PROTOCOL
        and protocol.get("status") == "frozen_public_inner_screen"
        and protocol.get("adaptive_source_evidence") is True
        and protocol.get("outer_common_holdout_evaluation_authorized") is False
        and protocol.get("external_or_field_evaluation_authorized") is False
        and str(protocol.get("output_root")) == str(root),
        "protocol identity/authorization drift",
    )
    _require(dict(protocol["materialization"]) == _expected_materialization_contract(),
             "materialization contract drift")
    _require(dict(protocol["model"]) == _expected_model_contract(), "model contract drift")
    _require(dict(protocol["training"]) == _expected_training_contract(),
             "training contract drift")
    _require(dict(protocol["screen_evaluation"]) == _expected_screen_contract(),
             "screen evaluation contract drift")
    _require(dict(protocol["promotion"]) == promotion_contract(), "promotion gates drift")
    expected_lineage = {
        "frozen_v2_protocol": str(V2_PROTOCOL.resolve()),
        "frozen_v2_protocol_sha256": EXPECTED_V2_PROTOCOL_SHA256,
        "reviewed_v2_source_statistics": str(V2_SOURCE_STATISTICS.resolve()),
        "reviewed_v2_source_statistics_sha256": EXPECTED_V2_SOURCE_STATISTICS_SHA256,
        "claim_limit": "locked adaptive source confirmation, not pristine method-family confirmation",
    }
    _require(dict(protocol["adaptive_lineage"]) == expected_lineage,
             "adaptive lineage drift")
    inputs = protocol["inputs"]
    expected_input_keys = {
        "source_public_manifest", "source_public_manifest_sha256",
        "source_public_protocol_sha256", "source_content_inventory",
        "source_content_inventory_sha256", "fit_manifest", "fit_manifest_sha256",
        "fit_manifest_protocol", "fit_manifest_protocol_sha256", "inner_rosters",
        "inner_rosters_sha256", "inner_train_manifest",
        "inner_train_manifest_sha256", "inner_train_manifest_protocol",
        "inner_train_manifest_protocol_sha256", "inner_screen_manifest",
        "inner_screen_manifest_sha256", "inner_screen_manifest_protocol",
        "inner_screen_manifest_protocol_sha256", "outer_identity", "inner_identity",
        "source_manifest_protocol",
    }
    _require(isinstance(inputs, Mapping) and set(inputs) == expected_input_keys,
             "input contract schema drift")
    _require(dict(inputs["outer_identity"]) == {
        "public_samples": EXPECTED_PUBLIC[0], "public_groups": EXPECTED_PUBLIC[1],
        "outer_fit_samples": EXPECTED_OUTER_FIT[0], "outer_fit_groups": EXPECTED_OUTER_FIT[1],
        "outer_holdout_samples": EXPECTED_OUTER_HOLDOUT[0],
        "outer_holdout_groups": EXPECTED_OUTER_HOLDOUT[1],
        "outer_fit_ids_sha256": EXPECTED_HASHES["outer_fit_ids_sha256"],
        "outer_holdout_ids_sha256": EXPECTED_HASHES["outer_holdout_ids_sha256"],
        "outer_fit_groups_sha256": EXPECTED_HASHES["outer_fit_groups_sha256"],
        "outer_holdout_groups_sha256": EXPECTED_HASHES["outer_holdout_groups_sha256"],
        "sample_overlap": 0, "group_overlap": 0,
    }, "outer identity drift")
    inner_identity = dict(inputs["inner_identity"])
    for key, expected in EXPECTED_HASHES.items():
        if key.startswith("inner_") or key.startswith("outer_fit_"):
            _require(inner_identity.get(key) == expected, f"inner identity drift: {key}")
    _require(
        inner_identity.get("outer_fit_samples") == EXPECTED_OUTER_FIT[0]
        and inner_identity.get("outer_fit_groups") == EXPECTED_OUTER_FIT[1]
        and inner_identity.get("inner_train_samples") == EXPECTED_INNER_TRAIN[0]
        and inner_identity.get("inner_train_groups") == EXPECTED_INNER_TRAIN[1]
        and inner_identity.get("inner_screen_samples") == EXPECTED_INNER_SCREEN[0]
        and inner_identity.get("inner_screen_groups") == EXPECTED_INNER_SCREEN[1]
        and inner_identity.get("sample_overlap") == 0
        and inner_identity.get("group_overlap") == 0
        and inner_identity.get("split_salt") == INNER_SPLIT_SALT
        and inner_identity.get("split_hash_encoding")
        == "sha256(utf8(salt + ':' + group_id))",
        "inner identity counts/split drift",
    )
    expected_paths = {
        "source_public_manifest": PUBLIC_MANIFEST.resolve(),
        "source_content_inventory": CONTENT_INVENTORY.resolve(),
        "fit_manifest": root / "inputs/syncg_outer_fit_only.jsonl",
        "fit_manifest_protocol": root / "inputs/syncg_outer_fit_only.jsonl.protocol.json",
        "inner_rosters": root / "inputs/inner_rosters.label_free.json",
        "inner_train_manifest": root / "inputs/syncg_inner_train_only.jsonl",
        "inner_train_manifest_protocol": root / "inputs/syncg_inner_train_only.jsonl.protocol.json",
        "inner_screen_manifest": root / "inputs/syncg_inner_screen_only.jsonl",
        "inner_screen_manifest_protocol": root / "inputs/syncg_inner_screen_only.jsonl.protocol.json",
    }
    for key, expected in expected_paths.items():
        _require(Path(str(inputs[key])).resolve(strict=False) == expected.resolve(strict=False),
                 f"input path drift: {key}")
    for key, value in inputs.items():
        if key.endswith("_sha256"):
            _require(isinstance(value, str) and len(value) == 64
                     and all(char in "0123456789abcdef" for char in value),
                      f"input SHA256 malformed: {key}")
    _require(
        inputs["source_public_manifest_sha256"] == EXPECTED_PUBLIC_MANIFEST_SHA256
        and inputs["source_public_protocol_sha256"]
        == EXPECTED_PUBLIC_MANIFEST_PROTOCOL_SHA256
        and inputs["source_content_inventory_sha256"]
        == EXPECTED_CONTENT_INVENTORY_SHA256,
        "pinned public source artifact drift",
    )
    _require(
        dict(inputs["source_manifest_protocol"])
        == dict(strict_json(PUBLIC_MANIFEST_PROTOCOL)),
        "public manifest protocol semantic drift",
    )
    actual_sources = source_manifest()
    _require(
        dict(protocol["source_manifest"]) == {
            "sha256": actual_sources,
            "manifest_sha256": canonical_sha256(actual_sources),
        },
        "protocol source manifest drift",
    )
    _require(
        dict(protocol["test_sources"]) == _expected_test_sources(),
        "protocol test source roster/hash drift",
    )
    _require(
        dict(protocol["environment"]) == environment_identity(),
        "protocol environment drift",
    )
    p0_protocol = (
        _verify_p0_archive_closure()[0]
        if scope in {None, "p0_import"}
        else _load_imported_p0_protocol(root)
    )
    _validate_protocol_migration(protocol, root, p0=p0_protocol)


def _write_fit_package(output_root: Path) -> dict[str, Any]:
    rows = strict_jsonl(PUBLIC_MANIFEST)
    samples, source_protocol = load_syncg_manifest(
        PUBLIC_MANIFEST, expected_split="train"
    )
    _require((len(samples), len({value.group_id for value in samples})) == EXPECTED_PUBLIC,
             "public manifest inventory drift")
    outer_fit, outer_holdout = grouped_train_val_split(
        samples, validation_fraction=0.10, seed=OUTER_SPLIT_SEED
    )
    fit_ids = {str(value.sample_id) for value in outer_fit}
    holdout_ids = {str(value.sample_id) for value in outer_holdout}
    fit_groups = {str(value.group_id) for value in outer_fit}
    holdout_groups = {str(value.group_id) for value in outer_holdout}
    _require((len(fit_ids), len(fit_groups)) == EXPECTED_OUTER_FIT, "outer fit drift")
    _require((len(holdout_ids), len(holdout_groups)) == EXPECTED_OUTER_HOLDOUT,
             "outer holdout drift")
    _require(not fit_ids & holdout_ids and not fit_groups & holdout_groups,
             "outer split overlap")
    outer_identity = {
        "public_samples": len(samples),
        "public_groups": len({value.group_id for value in samples}),
        "outer_fit_samples": len(fit_ids),
        "outer_fit_groups": len(fit_groups),
        "outer_holdout_samples": len(holdout_ids),
        "outer_holdout_groups": len(holdout_groups),
        "outer_fit_ids_sha256": _ids_hash(list(fit_ids)),
        "outer_holdout_ids_sha256": _ids_hash(list(holdout_ids)),
        "outer_fit_groups_sha256": _ids_hash(list(fit_groups)),
        "outer_holdout_groups_sha256": _ids_hash(list(holdout_groups)),
        "sample_overlap": 0,
        "group_overlap": 0,
    }
    for key in (
        "outer_fit_ids_sha256",
        "outer_holdout_ids_sha256",
        "outer_fit_groups_sha256",
        "outer_holdout_groups_sha256",
    ):
        _require(outer_identity[key] == EXPECTED_HASHES[key], f"{key} drift")

    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        _require(sample_id and sample_id not in row_by_id, "public row identity drift")
        row_by_id[sample_id] = row
    _require(set(row_by_id) == {str(value.sample_id) for value in samples},
             "loader/raw public roster mismatch")
    fit_rows = [row for row in rows if str(row.get("sample_id")) in fit_ids]
    _require(len(fit_rows) == EXPECTED_OUTER_FIT[0], "fit package row drift")

    inputs = output_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    fit_manifest = inputs / "syncg_outer_fit_only.jsonl"
    fit_protocol = fit_manifest.with_name(fit_manifest.name + ".protocol.json")
    atomic_jsonl(fit_manifest, fit_rows)
    fit_sidecar = {
        "protocol": "syncg_official_split_v1",
        "split": "train",
        "release_identity_verified": True,
        "expected_rows": len(fit_rows),
        "expected_sample_ids_sha256": syncg_sample_ids_sha256(
            str(row["sample_id"]) for row in fit_rows
        ),
        "derived_fit_only": True,
        "source_manifest_sha256": sha256_file(PUBLIC_MANIFEST),
        "source_manifest_protocol_sha256": sha256_file(PUBLIC_MANIFEST_PROTOCOL),
        "outer_split_seed": OUTER_SPLIT_SEED,
        "outer_holdout_rows_embedded": 0,
        "outer_holdout_ids_embedded": 0,
    }
    atomic_json(fit_protocol, fit_sidecar)
    loaded_fit, _ = load_syncg_manifest(fit_manifest, expected_split="train")
    discovery = _derive_inner(loaded_fit)
    inner_train_ids = {
        str(value.sample.sample_id) for value in discovery.inner_train
    }
    inner_screen_ids = {
        str(value.sample.sample_id) for value in discovery.inner_screen
    }
    inner_train_rows = [
        row for row in fit_rows if str(row.get("sample_id")) in inner_train_ids
    ]
    inner_screen_rows = [
        row for row in fit_rows if str(row.get("sample_id")) in inner_screen_ids
    ]
    _require(
        len(inner_train_rows) == EXPECTED_INNER_TRAIN[0]
        and len(inner_screen_rows) == EXPECTED_INNER_SCREEN[0],
        "inner partition manifest row drift",
    )
    _require(
        {str(row["sample_id"]) for row in inner_train_rows} == inner_train_ids
        and {str(row["sample_id"]) for row in inner_screen_rows} == inner_screen_ids,
        "inner partition manifest identity drift",
    )

    def write_partition_manifest(
        filename: str,
        partition_rows: Sequence[Mapping[str, Any]],
        *,
        partition: str,
        expected_ids_sha256: str,
        expected_groups_sha256: str,
    ) -> tuple[Path, Path]:
        path = inputs / filename
        sidecar_path = path.with_name(path.name + ".protocol.json")
        atomic_jsonl(path, partition_rows)
        atomic_json(
            sidecar_path,
            {
                "protocol": "syncg_official_split_v1",
                "split": "train",
                "release_identity_verified": True,
                "expected_rows": len(partition_rows),
                "expected_sample_ids_sha256": syncg_sample_ids_sha256(
                    str(row["sample_id"]) for row in partition_rows
                ),
                "derived_fit_only": True,
                "inner_partition": partition,
                "source_fit_manifest_sha256": sha256_file(fit_manifest),
                "expected_inner_ids_sha256": expected_ids_sha256,
                "expected_inner_groups_sha256": expected_groups_sha256,
                "opposite_partition_rows_embedded": 0,
                "outer_holdout_rows_embedded": 0,
            },
        )
        loaded, _ = load_syncg_manifest(path, expected_split="train")
        _require(
            len(loaded) == len(partition_rows)
            and _ids_hash([str(value.sample_id) for value in loaded])
            == expected_ids_sha256,
            f"{partition} manifest loader drift",
        )
        return path, sidecar_path

    inner_train_manifest, inner_train_protocol = write_partition_manifest(
        "syncg_inner_train_only.jsonl",
        inner_train_rows,
        partition="inner_train",
        expected_ids_sha256=EXPECTED_HASHES["inner_train_ids_sha256"],
        expected_groups_sha256=EXPECTED_HASHES["inner_train_groups_sha256"],
    )
    inner_screen_manifest, inner_screen_protocol = write_partition_manifest(
        "syncg_inner_screen_only.jsonl",
        inner_screen_rows,
        partition="inner_screen",
        expected_ids_sha256=EXPECTED_HASHES["inner_screen_ids_sha256"],
        expected_groups_sha256=EXPECTED_HASHES["inner_screen_groups_sha256"],
    )
    roster_path = inputs / "inner_rosters.label_free.json"
    roster = {
        # This is a byte-identical migration of the partition roster frozen in
        # P0.  The protocol field identifies the protocol that created the
        # split; changing it to the P1 runner identity would silently change
        # the otherwise identical input artifact and break migration closure.
        "protocol": P0_PROTOCOL_ID,
        "outer_fit_sample_ids": sorted(str(value.sample.sample_id) for value in (*discovery.inner_train, *discovery.inner_screen)),
        "inner_train_sample_ids": sorted(str(value.sample.sample_id) for value in discovery.inner_train),
        "inner_train_group_ids": sorted({str(value.sample.group_id) for value in discovery.inner_train}),
        "inner_screen_sample_ids": sorted(str(value.sample.sample_id) for value in discovery.inner_screen),
        "inner_screen_group_ids": sorted({str(value.sample.group_id) for value in discovery.inner_screen}),
    }
    atomic_json(roster_path, roster)
    return {
        "source_public_manifest": str(PUBLIC_MANIFEST.resolve()),
        "source_public_manifest_sha256": sha256_file(PUBLIC_MANIFEST),
        "source_public_protocol_sha256": sha256_file(PUBLIC_MANIFEST_PROTOCOL),
        "source_content_inventory": str(CONTENT_INVENTORY.resolve()),
        "source_content_inventory_sha256": sha256_file(CONTENT_INVENTORY),
        "fit_manifest": str(fit_manifest.resolve()),
        "fit_manifest_sha256": sha256_file(fit_manifest),
        "fit_manifest_protocol": str(fit_protocol.resolve()),
        "fit_manifest_protocol_sha256": sha256_file(fit_protocol),
        "inner_rosters": str(roster_path.resolve()),
        "inner_rosters_sha256": sha256_file(roster_path),
        "inner_train_manifest": str(inner_train_manifest.resolve()),
        "inner_train_manifest_sha256": sha256_file(inner_train_manifest),
        "inner_train_manifest_protocol": str(inner_train_protocol.resolve()),
        "inner_train_manifest_protocol_sha256": sha256_file(inner_train_protocol),
        "inner_screen_manifest": str(inner_screen_manifest.resolve()),
        "inner_screen_manifest_sha256": sha256_file(inner_screen_manifest),
        "inner_screen_manifest_protocol": str(inner_screen_protocol.resolve()),
        "inner_screen_manifest_protocol_sha256": sha256_file(inner_screen_protocol),
        "outer_identity": outer_identity,
        "inner_identity": dict(discovery.identity),
        "source_manifest_protocol": source_protocol,
    }


def build_protocol_candidate(output_root: Path) -> dict[str, Any]:
    configure_determinism(PEPD_SEED)
    _verify_static_external_artifacts()
    root = _guard_output_root(output_root, allow_create=True)
    _require(not root.exists(), f"candidate output root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    try:
        inputs = _write_fit_package(root)
        sources = source_manifest()
        v2_protocol = strict_json(V2_PROTOCOL)
        runtime_signature = inspect.signature(CAGHV5SolverGatedResidual.forward)
        _require(
            set(runtime_signature.parameters)
            == {"self", "image", "final_to_isotropic", "crop_affine", "arm"},
            "solver-gated runtime signature drift",
        )
        tests = _expected_test_sources()
        candidate = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "frozen_public_inner_screen",
            "adaptive_source_evidence": True,
            "outer_common_holdout_evaluation_authorized": False,
            "external_or_field_evaluation_authorized": False,
            "output_root": str(root),
            "inputs": inputs,
            "materialization": {
                "canonical_roi_contract": v2_protocol["materialization"]["crop_contract"],
                "canonical_roi_contract_sha256": v2_protocol["materialization"]["crop_contract_sha256"],
                "public_16000_roi_root_sha256": v2_protocol["materialization"]["roi_sha256"],
                "public_16000_target_root_sha256": v2_protocol["materialization"]["target_sha256"],
                "shared_native_roi_contract_sha256": v2_protocol["materialization"]["shared_native_roi_contract_sha256"],
                "crop_expansion": 1.0,
                "letterbox": False,
                "padding": False,
                "black_border": False,
                "training_augmentation_border": "reflect_101 only after canonical native ROI",
                "pepd_heatmap_coordinate_interval": [0.0, 63.0],
                "pepd_normalized_homography": "F_second @ inverse(F_first), first-to-second",
                "pepd_equivariance_pivot_normalizer": 63.0,
                "pepd_equivariance_ray_length_normalized": 0.25,
            },
            "adaptive_lineage": {
                "frozen_v2_protocol": str(V2_PROTOCOL.resolve()),
                "frozen_v2_protocol_sha256": EXPECTED_V2_PROTOCOL_SHA256,
                "reviewed_v2_source_statistics": str(V2_SOURCE_STATISTICS.resolve()),
                "reviewed_v2_source_statistics_sha256": EXPECTED_V2_SOURCE_STATISTICS_SHA256,
                "claim_limit": "locked adaptive source confirmation, not pristine method-family confirmation",
            },
            "infrastructure_migration": _expected_migration_contract(root),
            "model": {
                "name": "solver-anchored reliability-gated residual CAGH-V5",
                "progress_bins": 72,
                "anchor_pivot": [0.5, 0.5],
                "anchor_evidence": "raw keypoint_angle_evidence from predicted reference geometry",
                "legacy_pepd_evidence_enabled": False,
                "legacy_mask_evidence_enabled": False,
                "residual_standardization": "z(E)=(E-mean(E))/sqrt(mean((E-mean(E))^2)+1e-6)",
                "residual_formula": "R_branch=s(K)*(z(E_branch)-z(K))",
                "fusion_formula": "K+tanh(a_p)*stopgrad(g_p*R_p)+tanh(a_m)*stopgrad(g_m*R_m)",
                "signed_gain_range": [-1.0, 1.0],
                "gain_initialization": 0.0,
                "gate_range": [0.0, 1.0],
                "absolute_logit_clip": None,
                "description_limit": "relative-RMS residual with bounded signed gain; not a convex mixture or absolute bound",
                "nan_policy": "optional nonfinite branch gate=0 and anchor exact; nonfinite anchor/fusion => uniform invalid",
                "arms": list(ARM_NAMES),
                "runtime_signature": str(runtime_signature),
                "runtime_model_inputs": ["image", "final_to_isotropic", "crop_affine", "arm"],
                "runtime_forbidden": sorted(LABEL_TOKENS | {"sample_id", "group_id"}),
            },
            "training": _expected_training_contract(),
            "screen_evaluation": {
                "cohort": "inner_screen only",
                "conditions": ["clean", "frozen_stress"],
                "seed": EVALUATION_SEED,
                "stress_epoch": STRESS_EPOCH,
                "failure_penalty": 1.0,
                "all_arms_one_shared_batch_pass": True,
                "prediction_bundle_label_free": True,
                "score_bundle_separate": True,
                "outer_holdout_images_read": 0,
            },
            "promotion": promotion_contract(),
            "source_manifest": {
                "sha256": sources,
                "manifest_sha256": canonical_sha256(sources),
            },
            "test_sources": tests,
            "environment": environment_identity(),
        }
        _validate_protocol_semantics(candidate, root)
        candidate_path = root / "protocol_candidate.json"
        atomic_json(candidate_path, candidate)
        return candidate
    except BaseException:
        # Preserve any partial root for forensic inspection; never delete data.
        raise


def freeze_protocol_candidate(output_root: Path) -> Path:
    """Freeze the already reviewed candidate byte-for-byte, once."""

    root = _guard_output_root(output_root, allow_create=False)
    _require(root == DEFAULT_OUTPUT_ROOT.resolve(strict=True), "freeze root drift")
    candidate_path = root / "protocol_candidate.json"
    protocol_path = root / "protocol.json"
    receipt_path = root / "protocol_freeze_receipt.json"
    _require(candidate_path.is_file(), "missing protocol candidate")
    _require(not protocol_path.exists() and not receipt_path.exists(), "protocol already frozen")
    candidate = strict_json(candidate_path)
    _require(candidate.get("protocol") == PROTOCOL, "candidate identity drift")
    _require(candidate.get("status") == "frozen_public_inner_screen", "candidate status drift")
    _require(str(candidate.get("output_root")) == str(root), "candidate root drift")
    _require(dict(candidate.get("promotion") or {}) == promotion_contract(), "candidate gates drift")
    actual_sources = source_manifest()
    _require(
        dict(candidate["source_manifest"]["sha256"]) == actual_sources
        and candidate["source_manifest"]["manifest_sha256"]
        == canonical_sha256(actual_sources),
        "candidate source drift",
    )
    _require(dict(candidate["environment"]) == environment_identity(), "candidate environment drift")
    _validate_protocol_semantics(candidate, root)
    atomic_json_new(protocol_path, candidate)
    _require(protocol_path.read_bytes() == candidate_path.read_bytes(), "freeze byte parity failed")
    receipt = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "frozen",
        "candidate": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "canonical_protocol_sha256": canonical_sha256(candidate),
        "source_manifest_sha256": canonical_sha256(actual_sources),
        "frozen_at_unix": time.time(),
    }
    atomic_json_new(receipt_path, receipt)
    return protocol_path


def load_protocol(*, scope: str) -> Mapping[str, Any]:
    _require(
        scope in {"inner_train", "inner_screen", "no_partition", "p0_import"},
        "invalid protocol scope",
    )
    _require(PROTOCOL_PATH.is_file(), f"missing frozen protocol: {PROTOCOL_PATH}")
    protocol = strict_json(PROTOCOL_PATH)
    root = _guard_output_root(PROTOCOL_PATH.parent, allow_create=False)
    candidate_path = root / "protocol_candidate.json"
    receipt_path = root / "protocol_freeze_receipt.json"
    _require(candidate_path.is_file() and receipt_path.is_file(), "protocol freeze closure missing")
    receipt = strict_json(receipt_path)
    _require(
        sha256_file(candidate_path) == receipt.get("candidate_sha256")
        and sha256_file(PROTOCOL_PATH) == receipt.get("protocol_sha256")
        and candidate_path.read_bytes() == PROTOCOL_PATH.read_bytes(),
        "frozen protocol byte/hash drift",
    )
    _require(
        canonical_sha256(protocol) == receipt.get("canonical_protocol_sha256"),
        "frozen protocol canonical drift",
    )
    _require(protocol.get("schema_version") == 1, "protocol schema drift")
    _require(protocol.get("protocol") == PROTOCOL, "protocol identity drift")
    _require(protocol.get("status") == "frozen_public_inner_screen", "protocol not frozen")
    _require(protocol.get("outer_common_holdout_evaluation_authorized") is False,
             "protocol authorizes outer holdout")
    _require(protocol.get("external_or_field_evaluation_authorized") is False,
             "protocol authorizes restricted data")
    root = _guard_output_root(Path(str(protocol["output_root"])), allow_create=False)
    _require(str(root) == str(DEFAULT_OUTPUT_ROOT.resolve(strict=False)), "output root drift")
    _validate_protocol_semantics(protocol, root, scope=scope)
    inputs = protocol["inputs"]
    common_artifacts = (
        ("source_content_inventory", "source_content_inventory_sha256"),
    )
    partition_artifacts = {
        "inner_train": (
            ("inner_train_manifest", "inner_train_manifest_sha256"),
            ("inner_train_manifest_protocol", "inner_train_manifest_protocol_sha256"),
        ),
        "inner_screen": (
            ("inner_screen_manifest", "inner_screen_manifest_sha256"),
            ("inner_screen_manifest_protocol", "inner_screen_manifest_protocol_sha256"),
        ),
    }
    provenance_artifacts = (
        ("source_public_manifest", "source_public_manifest_sha256"),
        ("fit_manifest", "fit_manifest_sha256"),
        ("fit_manifest_protocol", "fit_manifest_protocol_sha256"),
        ("inner_rosters", "inner_rosters_sha256"),
    )
    selected_artifacts = common_artifacts + (
        () if scope in {"no_partition", "p0_import"} else partition_artifacts[scope]
    )
    for path_key, hash_key in selected_artifacts:
        path = Path(str(inputs[path_key]))
        allowed_root = root if path.drive.casefold() == "c:" else PROJECT_ROOT
        guarded = _guard_public_path(path, allowed_root, label=f"input.{path_key}")
        _require(guarded.is_file(), f"input artifact invalid: {path_key}")
        _require(sha256_file(path) == inputs[hash_key], f"input artifact drift: {path_key}")
    if scope is None:
        _require(sha256_file(PUBLIC_MANIFEST_PROTOCOL) == inputs["source_public_protocol_sha256"],
                 "source public protocol drift")
    actual_sources = source_manifest()
    _require(actual_sources == dict(protocol["source_manifest"]["sha256"]),
             "transitive source hash drift")
    _require(canonical_sha256(actual_sources) == protocol["source_manifest"]["manifest_sha256"],
             "source manifest root drift")
    for relative, expected in dict(protocol["test_sources"]).items():
        _require(sha256_file(PROJECT_ROOT / relative) == expected, f"test source drift: {relative}")
    _require(environment_identity() == dict(protocol["environment"]), "environment drift")
    return protocol


def load_fit_discovery(
    protocol: Mapping[str, Any], *, scope: str
) -> ScreenDiscovery:
    """Load exactly one inner partition into the current process.

    Training and preflight use ``scope='inner_train'`` and therefore never
    parse the screen manifest.  The once-only evaluation uses
    ``scope='inner_screen'`` and never parses the training annotations.
    """

    _require(scope in {"inner_train", "inner_screen"}, "invalid discovery scope")
    inputs = protocol["inputs"]
    artifact_root = Path(str(protocol["output_root"]))
    manifest = _guard_public_path(
        Path(str(inputs[f"{scope}_manifest"])), artifact_root,
        label=f"{scope}.manifest",
    )
    manifest_protocol = _guard_public_path(
        Path(str(inputs[f"{scope}_manifest_protocol"])), artifact_root,
        label=f"{scope}.manifest_protocol",
    )
    _require(
        sha256_file(manifest) == inputs[f"{scope}_manifest_sha256"]
        and sha256_file(manifest_protocol)
        == inputs[f"{scope}_manifest_protocol_sha256"],
        f"{scope} manifest artifact drift",
    )
    strict_jsonl(manifest)
    strict_json(manifest_protocol)
    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    identity = dict(inputs["inner_identity"])
    expected_count = (
        EXPECTED_INNER_TRAIN if scope == "inner_train" else EXPECTED_INNER_SCREEN
    )
    ids_key = f"{scope}_ids_sha256"
    groups_key = f"{scope}_groups_sha256"
    _require(
        (len(samples), len({str(value.group_id) for value in samples}))
        == expected_count,
        f"{scope} inventory drift",
    )
    _require(
        _ids_hash([str(value.sample_id) for value in samples]) == identity[ids_key]
        and _ids_hash(sorted({str(value.group_id) for value in samples}))
        == identity[groups_key],
        f"{scope} roster hash drift",
    )
    image_root = (PROJECT_ROOT / "datasets/SyncG/syncG/images/train").resolve(strict=True)
    annotation_root = (
        PROJECT_ROOT / "datasets/SyncG/syncG/annotations/train"
    ).resolve(strict=True)
    for sample in samples:
        _require(sample.dataset == "SyncG" and sample.split == "train", "non-public sample")
        _guard_public_path(Path(sample.image_path), image_root, label=f"{sample.sample_id}.image")
        annotation = Path(str(sample.metadata.get("annotation_path") or ""))
        _guard_public_path(annotation, annotation_root, label=f"{sample.sample_id}.annotation")
    records = _weighted_records(samples, scope)
    return ScreenDiscovery(
        inner_train=records if scope == "inner_train" else (),
        inner_screen=records if scope == "inner_screen" else (),
        identity=identity,
    )


def _foreign_cuda_process_rows(
    output: str,
    *,
    current_pid: int,
    driver_model: str,
) -> list[str]:
    """Reject real foreign CUDA contexts without treating WDDM UI rows as jobs."""

    model = str(driver_model).strip().upper()
    _require(model in {"WDDM", "TCC", "N/A"}, "invalid NVIDIA driver model")
    foreign: list[str] = []
    for raw_line in str(output).splitlines():
        text_value = raw_line.strip()
        if not text_value or "No running processes" in text_value:
            continue
        fields = next(csv.reader([text_value], skipinitialspace=True))
        _require(len(fields) == 3, f"unparseable nvidia-smi process row: {text_value}")
        try:
            pid = int(fields[0].strip())
        except ValueError as error:
            raise ValueError(f"unparseable nvidia-smi process pid: {text_value}") from error
        process_name = fields[1].strip()
        memory_text = fields[2].strip().replace("[", "").replace("]", "")
        memory_is_numeric = False
        try:
            int(memory_text)
            memory_is_numeric = True
        except ValueError:
            _require(
                memory_text.casefold() in {"n/a", "not supported"},
                f"unparseable nvidia-smi memory field: {text_value}",
            )
        executable = Path(process_name).name.casefold()
        is_compute_context = (
            model != "WDDM"
            or memory_is_numeric
            or executable not in WDDM_GRAPHICS_ALLOWLIST
        )
        if pid != int(current_pid) and is_compute_context:
            foreign.append(text_value)
    return foreign


def _require_cuda_runtime(device: torch.device) -> dict[str, Any]:
    _require(device.type == "cuda", "formal execution requires CUDA")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "formal execution requires exactly one CUDA GPU")
    name = torch.cuda.get_device_name(device)
    _require(name == "NVIDIA GeForce RTX 4060", f"unexpected CUDA device: {name}")
    driver_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_model.current",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    driver_rows = [row.strip().upper() for row in driver_result.stdout.splitlines() if row.strip()]
    _require(len(driver_rows) == 1, f"unexpected NVIDIA driver-model rows: {driver_rows}")
    driver_model = driver_rows[0]
    _require(driver_model in {"WDDM", "TCC", "N/A"}, f"unexpected driver model: {driver_model}")
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=15
    )
    foreign = _foreign_cuda_process_rows(
        result.stdout,
        current_pid=os.getpid(),
        driver_model=driver_model,
    )
    _require(not foreign, f"foreign CUDA compute process present: {foreign}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    _require(free_bytes >= 6 * 1024**3, "insufficient free RTX 4060 memory")
    return {
        "device": str(device),
        "device_name": name,
        "free_memory_bytes": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
        "driver_model": driver_model,
        "process_filter_policy": GPU_PROCESS_FILTER_POLICY,
        "wddm_graphics_allowlist": list(WDDM_GRAPHICS_ALLOWLIST),
        "foreign_compute_processes": foreign,
    }


def _verify_partition_content(
    protocol: Mapping[str, Any],
    records_to_verify: Sequence[PublicRecord],
    *,
    scope: str,
    workers: int,
) -> dict[str, Any]:
    _require(scope in {"inner_train", "inner_screen"}, "invalid content scope")
    inventory = strict_json(Path(str(protocol["inputs"]["source_content_inventory"])))
    records = inventory.get("records")
    _require(isinstance(records, list) and len(records) == EXPECTED_PUBLIC[0],
             "content inventory record count drift")
    by_id: dict[str, Mapping[str, Any]] = {}
    for value in records:
        _require(isinstance(value, Mapping), "content inventory row is not an object")
        sample_id = str(value.get("sample_id") or "")
        _require(sample_id and sample_id not in by_id, "content inventory identity drift")
        by_id[sample_id] = value

    def verify(record: PublicRecord) -> tuple[str, int, int, str, str]:
        sample = record.sample
        sample_id = str(sample.sample_id)
        expected = by_id.get(sample_id)
        _require(expected is not None, f"missing content inventory row: {sample_id}")
        image = Path(sample.image_path).resolve(strict=True)
        annotation = Path(str(sample.metadata.get("annotation_path") or "")).resolve(strict=True)
        _require(
            image == (PROJECT_ROOT / str(expected["image_path"])).resolve(strict=True)
            and annotation
            == (PROJECT_ROOT / str(expected["annotation_path"])).resolve(strict=True),
            f"content path drift: {sample_id}",
        )
        image_size = image.stat().st_size
        annotation_size = annotation.stat().st_size
        image_hash = sha256_file(image)
        annotation_hash = sha256_file(annotation)
        _require(
            image_size == int(expected["image_size_bytes"])
            and annotation_size == int(expected["annotation_size_bytes"])
            and image_hash == str(expected["image_sha256"])
            and annotation_hash == str(expected["annotation_sha256"]),
            f"content byte drift: {sample_id}",
        )
        return sample_id, image_size, annotation_size, image_hash, annotation_hash

    worker_count = max(1, min(8, int(workers)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        verified = list(executor.map(verify, records_to_verify))
    verified.sort(key=lambda value: value[0])
    expected = EXPECTED_INNER_TRAIN if scope == "inner_train" else EXPECTED_INNER_SCREEN
    _require(len(verified) == expected[0], "verified content roster drift")
    return {
        "scope": scope,
        "samples": len(verified),
        "image_files": len(verified),
        "annotation_files": len(verified),
        "image_bytes": int(sum(value[1] for value in verified)),
        "annotation_bytes": int(sum(value[2] for value in verified)),
        "receipt_sha256": canonical_sha256(verified),
        "inner_screen_files_read": 0 if scope == "inner_train" else 2 * len(verified),
        "outer_holdout_files_read": 0,
        "restricted_files_read": 0,
    }


def run_preflight(
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    *,
    root: Path,
    device: torch.device,
    workers: int,
) -> Path:
    """One public inner-train smoke without optimizer updates or screen reads."""

    output = root / "preflight.json"
    _require(not output.exists(), "preflight already exists")
    _require(discovery.inner_train and not discovery.inner_screen,
             "preflight discovery is not inner-train-only")
    _require(not (root / "locks").exists(), "preflight found existing locks")
    _require(not list(root.rglob("*.tmp.*")), "preflight found temporary artifacts")
    disk = shutil.disk_usage(root)
    _require(disk.free >= 20 * 1024**3, "C: free space below 20 GiB")
    configure_determinism(PEPD_SEED)
    gpu = _require_cuda_runtime(device)
    content_verification = _verify_partition_content(
        protocol,
        discovery.inner_train,
        scope="inner_train",
        workers=max(1, int(workers)),
    )
    pepd_terminal, pepd_payload, _, pepd_import_receipt = _validated_pepd_import(
        protocol, discovery, root
    )

    records = discovery.inner_train[:2]
    _require(len(records) == 2, "preflight needs two inner-train records")
    pair_dataset = CanonicalPEPDPairDataset(
        records,
        seed=PEPD_SEED,
        augmentation=_augmentation(protocol, enabled=True),
        public_image_root=(
            PROJECT_ROOT / "datasets/SyncG/syncG/images/train"
        ).resolve(strict=True),
    )
    pair_dataset.set_epoch(1)
    first_row = pair_dataset[0]
    second_row = pair_dataset[0]
    first_receipt = tensor_receipt(first_row)
    _require(first_receipt == tensor_receipt(second_row),
             "paired input determinism drift")
    del first_row, second_row
    pair_loader = _loader(
        pair_dataset,
        batch_size=2,
        workers=0,
        shuffle=False,
        seed=PEPD_SEED,
        device=device,
    )
    pair_batch = next(iter(pair_loader))
    pepd = build_probabilistic_pivot_direction_model(
        angle_bins=72, imagenet_pretrained=False
    )
    pepd.load_state_dict(pepd_payload["model_state"], strict=True)
    pepd = pepd.to(device).eval()
    joined = pepd(
        torch.cat((pair_batch["image"], pair_batch["paired_image"]), dim=0).to(device)
    )
    pair_device = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in pair_batch.items()
    }
    pepd_loss, pepd_parts = _pepd_losses(
        tuple(value.float() for value in joined), pair_device, protocol
    )
    pepd_state = _cpu_state(pepd)
    _require(
        _tensor_mapping_sha256(pepd_state) == EXPECTED_P0_PEPD_MODEL_STATE_SHA256,
        "preflight imported P0 PEPD state drift",
    )
    pepd_loss_value = float(pepd_loss.detach().cpu())
    del joined, pepd_loss, pair_device, pair_batch, pair_loader, pair_dataset, pepd
    gc.collect()
    torch.cuda.empty_cache()

    canonical_dataset = CanonicalTightROIDataset(
        records,
        training=True,
        seed=PARENT_SEED,
        augmentation=_augmentation(protocol, enabled=True),
    )
    canonical_dataset.set_epoch(1)
    canonical_loader = _loader(
        canonical_dataset,
        batch_size=2,
        workers=0,
        shuffle=False,
        seed=PARENT_SEED,
        device=device,
    )
    batch = next(iter(canonical_loader))
    parent = CAGHV5SolverGatedResidual(progress_bins=72, dropout=0.10)
    parent.parent.load_pepd_state_dict(pepd_state)
    parent_trainable = _configure_solver_parent(parent)
    parent = parent.to(device)
    _set_parent_train_mode(parent)
    parent_output, parent_loss, parent_parts = _parent_forward_loss(parent, batch, device)
    parent_training_view = _solver_core_training_view(parent_output)
    _require(
        torch.equal(
            parent_training_view.cagh.progress_log_probability,
            parent_output.progress_log_probability,
        )
        and torch.equal(
            parent_training_view.cagh.expected_progress,
            parent_output.expected_progress,
        )
        and torch.equal(parent_training_view.cagh.valid, parent_output.valid),
        "preflight solver-parent/runtime-anchor parity drift",
    )
    direct_progress_loss, direct_progress_parts = solver_gated_progress_loss(
        parent_output,
        target_progress=batch["target_progress"].to(device),
        group_weight=batch["group_weight"].to(device),
    )
    _require(
        torch.equal(
            parent_parts["cagh_final_soft_ce"],
            direct_progress_parts["final_soft_ce"],
        )
        and torch.equal(
            parent_parts["cagh_final_expected_smooth_l1"],
            direct_progress_parts["final_expected_smooth_l1"],
        ),
        "preflight parent final loss is not the runtime solver-core loss",
    )
    parent_loss.backward()
    _require_trainable_gradients_finite(
        parent,
        parent_trainable,
        label="parent preflight",
    )
    _require_tensor_mapping_finite(
        parent.parent.state_dict(),
        label="parent preflight state",
    )
    parent_state = _cpu_state(parent.parent)
    parent_loss_value = float(parent_loss.detach().cpu())
    del (
        parent_output,
        parent_training_view,
        parent_parts,
        direct_progress_loss,
        direct_progress_parts,
        parent_loss,
        parent,
        pepd_state,
    )
    gc.collect()
    torch.cuda.empty_cache()

    gated = CAGHV5SolverGatedResidual(progress_bins=72, dropout=0.10)
    gated.load_parent_state_dict(parent_state)
    trainable = gated.configure_trainable_arm(FULL_GATED)
    _require(tuple(trainable) == ("pepd_residual_gain", "mask_residual_gain"),
             "preflight gain trainable roster drift")
    gated = gated.to(device).train()
    parent_digest = _module_state_sha256(gated.parent)
    gated_output = gated(
        batch["image"].to(device),
        batch["final_to_isotropic"].to(device),
        batch["crop_affine"].to(device),
        arm=FULL_GATED,
    )
    gated_loss, _ = solver_gated_progress_loss(
        gated_output,
        target_progress=batch["target_progress"].to(device),
        group_weight=batch["group_weight"].to(device),
    )
    gated_loss.backward()
    _require(
        gated.pepd_residual_gain.grad is not None
        and gated.mask_residual_gain.grad is not None
        and bool(torch.isfinite(gated.pepd_residual_gain.grad))
        and bool(torch.isfinite(gated.mask_residual_gain.grad)),
        "gain preflight gradient is non-finite",
    )
    _require(_module_state_sha256(gated.parent) == parent_digest,
             "preflight mutated frozen parent")
    gated_loss_value = float(gated_loss.detach().cpu())
    del gated_output, gated_loss, gated, parent_state, batch, canonical_loader, canonical_dataset
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "preflight_pass",
        "protocol_snapshot_sha256": canonical_sha256(protocol),
        "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
        "optimizer_steps": 0,
        "public_inner_train_images_read": 6,
        "p0_archive_files_read": 0,
        "archived_inner_screen_manifest_files_read": 0,
        "inner_screen_manifest_rows_read": 0,
        "inner_screen_images_read": 0,
        "outer_holdout_manifest_rows_read": 0,
        "outer_holdout_images_read": 0,
        "restricted_images_read": 0,
        "paired_input_receipt": first_receipt,
        "pepd_mode": "imported_frozen_p0",
        "pepd_import_receipt_sha256": sha256_file(
            root / "imports/pepd_p0/pepd_import_complete.json"
        ),
        "pepd_terminal_sha256": sha256_file(pepd_terminal),
        "pepd_model_state_sha256": _tensor_mapping_sha256(pepd_payload["model_state"]),
        "pepd_import_optimizer_steps": int(pepd_import_receipt["optimizer_steps"]),
        "pepd_import_images_read": int(pepd_import_receipt["images_read"]),
        "pepd_import_annotations_read": int(pepd_import_receipt["annotations_read"]),
        "pepd_loss": pepd_loss_value,
        "pepd_loss_components": sorted(pepd_parts),
        "parent_loss": parent_loss_value,
        "parent_solver_core_exact_parity": True,
        "parent_all_trainable_gradients_present_and_finite": True,
        "parent_state_all_finite": True,
        "parent_trainable_parameters": list(parent_trainable),
        "gated_loss": gated_loss_value,
        "disk_free_bytes": int(disk.free),
        "gpu": gpu,
        "inner_train_content_verification": content_verification,
    }
    atomic_json(output, result)
    receipt_path = root / "preflight_receipt.json"
    _require(not receipt_path.exists(), "preflight receipt already exists")
    atomic_json_new(
        receipt_path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "sealed_preflight_pass",
            "preflight": str(output),
            "preflight_sha256": sha256_file(output),
            "protocol_snapshot_sha256": canonical_sha256(protocol),
            "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
        },
    )
    return output


def _augmentation(protocol: Mapping[str, Any], *, enabled: bool) -> PhotoAugmentation:
    if not enabled:
        return PhotoAugmentation.disabled()
    value = PhotoAugmentation(**dict(protocol["training"]["augmentation"]))
    value.validate()
    return value


def _loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = torch.as_tensor(values[name]).detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _module_state_sha256(module: torch.nn.Module) -> str:
    return _tensor_mapping_sha256(_cpu_state(module))


def _require_tensor_mapping_finite(
    values: Mapping[str, torch.Tensor], *, label: str
) -> None:
    _require(bool(values), f"{label}: tensor mapping is empty")
    for name, value in values.items():
        _require(torch.is_tensor(value), f"{label}: non-tensor state: {name}")
        tensor = torch.as_tensor(value)
        if tensor.is_floating_point() or tensor.is_complex():
            _require(
                bool(torch.isfinite(tensor).all()),
                f"{label}: non-finite state tensor: {name}",
            )


def _require_trainable_gradients_finite(
    module: torch.nn.Module,
    trainable_names: Sequence[str],
    *,
    label: str,
) -> None:
    expected = tuple(map(str, trainable_names))
    named = dict(module.named_parameters())
    actual = tuple(name for name, parameter in named.items() if parameter.requires_grad)
    _require(actual == expected, f"{label}: trainable parameter roster drift")
    gradients: list[torch.Tensor] = []
    for name in expected:
        gradient = named[name].grad
        _require(gradient is not None, f"{label}: missing gradient: {name}")
        gradients.append(gradient)
    try:
        norms = torch.stack(tuple(torch._foreach_norm(gradients)))
    except (RuntimeError, TypeError):
        norms = torch.stack(
            tuple(torch.linalg.vector_norm(value.detach().reshape(-1)) for value in gradients)
        )
    if bool(torch.isfinite(norms).all()):
        return
    for name, gradient in zip(expected, gradients, strict=True):
        _require(
            bool(torch.isfinite(gradient).all()),
            f"{label}: non-finite gradient: {name}",
        )
    raise ValueError(f"{label}: non-finite gradient norm")


def _require_named_parameters_finite(
    module: torch.nn.Module,
    parameter_names: Sequence[str],
    *,
    label: str,
) -> None:
    named = dict(module.named_parameters())
    names = tuple(map(str, parameter_names))
    values: list[torch.Tensor] = []
    for name in names:
        _require(name in named, f"{label}: missing parameter: {name}")
        values.append(named[name])
    try:
        norms = torch.stack(tuple(torch._foreach_norm(values)))
    except (RuntimeError, TypeError):
        norms = torch.stack(
            tuple(torch.linalg.vector_norm(value.detach().reshape(-1)) for value in values)
        )
    if bool(torch.isfinite(norms).all()):
        return
    for name, value in zip(names, values, strict=True):
        _require(
            bool(torch.isfinite(value).all()),
            f"{label}: non-finite parameter: {name}",
        )
    raise ValueError(f"{label}: non-finite parameter norm")


def _pepd_state_sha256(model: CAGHV5UnifiedModel) -> str:
    prefixes = model.cagh._PEPD_PREFIXES
    values = {
        name: value
        for name, value in model.cagh.state_dict().items()
        if name.startswith(prefixes)
    }
    _require(bool(values), "PEPD state selection is empty")
    return _tensor_mapping_sha256(values)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _stage_paths(root: Path, stage: str) -> tuple[Path, Path, Path]:
    directory = root / "runs" / stage
    return directory, directory / "journal.pt", directory / "terminal.pt"


def _publication_stage_state(directory: Path, *, label: str) -> str:
    if not directory.exists():
        return "EMPTY"
    _require(directory.is_dir() and not directory.is_symlink() and not _is_reparse(directory),
             f"{label} stage path is not a regular directory")
    entries = list(directory.iterdir())
    _require(
        all(
            path.is_file() and not path.is_symlink() and not _is_reparse(path)
            for path in entries
        ),
        f"{label} stage contains non-regular/reparse artifacts",
    )
    names = {path.name for path in entries}
    states = {
        frozenset(): "EMPTY",
        frozenset({"journal.pt"}): "JOURNALED",
        frozenset({"journal.pt", "terminal.pt"}): "TERMINAL_PUBLISHED",
        frozenset({"journal.pt", "terminal.pt", "summary.json"}): "SUMMARY_COMPLETE",
    }
    _require(frozenset(names) in states, f"invalid {label} publication state")
    return states[frozenset(names)]


def _parent_stage_state(directory: Path) -> str:
    return _publication_stage_state(directory, label="parent")


def _gain_stage_state(directory: Path) -> str:
    return _publication_stage_state(directory, label="gain")


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    if os.name == "nt":
        # Python's os.kill(pid, 0) is not a POSIX-style read-only probe on
        # Windows. Query the process handle without termination rights.
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        error_invalid_parameter = 87
        error_not_found = 1168
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        handle = open_process(process_query_limited_information, False, int(pid))
        if not handle:
            error = ctypes.get_last_error()
            if error in (error_invalid_parameter, error_not_found):
                return False
            # Access denied or an unknown query error is fail-safe alive: never
            # move a lock when liveness cannot be disproved read-only.
            return True
        try:
            code = wintypes.DWORD()
            if not get_exit_code(handle, ctypes.byref(code)):
                return True
            return int(code.value) == still_active
        finally:
            close_handle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_lock(root: Path, stage: str, *, resume: bool = False) -> Path:
    locks = root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    _require(
        locks.is_dir() and not locks.is_symlink() and not _is_reparse(locks),
        "lock root is not a regular directory",
    )
    lock = locks / f"{stage}.lock"
    if lock.exists():
        _require(
            lock.is_file() and not lock.is_symlink() and not _is_reparse(lock),
            f"lock is not a regular file: {lock}",
        )
        _require(resume, f"active/orphan lock exists: {lock}")
        lines = lock.read_text(encoding="utf-8").splitlines()
        fields = dict(line.split("=", 1) for line in lines if "=" in line)
        pid = int(fields.get("pid", "-1"))
        _require(pid > 0 and not _pid_alive(pid), f"run process is still alive: pid={pid}")
        forensic = locks / "forensics"
        forensic.mkdir(parents=True, exist_ok=True)
        _require(
            forensic.is_dir()
            and not forensic.is_symlink()
            and not _is_reparse(forensic),
            "lock forensic root is not a regular directory",
        )
        destination = forensic / f"{lock.name}.stale.{int(time.time())}"
        _require(not destination.exists(), "stale lock forensic destination exists")
        os.replace(lock, destination)
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\nstarted={time.time()}\n")
    return lock


def _complete_lock(lock: Path) -> None:
    lock.unlink(missing_ok=False)


def _require_preflight(
    protocol: Mapping[str, Any],
    root: Path,
    *,
    discovery: ScreenDiscovery | None = None,
    workers: int = 1,
) -> Mapping[str, Any]:
    path = root / "preflight.json"
    receipt_path = root / "preflight_receipt.json"
    _require(path.is_file(), "training requires completed preflight")
    _require(receipt_path.is_file(), "training requires sealed preflight receipt")
    value = strict_json(path)
    receipt = strict_json(receipt_path)
    _require(set(receipt) == {
        "schema_version", "protocol", "status", "preflight", "preflight_sha256",
        "protocol_snapshot_sha256", "source_manifest_sha256",
    }, "preflight receipt schema drift")
    _require(
        receipt.get("schema_version") == 1
        and receipt.get("protocol") == PROTOCOL
        and receipt.get("status") == "sealed_preflight_pass"
        and Path(str(receipt.get("preflight"))).resolve(strict=True) == path.resolve(strict=True)
        and receipt.get("preflight_sha256") == sha256_file(path),
        "preflight receipt/hash drift",
    )
    _require(set(value) == {
        "schema_version", "protocol", "status", "protocol_snapshot_sha256",
        "source_manifest_sha256", "optimizer_steps", "public_inner_train_images_read",
        "p0_archive_files_read", "archived_inner_screen_manifest_files_read",
        "inner_screen_manifest_rows_read", "inner_screen_images_read",
        "outer_holdout_manifest_rows_read", "outer_holdout_images_read",
        "restricted_images_read", "paired_input_receipt", "pepd_mode",
        "pepd_import_receipt_sha256", "pepd_terminal_sha256",
        "pepd_model_state_sha256", "pepd_import_optimizer_steps",
        "pepd_import_images_read", "pepd_import_annotations_read", "pepd_loss",
        "pepd_loss_components", "parent_loss", "gated_loss", "disk_free_bytes",
        "gpu", "inner_train_content_verification", "parent_solver_core_exact_parity",
        "parent_all_trainable_gradients_present_and_finite",
        "parent_state_all_finite", "parent_trainable_parameters",
    }, "preflight schema drift")
    _require(value.get("status") == "preflight_pass", "preflight did not pass")
    _require(value.get("schema_version") == 1 and value.get("protocol") == PROTOCOL,
             "preflight identity drift")
    _require(value.get("protocol_snapshot_sha256") == canonical_sha256(protocol),
             "preflight protocol drift")
    _require(value.get("source_manifest_sha256") == protocol["source_manifest"]["manifest_sha256"],
             "preflight source drift")
    _require(value.get("optimizer_steps") == 0, "preflight performed optimizer steps")
    pepd_terminal, pepd_payload, _, pepd_import = _validated_pepd_import(
        protocol, discovery, root
    )
    _require(
        value.get("pepd_mode") == "imported_frozen_p0"
        and value.get("pepd_import_receipt_sha256")
        == sha256_file(root / "imports/pepd_p0/pepd_import_complete.json")
        and value.get("pepd_terminal_sha256") == sha256_file(pepd_terminal)
        and value.get("pepd_terminal_sha256") == EXPECTED_P0_PEPD_TERMINAL_SHA256
        and value.get("pepd_model_state_sha256")
        == _tensor_mapping_sha256(pepd_payload["model_state"])
        and value.get("pepd_model_state_sha256") == EXPECTED_P0_PEPD_MODEL_STATE_SHA256
        and value.get("pepd_import_optimizer_steps") == pepd_import["optimizer_steps"] == 0
        and value.get("pepd_import_images_read") == pepd_import["images_read"] == 0
        and value.get("pepd_import_annotations_read")
        == pepd_import["annotations_read"] == 0,
        "preflight imported P0 PEPD binding drift",
    )
    _require(value.get("outer_holdout_images_read") == 0, "preflight touched outer holdout")
    _require(value.get("inner_screen_images_read") == 0, "preflight touched inner screen")
    _require(
        value.get("inner_screen_manifest_rows_read") == 0
        and value.get("p0_archive_files_read") == 0
        and value.get("archived_inner_screen_manifest_files_read") == 0
        and value.get("outer_holdout_manifest_rows_read") == 0
        and value.get("outer_holdout_images_read") == 0
        and value.get("restricted_images_read") == 0,
        "preflight data firewall drift",
    )
    _require(
        value.get("public_inner_train_images_read") == 6
        and isinstance(value.get("disk_free_bytes"), int)
        and value["disk_free_bytes"] >= 20 * 1024**3,
        "preflight public smoke/disk drift",
    )
    for name in ("pepd_loss", "parent_loss", "gated_loss"):
        _require(math.isfinite(float(value.get(name))), f"preflight {name} is non-finite")
    _require(
        value.get("pepd_loss_components") == sorted({
            "first_supervised", "second_supervised", "equivariance", "pivot",
            "direction", "equivariance_pivot", "equivariance_direction",
            "equivariance_valid_fraction",
        }),
        "preflight PEPD loss component roster drift",
    )
    _require(
        value.get("parent_solver_core_exact_parity") is True
        and value.get("parent_all_trainable_gradients_present_and_finite") is True
        and value.get("parent_state_all_finite") is True
        and isinstance(value.get("parent_trainable_parameters"), list)
        and bool(value["parent_trainable_parameters"])
        and all(str(name).startswith("parent.") for name in value["parent_trainable_parameters"])
        and not any("evidence_refiner" in str(name) for name in value["parent_trainable_parameters"]),
        "preflight solver-parent alignment drift",
    )
    _require(
        isinstance(value.get("paired_input_receipt"), str)
        and len(value["paired_input_receipt"]) == 64,
        "preflight paired receipt drift",
    )
    gpu = value.get("gpu")
    _require(
        isinstance(gpu, Mapping)
        and set(gpu) == {
            "device", "device_name", "free_memory_bytes", "total_memory_bytes",
            "driver_model", "process_filter_policy", "wddm_graphics_allowlist",
            "foreign_compute_processes",
        }
        and gpu.get("device") == "cuda:0"
        and gpu.get("device_name") == "NVIDIA GeForce RTX 4060"
        and gpu.get("driver_model") in {"WDDM", "TCC", "N/A"}
        and gpu.get("process_filter_policy") == GPU_PROCESS_FILTER_POLICY
        and gpu.get("wddm_graphics_allowlist") == list(WDDM_GRAPHICS_ALLOWLIST)
        and gpu.get("foreign_compute_processes") == []
        and int(gpu.get("free_memory_bytes", 0)) >= 6 * 1024**3,
        "preflight GPU identity drift",
    )
    content = value.get("inner_train_content_verification")
    _require(
        isinstance(content, Mapping)
        and set(content) == {
            "scope", "samples", "image_files", "annotation_files", "image_bytes",
            "annotation_bytes", "receipt_sha256", "inner_screen_files_read",
            "outer_holdout_files_read", "restricted_files_read",
        }
        and content.get("scope") == "inner_train"
        and content.get("samples") == EXPECTED_INNER_TRAIN[0]
        and content.get("image_files") == EXPECTED_INNER_TRAIN[0]
        and content.get("annotation_files") == EXPECTED_INNER_TRAIN[0]
        and int(content.get("image_bytes", 0)) > 0
        and int(content.get("annotation_bytes", 0)) > 0
        and isinstance(content.get("receipt_sha256"), str)
        and len(content["receipt_sha256"]) == 64
        and content.get("inner_screen_files_read") == 0
        and content.get("outer_holdout_files_read") == 0
        and content.get("restricted_files_read") == 0,
        "preflight content verification drift",
    )
    _require(
        receipt.get("protocol_snapshot_sha256") == canonical_sha256(protocol)
        and receipt.get("source_manifest_sha256")
        == protocol["source_manifest"]["manifest_sha256"],
        "preflight receipt binding drift",
    )
    if discovery is not None:
        _require(discovery.inner_train and not discovery.inner_screen,
                 "training preflight recheck requires inner-train-only discovery")
        current = _verify_partition_content(
            protocol, discovery.inner_train, scope="inner_train", workers=workers
        )
        _require(dict(content) == current, "inner-train content changed after preflight")
    return value


def _pepd_source_training_contract(
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the frozen P0 loss contract nested inside the P1 import wrapper."""

    training = protocol.get("training")
    _require(isinstance(training, Mapping), "missing P1 training contract")
    stage = training.get("pepd")
    expected = _expected_training_contract()["pepd"]
    stage_wrapper = dict(stage) if isinstance(stage, Mapping) else {}
    source = stage_wrapper.pop("source_training_contract", None)
    expected_wrapper = dict(expected)
    expected_source = expected_wrapper.pop("source_training_contract")
    _require(
        isinstance(stage, Mapping)
        and stage_wrapper == expected_wrapper
        and stage.get("mode") == "imported_frozen_p0"
        and stage.get("training_in_p1_authorized") is False,
        "P1 imported PEPD training contract drift",
    )
    _require(
        isinstance(source, Mapping)
        and dict(source) == expected_source,
        "P0 PEPD source loss contract drift",
    )
    return source


def _pepd_losses(
    joined: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    count = int(batch["image"].shape[0])
    first = tuple(value[:count] for value in joined)
    second = tuple(value[count:] for value in joined)
    config = _pepd_source_training_contract(protocol)
    first_loss, first_parts = probabilistic_direction_loss(
        *first,
        batch["heatmap"],
        batch["direction"],
        pivot_weight=float(config["pivot_loss_weight"]),
        bin_weight=float(config["bin_loss_weight"]),
        vector_weight=float(config["vector_loss_weight"]),
        soft_target_sigma_bins=float(config["soft_target_sigma_bins"]),
    )
    second_loss, second_parts = probabilistic_direction_loss(
        *second,
        batch["paired_heatmap"],
        batch["paired_direction"],
        pivot_weight=float(config["pivot_loss_weight"]),
        bin_weight=float(config["bin_loss_weight"]),
        vector_weight=float(config["vector_loss_weight"]),
        soft_target_sigma_bins=float(config["soft_target_sigma_bins"]),
    )
    first_prediction = decode_probabilistic_pivot_direction(*first)
    second_prediction = decode_probabilistic_pivot_direction(*second)
    first_pivot = soft_pivot_coordinates(first[0]) / 63.0
    second_pivot = soft_pivot_coordinates(second[0]) / 63.0
    expected_pivot, expected_direction, valid_h = transform_pivot_direction(
        first_pivot,
        first_prediction.direction,
        batch["homography"],
        ray_length=0.25,
    )
    valid_equivariance = valid_h & first_prediction.valid & second_prediction.valid
    if bool(valid_equivariance.any()):
        equivariance_pivot = torch.nn.functional.smooth_l1_loss(
            second_pivot[valid_equivariance],
            expected_pivot[valid_equivariance],
        )
        equivariance_direction = torch.mean(
            1.0
            - torch.sum(
                second_prediction.direction[valid_equivariance]
                * expected_direction[valid_equivariance],
                dim=1,
            )
        )
    else:
        equivariance_pivot = first[0].sum() * 0.0
        equivariance_direction = first[1].sum() * 0.0
    equivariant = (
        equivariance_direction
        + float(config["equivariance_pivot_weight"]) * equivariance_pivot
    )
    total = (
        first_loss
        + float(config["paired_supervision_weight"]) * second_loss
        + float(config["equivariance_weight"]) * equivariant
    )
    parts = {
        "first_supervised": first_loss.detach(),
        "second_supervised": second_loss.detach(),
        "equivariance": equivariant.detach(),
        "pivot": 0.5 * (first_parts["pivot_loss"] + second_parts["pivot_loss"]),
        "direction": 0.5 * (first_parts["direction_loss"] + second_parts["direction_loss"]),
        "equivariance_pivot": equivariance_pivot.detach(),
        "equivariance_direction": equivariance_direction.detach(),
        "equivariance_valid_fraction": valid_equivariance.float().mean().detach(),
    }
    _require(bool(torch.isfinite(total)), "PEPD loss is non-finite")
    _require(all(bool(torch.isfinite(value)) for value in parts.values()),
             "PEPD loss component is non-finite")
    return total, parts


def train_pepd(
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    *,
    root: Path,
    workers: int,
    device: torch.device,
    resume: bool,
) -> Path:
    raise ValueError(
        "P1 forbids PEPD training; use the audited new-only P0 import before preflight"
    )
    _require_preflight(
        protocol, root, discovery=discovery, workers=max(1, int(workers))
    )
    config = protocol["training"]["pepd"]
    stage = f"pepd_inner_seed_{int(config['seed'])}"
    directory, journal_path, terminal_path = _stage_paths(root, stage)
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        _require(resume, f"PEPD run already exists: {summary_path}")
        summary = strict_json(summary_path)
        _require(summary.get("status") == "complete", "PEPD summary incomplete")
        _require(sha256_file(terminal_path) == summary["terminal_sha256"], "PEPD terminal drift")
        return terminal_path
    directory.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(root, stage, resume=resume)
    completed = False
    try:
        configure_determinism(int(config["seed"]))
        dataset = CanonicalPEPDPairDataset(
            discovery.inner_train,
            seed=int(config["seed"]),
            augmentation=_augmentation(protocol, enabled=True),
            public_image_root=(PROJECT_ROOT / "datasets/SyncG/syncG/images/train").resolve(strict=True),
        )
        model = build_probabilistic_pivot_direction_model(
            angle_bins=72, imagenet_pretrained=True
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["epochs"]),
            eta_min=float(config["eta_min"]),
        )
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=device.type == "cuda" and bool(config["amp"]),
            init_scale=512.0,
        )
        history: list[dict[str, Any]] = []
        start_epoch = 1
        global_step = 0
        if journal_path.is_file():
            _require(resume, "orphan PEPD journal requires --resume")
            checkpoint = torch.load(journal_path, map_location="cpu", weights_only=False)
            _require(checkpoint.get("protocol_snapshot_sha256") == canonical_sha256(protocol),
                     "PEPD resume protocol drift")
            _require(checkpoint.get("source_manifest_sha256") == protocol["source_manifest"]["manifest_sha256"],
                     "PEPD resume source drift")
            _require(checkpoint.get("inner_train_ids_sha256") == discovery.identity["inner_train_ids_sha256"],
                     "PEPD resume roster drift")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _optimizer_to_device(optimizer, device)
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            history = list(checkpoint["history"])
            start_epoch = int(checkpoint["completed_epoch"]) + 1
            global_step = int(checkpoint["global_step"])
        elif resume:
            _require(not terminal_path.exists(), "PEPD terminal exists without summary")

        started = time.time()
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            configure_determinism(int(config["seed"]) + epoch * 1_000_003)
            dataset.set_epoch(epoch)
            loader = _loader(
                dataset,
                batch_size=int(config["batch_size"]),
                workers=workers,
                shuffle=True,
                seed=int(config["seed"]) + epoch,
                device=device,
            )
            model.train()
            totals = Counter()
            samples = 0
            steps = 0
            skipped = 0
            augmented = 0
            for cpu_batch in loader:
                batch = {
                    key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in cpu_batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                    joined = model(torch.cat((batch["image"], batch["paired_image"]), dim=0))
                loss, parts = _pepd_losses(
                    tuple(value.float() for value in joined), batch, protocol
                )
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    skipped += 1
                else:
                    steps += 1
                count = int(batch["image"].shape[0])
                samples += count
                augmented += int(
                    ((batch["augmentation_code"] != 0) | (batch["paired_augmentation_code"] != 0)).sum()
                )
                totals["loss"] += float(loss.detach()) * count
                for name, value in parts.items():
                    totals[name] += float(value) * count
            _require(samples == len(dataset), f"PEPD epoch{epoch}: sample inventory drift")
            scheduler.step()
            global_step += steps
            row = {
                "epoch": epoch,
                "terminal_epoch": int(config["epochs"]),
                "samples": samples,
                "optimizer_steps": steps,
                "skipped_optimizer_steps": skipped,
                "global_step": global_step,
                "learning_rate_after_step": float(optimizer.param_groups[0]["lr"]),
                "augmented_fraction": augmented / samples,
                **{name: value / samples for name, value in totals.items()},
            }
            history.append(row)
            checkpoint = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "role": "inner_train_pepd",
                "seed": int(config["seed"]),
                "terminal_epoch": int(config["epochs"]),
                "completed_epoch": epoch,
                "global_step": global_step,
                "checkpoint_selection": "none; fixed terminal",
                "protocol_snapshot_sha256": canonical_sha256(protocol),
                "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
                "inner_train_ids_sha256": discovery.identity["inner_train_ids_sha256"],
                "inner_screen_images_read": 0,
                "outer_holdout_images_read": 0,
                "model_state": _cpu_state(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
            }
            atomic_torch(journal_path, checkpoint)
            print(
                f"solver-gated PEPD epoch={epoch}/{config['epochs']} "
                f"loss={row['loss']:.6f} lr={row['learning_rate_after_step']:.3e}",
                flush=True,
            )
        _require(len(history) == int(config["epochs"]), "PEPD terminal history incomplete")
        payload = torch.load(journal_path, map_location="cpu", weights_only=False)
        atomic_torch(terminal_path, payload)
        summary = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "role": "inner_train_pepd",
            "seed": int(config["seed"]),
            "terminal_epoch": int(config["epochs"]),
            "checkpoint_selection": "none; terminal only",
            "history": history,
            "terminal": str(terminal_path),
            "terminal_sha256": sha256_file(terminal_path),
            "model_state_sha256": _tensor_mapping_sha256(payload["model_state"]),
            "inner_train_ids_sha256": discovery.identity["inner_train_ids_sha256"],
            "inner_screen_images_read": 0,
            "outer_holdout_images_read": 0,
            "elapsed_seconds": time.time() - started,
            "device_name": torch.cuda.get_device_name(device),
            "protocol_snapshot_sha256": canonical_sha256(protocol),
        }
        atomic_json(summary_path, summary)
        completed = True
        return terminal_path
    finally:
        if completed:
            _complete_lock(lock)


def _configure_solver_parent(model: CAGHV5SolverGatedResidual) -> tuple[str, ...]:
    """Open only modules used by the exact runtime solver-core training view."""

    model.configure_trainable_arm(SOLVER_CORE)
    model.parent.configure_trainable_arm(NO_PEPD_DIRECTION)
    # The runtime anchor is raw keypoint evidence.  The legacy evidence refiner
    # is deliberately disconnected and therefore must not appear trainable.
    for parameter in model.parent.cagh.evidence_refiner.parameters():
        parameter.requires_grad_(False)
    _require(
        not model.pepd_residual_gain.requires_grad
        and not model.mask_residual_gain.requires_grad
        and float(model.pepd_residual_gain.detach()) == 0.0
        and float(model.mask_residual_gain.detach()) == 0.0,
        "solver parent residual gains are not frozen at exact zero",
    )
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    _require(bool(trainable), "solver parent trainable roster is empty")
    _require(
        all(name.startswith("parent.") for name in trainable)
        and not any("evidence_refiner" in name for name in trainable),
        "solver parent trainable roster escaped the aligned core",
    )
    return trainable


def _solver_core_training_view(
    output: Any,
) -> CAGHV5UnifiedOutputs:
    """Adapt the exact runtime solver-core posterior to the existing joint loss.

    ``NO_PEPD_DIRECTION`` is used only to select the already frozen auxiliary
    loss roster (including mask-support supervision).  No unified-model forward
    is called here.  The final CE/expected terms consume the runtime raw
    keypoint posterior byte-for-byte.
    """

    training_core = output.core._replace(
        progress_log_probability=output.progress_log_probability,
        expected_progress=output.expected_progress,
        unified_angle_logits=output.core_angle_evidence,
        valid=output.valid,
    )
    return CAGHV5UnifiedOutputs(
        arm=NO_PEPD_DIRECTION,
        progress_log_probability=output.progress_log_probability,
        expected_progress=output.expected_progress,
        valid=output.valid,
        pointer_valid=output.pointer_valid,
        reference_valid=output.reference_valid,
        geometry_solver_valid=output.geometry_solver_valid,
        fusion_solver_valid=output.fusion_solver_valid,
        features=output.features,
        pepd=output.pepd,
        reference=output.reference,
        cagh=training_core,
        reference_start_angle=output.reference_start_angle,
        reference_range_angle=output.reference_range_angle,
        reference_radii=output.reference_radii,
    )


def _parent_forward_loss(
    model: CAGHV5SolverGatedResidual,
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, torch.Tensor, Mapping[str, torch.Tensor]]:
    # The model callback receives pixels and preprocessing transforms only.
    output = model(
        batch["image"].to(device, non_blocking=True),
        batch["final_to_isotropic"].to(device, non_blocking=True),
        batch["crop_affine"].to(device, non_blocking=True),
        arm=SOLVER_CORE,
    )
    training_view = _solver_core_training_view(output)
    loss, parts = cagh_v5_unified_loss(
        training_view,
        target_progress=batch["target_progress"].to(device, non_blocking=True),
        target_tip_xy=batch["pointer_tip"].to(device, non_blocking=True),
        target_tail_xy=batch["pointer_tail"].to(device, non_blocking=True),
        target_endpoints=batch["endpoints"].to(device, non_blocking=True),
        target_tick_heatmap=batch["tick_heatmap"].to(device, non_blocking=True),
        target_start_angle=batch["gt_start"].to(device, non_blocking=True),
        target_range_angle=batch["gt_range"].to(device, non_blocking=True),
        group_weight=batch["group_weight"].to(device, non_blocking=True),
    )
    _require(bool(torch.isfinite(loss)), "parent loss is non-finite")
    return output, loss, parts


def _set_parent_train_mode(model: CAGHV5SolverGatedResidual) -> None:
    model.train()
    model.parent.reference_head.train()
    for name in model.parent.cagh._NEW_MODULE_NAMES:
        module = getattr(model.parent.cagh, name)
        module.train(any(parameter.requires_grad for parameter in module.parameters()))


def _validated_pepd_terminal(
    protocol: Mapping[str, Any], discovery: ScreenDiscovery, root: Path
) -> tuple[Path, Mapping[str, Any]]:
    terminal, payload, _, _ = _validated_pepd_import(protocol, discovery, root)
    return terminal, payload


PARENT_CHECKPOINT_KEYS = frozenset({
    "schema_version", "protocol", "role", "arm", "runtime_core_arm",
    "legacy_pepd_evidence_enabled", "legacy_mask_evidence_enabled",
    "parent_loss_contract_sha256", "gradient_policy",
    "terminal_state_all_finite", "optimizer_state_all_finite", "seed",
    "terminal_epoch", "completed_epoch", "global_step", "checkpoint_selection",
    "protocol_snapshot_sha256", "source_manifest_sha256", "preflight_sha256",
    "inner_train_content_receipt_sha256", "inner_train_ids_sha256",
    "pepd_terminal", "pepd_terminal_sha256", "pepd_import_receipt_sha256",
    "pepd_model_state_sha256", "frozen_pepd_state_sha256",
    "trainable_parameters", "model_state", "model_state_sha256",
    "optimizer_state", "history", "history_sha256", "elapsed_training_seconds",
    "inner_screen_images_read", "outer_holdout_images_read",
})
PARENT_SUMMARY_KEYS = frozenset({
    "schema_version", "protocol", "status", "role", "arm", "runtime_core_arm",
    "legacy_pepd_evidence_enabled", "legacy_mask_evidence_enabled",
    "parent_loss_contract_sha256", "gradient_policy", "terminal_state_all_finite",
    "optimizer_state_all_finite", "seed", "terminal_epoch", "checkpoint_selection",
    "trainable_parameters", "history", "history_sha256", "preflight_sha256",
    "inner_train_content_receipt_sha256", "pepd_terminal_sha256",
    "pepd_import_receipt_sha256", "pepd_model_state_sha256",
    "frozen_pepd_state_sha256", "journal", "journal_sha256", "terminal",
    "terminal_sha256", "model_state_sha256", "inner_train_ids_sha256",
    "inner_screen_images_read", "outer_holdout_images_read",
    "elapsed_training_seconds", "device_name", "protocol_snapshot_sha256",
    "source_manifest_sha256", "publication_state_machine",
    "recovered_terminal_without_summary",
})


def _validate_parent_history(
    history: Any,
    *,
    completed_epoch: int,
    terminal_epoch: int,
    frozen_pepd_state_sha256: str,
) -> int:
    _require(isinstance(history, list) and len(history) == completed_epoch,
             "parent history length drift")
    expected_row_keys = {
        "epoch", "terminal_epoch", "samples", "optimizer_steps", "global_step",
        "loss", "augmented_fraction", "frozen_pepd_state_sha256",
    }
    cumulative = 0
    expected_steps = math.ceil(EXPECTED_INNER_TRAIN[0] / 16)
    for epoch, row in enumerate(history, start=1):
        _require(isinstance(row, Mapping) and set(row) == expected_row_keys,
                 f"parent history schema drift: epoch{epoch}")
        cumulative += expected_steps
        _require(
            row.get("epoch") == epoch
            and row.get("terminal_epoch") == terminal_epoch
            and row.get("samples") == EXPECTED_INNER_TRAIN[0]
            and row.get("optimizer_steps") == expected_steps
            and row.get("global_step") == cumulative
            and row.get("frozen_pepd_state_sha256") == frozen_pepd_state_sha256
            and math.isfinite(float(row.get("loss")))
            and 0.0 <= float(row.get("augmented_fraction")) <= 1.0,
            f"parent history semantic drift: epoch{epoch}",
        )
    return cumulative


def _validate_parent_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    root: Path,
    pepd_terminal: Path,
    frozen_pepd_state_sha256: str,
    trainable: Sequence[str],
    require_terminal: bool,
) -> None:
    config = protocol["training"]["parent"]
    _require(set(payload) == PARENT_CHECKPOINT_KEYS, "parent checkpoint schema drift")
    completed_epoch = int(payload.get("completed_epoch", -1))
    terminal_epoch = int(config["epochs"])
    _require(1 <= completed_epoch <= terminal_epoch, "parent completed epoch drift")
    if require_terminal:
        _require(completed_epoch == terminal_epoch, "parent terminal epoch incomplete")
    expected_preflight = root / "preflight.json"
    preflight = strict_json(expected_preflight)
    import_receipt = root / "imports/pepd_p0/pepd_import_complete.json"
    _require(
        payload.get("schema_version") == 2
        and payload.get("protocol") == PROTOCOL
        and payload.get("role") == "inner_train_solver_parent"
        and payload.get("arm") == PARENT_TRAINING_ARM
        and payload.get("runtime_core_arm") == SOLVER_CORE
        and payload.get("legacy_pepd_evidence_enabled") is False
        and payload.get("legacy_mask_evidence_enabled") is False
        and payload.get("parent_loss_contract_sha256") == canonical_sha256(config["loss"])
        and payload.get("gradient_policy") == PARENT_GRADIENT_POLICY
        and payload.get("terminal_state_all_finite") is True
        and payload.get("optimizer_state_all_finite") is True
        and int(payload.get("seed", -1)) == int(config["seed"])
        and int(payload.get("terminal_epoch", -1)) == terminal_epoch
        and payload.get("checkpoint_selection") == "none; fixed terminal"
        and payload.get("protocol_snapshot_sha256") == canonical_sha256(protocol)
        and payload.get("source_manifest_sha256")
        == protocol["source_manifest"]["manifest_sha256"]
        and payload.get("preflight_sha256") == sha256_file(expected_preflight)
        and payload.get("inner_train_content_receipt_sha256")
        == preflight["inner_train_content_verification"]["receipt_sha256"]
        and payload.get("inner_train_ids_sha256")
        == discovery.identity["inner_train_ids_sha256"]
        and Path(str(payload.get("pepd_terminal"))).resolve(strict=True)
        == pepd_terminal.resolve(strict=True)
        and payload.get("pepd_terminal_sha256") == sha256_file(pepd_terminal)
        and payload.get("pepd_terminal_sha256") == EXPECTED_P0_PEPD_TERMINAL_SHA256
        and payload.get("pepd_import_receipt_sha256") == sha256_file(import_receipt)
        and payload.get("pepd_model_state_sha256") == EXPECTED_P0_PEPD_MODEL_STATE_SHA256
        and payload.get("frozen_pepd_state_sha256") == frozen_pepd_state_sha256
        and tuple(payload.get("trainable_parameters") or ()) == tuple(trainable)
        and payload.get("inner_screen_images_read") == 0
        and payload.get("outer_holdout_images_read") == 0
        and math.isfinite(float(payload.get("elapsed_training_seconds")))
        and float(payload.get("elapsed_training_seconds")) >= 0.0,
        "parent checkpoint identity/provenance drift",
    )
    _require(isinstance(payload.get("model_state"), Mapping), "parent model state missing")
    _require_tensor_mapping_finite(payload["model_state"], label="parent checkpoint state")
    _require(
        _tensor_mapping_sha256(payload["model_state"])
        == payload.get("model_state_sha256"),
        "parent checkpoint model-state hash drift",
    )
    global_step = _validate_parent_history(
        payload["history"],
        completed_epoch=completed_epoch,
        terminal_epoch=terminal_epoch,
        frozen_pepd_state_sha256=frozen_pepd_state_sha256,
    )
    _require(payload.get("global_step") == global_step, "parent checkpoint global step drift")
    _require(config.get("optimizer") == "AdamW", "parent optimizer identity drift")
    _validate_adamw_optimizer_state(
        payload["optimizer_state"],
        model_state=payload["model_state"],
        trainable=trainable,
        global_step=global_step,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        param_group_contract=config["optimizer_param_group"],
        label="parent optimizer state",
    )
    _require(
        payload.get("history_sha256") == canonical_sha256(payload["history"]),
        "parent checkpoint history hash drift",
    )


def _parent_summary_value(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    root: Path,
    journal_path: Path,
    terminal_path: Path,
    device: torch.device,
    recovered: bool,
) -> dict[str, Any]:
    config = protocol["training"]["parent"]
    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "status": "complete",
        "role": "inner_train_solver_parent",
        "arm": PARENT_TRAINING_ARM,
        "runtime_core_arm": SOLVER_CORE,
        "legacy_pepd_evidence_enabled": False,
        "legacy_mask_evidence_enabled": False,
        "parent_loss_contract_sha256": canonical_sha256(config["loss"]),
        "gradient_policy": PARENT_GRADIENT_POLICY,
        "terminal_state_all_finite": True,
        "optimizer_state_all_finite": True,
        "seed": int(config["seed"]),
        "terminal_epoch": int(config["epochs"]),
        "checkpoint_selection": "none; terminal only",
        "trainable_parameters": list(payload["trainable_parameters"]),
        "history": list(payload["history"]),
        "history_sha256": canonical_sha256(payload["history"]),
        "preflight_sha256": sha256_file(root / "preflight.json"),
        "inner_train_content_receipt_sha256": strict_json(root / "preflight.json")[
            "inner_train_content_verification"
        ][
            "receipt_sha256"
        ],
        "pepd_terminal_sha256": payload["pepd_terminal_sha256"],
        "pepd_import_receipt_sha256": payload["pepd_import_receipt_sha256"],
        "pepd_model_state_sha256": payload["pepd_model_state_sha256"],
        "frozen_pepd_state_sha256": payload["frozen_pepd_state_sha256"],
        "journal": str(journal_path),
        "journal_sha256": sha256_file(journal_path),
        "terminal": str(terminal_path),
        "terminal_sha256": sha256_file(terminal_path),
        "model_state_sha256": payload["model_state_sha256"],
        "inner_train_ids_sha256": discovery.identity["inner_train_ids_sha256"],
        "inner_screen_images_read": 0,
        "outer_holdout_images_read": 0,
        "elapsed_training_seconds": float(payload["elapsed_training_seconds"]),
        "device_name": torch.cuda.get_device_name(device),
        "protocol_snapshot_sha256": canonical_sha256(protocol),
        "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
        "publication_state_machine": "EMPTY->JOURNALED->TERMINAL_PUBLISHED->SUMMARY_COMPLETE",
        "recovered_terminal_without_summary": bool(recovered),
    }


def train_parent(
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    *,
    root: Path,
    workers: int,
    device: torch.device,
    resume: bool,
) -> Path:
    preflight = _require_preflight(
        protocol, root, discovery=discovery, workers=max(1, int(workers))
    )
    pepd_terminal, pepd_payload = _validated_pepd_terminal(protocol, discovery, root)
    config = protocol["training"]["parent"]
    stage = f"solver_parent_seed_{int(config['seed'])}"
    directory, journal_path, terminal_path = _stage_paths(root, stage)
    summary_path = directory / "summary.json"
    stale_lock_evidence = (root / "locks" / f"{stage}.lock").is_file()
    lock = _acquire_lock(root, stage, resume=resume)
    completed = False
    try:
        directory.mkdir(parents=True, exist_ok=True)
        publication_state = _parent_stage_state(directory)
        configure_determinism(int(config["seed"]))
        model = CAGHV5SolverGatedResidual(progress_bins=72, dropout=0.10)
        model.parent.load_pepd_state_dict(pepd_payload["model_state"])
        trainable = _configure_solver_parent(model)
        model = model.to(device)
        initial_pepd_digest = _pepd_state_sha256(model.parent)
        pepd_import_receipt_sha256 = sha256_file(
            root / "imports/pepd_p0/pepd_import_complete.json"
        )
        dataset = CanonicalTightROIDataset(
            discovery.inner_train,
            training=True,
            seed=int(config["seed"]),
            augmentation=_augmentation(protocol, enabled=True),
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        history: list[dict[str, Any]] = []
        start_epoch = 1
        global_step = 0
        elapsed_before = 0.0
        if publication_state == "SUMMARY_COMPLETE":
            _require(resume,
                     "complete parent stage requires explicit --resume and full closure")
            validated, _, _ = _validated_parent_terminal(protocol, discovery, root)
            completed = True
            return validated
        if not resume:
            _require(publication_state == "EMPTY",
                     "fresh parent run refuses existing artifacts")
        else:
            _require(
                publication_state in {"JOURNALED", "TERMINAL_PUBLISHED"}
                or (publication_state == "EMPTY" and stale_lock_evidence),
                      "parent --resume requires a valid incomplete state")
        if journal_path.is_file():
            _require(resume, "orphan parent journal requires --resume")
            checkpoint = torch.load(journal_path, map_location="cpu", weights_only=False)
            _validate_parent_checkpoint_payload(
                checkpoint,
                protocol=protocol,
                discovery=discovery,
                root=root,
                pepd_terminal=pepd_terminal,
                frozen_pepd_state_sha256=initial_pepd_digest,
                trainable=trainable,
                require_terminal=terminal_path.is_file(),
            )
            model.parent.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _validate_loaded_adamw_optimizer(
                optimizer,
                module=model,
                trainable=trainable,
                global_step=int(checkpoint["global_step"]),
                label="parent resume optimizer",
            )
            history = list(checkpoint["history"])
            start_epoch = int(checkpoint["completed_epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            elapsed_before = float(checkpoint["elapsed_training_seconds"])
            _require(_pepd_state_sha256(model.parent) == initial_pepd_digest,
                     "parent resume mutated frozen PEPD")
            if terminal_path.is_file():
                terminal_payload = torch.load(
                    terminal_path, map_location="cpu", weights_only=False
                )
                _require(
                    _nested_equal(checkpoint, terminal_payload),
                    "parent journal/terminal semantic drift",
                )
                _validate_parent_checkpoint_payload(
                    terminal_payload,
                    protocol=protocol,
                    discovery=discovery,
                    root=root,
                    pepd_terminal=pepd_terminal,
                    frozen_pepd_state_sha256=initial_pepd_digest,
                    trainable=trainable,
                    require_terminal=True,
                )
                atomic_json_new(
                    summary_path,
                    _parent_summary_value(
                        terminal_payload,
                        protocol=protocol,
                        discovery=discovery,
                        root=root,
                        journal_path=journal_path,
                        terminal_path=terminal_path,
                        device=device,
                        recovered=True,
                    ),
                )
                completed = True
                return terminal_path

        started = time.time()
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            configure_determinism(int(config["seed"]) + epoch * 1_000_003)
            dataset.set_epoch(epoch)
            loader = _loader(
                dataset,
                batch_size=int(config["batch_size"]),
                workers=workers,
                shuffle=True,
                seed=int(config["seed"]) + epoch,
                device=device,
            )
            _set_parent_train_mode(model)
            total = 0.0
            samples = 0
            augmented = 0
            steps = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                output, loss, _ = _parent_forward_loss(model, batch, device)
                loss.backward()
                _require_trainable_gradients_finite(
                    model,
                    trainable,
                    label=f"parent epoch{epoch} step{steps + 1}",
                )
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    5.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                _require_named_parameters_finite(
                    model,
                    trainable,
                    label=f"parent epoch{epoch} step{steps + 1}",
                )
                count = int(output.expected_progress.shape[0])
                total += float(loss.detach()) * count
                samples += count
                steps += 1
                augmented += int((batch["augmentation_code"] != 0).sum())
            _require(samples == len(dataset), f"parent epoch{epoch}: sample inventory drift")
            _require(_pepd_state_sha256(model.parent) == initial_pepd_digest,
                     f"parent epoch{epoch}: frozen PEPD state drift")
            _require_tensor_mapping_finite(
                model.parent.state_dict(),
                label=f"parent epoch{epoch} state",
            )
            global_step += steps
            row = {
                "epoch": epoch,
                "terminal_epoch": int(config["epochs"]),
                "samples": samples,
                "optimizer_steps": steps,
                "global_step": global_step,
                "loss": total / samples,
                "augmented_fraction": augmented / samples,
                "frozen_pepd_state_sha256": initial_pepd_digest,
            }
            history.append(row)
            model_state = _cpu_state(model.parent)
            optimizer_state = _cpu_nested_state(optimizer.state_dict())
            _require_tensor_mapping_finite(
                model_state, label=f"parent epoch{epoch} checkpoint state"
            )
            _nested_tensors_finite(
                optimizer_state, label=f"parent epoch{epoch} optimizer state"
            )
            checkpoint = {
                "schema_version": 2,
                "protocol": PROTOCOL,
                "role": "inner_train_solver_parent",
                "arm": PARENT_TRAINING_ARM,
                "runtime_core_arm": SOLVER_CORE,
                "legacy_pepd_evidence_enabled": False,
                "legacy_mask_evidence_enabled": False,
                "parent_loss_contract_sha256": canonical_sha256(config["loss"]),
                "gradient_policy": PARENT_GRADIENT_POLICY,
                "terminal_state_all_finite": True,
                "optimizer_state_all_finite": True,
                "seed": int(config["seed"]),
                "terminal_epoch": int(config["epochs"]),
                "completed_epoch": epoch,
                "global_step": global_step,
                "checkpoint_selection": "none; fixed terminal",
                "protocol_snapshot_sha256": canonical_sha256(protocol),
                "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
                "preflight_sha256": sha256_file(root / "preflight.json"),
                "inner_train_content_receipt_sha256": preflight[
                    "inner_train_content_verification"
                ]["receipt_sha256"],
                "inner_train_ids_sha256": discovery.identity["inner_train_ids_sha256"],
                "pepd_terminal": str(pepd_terminal),
                "pepd_terminal_sha256": sha256_file(pepd_terminal),
                "pepd_import_receipt_sha256": pepd_import_receipt_sha256,
                "pepd_model_state_sha256": _tensor_mapping_sha256(
                    pepd_payload["model_state"]
                ),
                "frozen_pepd_state_sha256": initial_pepd_digest,
                "trainable_parameters": list(trainable),
                "model_state": model_state,
                "model_state_sha256": _tensor_mapping_sha256(model_state),
                "optimizer_state": optimizer_state,
                "history": history,
                "history_sha256": canonical_sha256(history),
                "elapsed_training_seconds": elapsed_before + time.time() - started,
                "inner_screen_images_read": 0,
                "outer_holdout_images_read": 0,
            }
            _validate_parent_checkpoint_payload(
                checkpoint,
                protocol=protocol,
                discovery=discovery,
                root=root,
                pepd_terminal=pepd_terminal,
                frozen_pepd_state_sha256=initial_pepd_digest,
                trainable=trainable,
                require_terminal=epoch == int(config["epochs"]),
            )
            atomic_torch(journal_path, checkpoint)
            print(
                f"solver parent epoch={epoch}/{config['epochs']} loss={row['loss']:.6f}",
                flush=True,
            )
        _require(len(history) == int(config["epochs"]), "parent terminal history incomplete")
        payload = torch.load(journal_path, map_location="cpu", weights_only=False)
        _validate_parent_checkpoint_payload(
            payload,
            protocol=protocol,
            discovery=discovery,
            root=root,
            pepd_terminal=pepd_terminal,
            frozen_pepd_state_sha256=initial_pepd_digest,
            trainable=trainable,
            require_terminal=True,
        )
        torch_new(terminal_path, payload)
        terminal_payload = torch.load(terminal_path, map_location="cpu", weights_only=False)
        _require(_nested_equal(payload, terminal_payload),
                 "published parent terminal semantic drift")
        atomic_json_new(
            summary_path,
            _parent_summary_value(
                terminal_payload,
                protocol=protocol,
                discovery=discovery,
                root=root,
                journal_path=journal_path,
                terminal_path=terminal_path,
                device=device,
                recovered=False,
            ),
        )
        completed = True
        return terminal_path
    finally:
        if completed:
            _complete_lock(lock)


def _validated_parent_terminal(
    protocol: Mapping[str, Any], discovery: ScreenDiscovery, root: Path
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    config = protocol["training"]["parent"]
    stage = f"solver_parent_seed_{int(config['seed'])}"
    directory, journal, terminal = _stage_paths(root, stage)
    summary_path = directory / "summary.json"
    _require(
        directory.is_dir()
        and {path.name for path in directory.iterdir() if path.is_file()}
        == {"journal.pt", "terminal.pt", "summary.json"}
        and not any(path.is_dir() for path in directory.iterdir()),
        "parent completed-stage file closure drift",
    )
    for path in (journal, terminal, summary_path):
        _require(path.is_file() and not _is_reparse(path),
                 f"parent completed artifact invalid: {path.name}")
    summary = strict_json(summary_path)
    _require(set(summary) == PARENT_SUMMARY_KEYS, "parent summary schema drift")
    pepd_terminal, pepd_payload = _validated_pepd_terminal(protocol, discovery, root)
    model = CAGHV5SolverGatedResidual(progress_bins=72, dropout=0.10)
    model.parent.load_pepd_state_dict(pepd_payload["model_state"])
    trainable = _configure_solver_parent(model)
    frozen_pepd_digest = _pepd_state_sha256(model.parent)
    payload = torch.load(terminal, map_location="cpu", weights_only=False)
    journal_payload = torch.load(journal, map_location="cpu", weights_only=False)
    _require(_nested_equal(payload, journal_payload),
             "parent terminal/journal semantic drift")
    _validate_parent_checkpoint_payload(
        payload,
        protocol=protocol,
        discovery=discovery,
        root=root,
        pepd_terminal=pepd_terminal,
        frozen_pepd_state_sha256=frozen_pepd_digest,
        trainable=trainable,
        require_terminal=True,
    )
    model.parent.load_state_dict(payload["model_state"], strict=True)
    _require(_pepd_state_sha256(model.parent) == frozen_pepd_digest,
             "validated parent mutated frozen PEPD")
    _require(
        summary.get("schema_version") == 2
        and summary.get("protocol") == PROTOCOL
        and summary.get("status") == "complete"
        and summary.get("role") == "inner_train_solver_parent"
        and summary.get("arm") == PARENT_TRAINING_ARM
        and summary.get("runtime_core_arm") == SOLVER_CORE
        and summary.get("legacy_pepd_evidence_enabled") is False
        and summary.get("legacy_mask_evidence_enabled") is False
        and summary.get("parent_loss_contract_sha256")
        == canonical_sha256(config["loss"])
        and summary.get("gradient_policy") == PARENT_GRADIENT_POLICY
        and summary.get("terminal_state_all_finite") is True
        and summary.get("optimizer_state_all_finite") is True
        and int(summary.get("seed", -1)) == int(config["seed"])
        and int(summary.get("terminal_epoch", -1)) == int(config["epochs"])
        and summary.get("checkpoint_selection") == "none; terminal only"
        and tuple(summary.get("trainable_parameters") or ()) == tuple(trainable)
        and summary.get("history") == payload["history"]
        and summary.get("history_sha256") == payload["history_sha256"]
        and summary.get("preflight_sha256") == payload["preflight_sha256"]
        and summary.get("inner_train_content_receipt_sha256")
        == payload["inner_train_content_receipt_sha256"]
        and summary.get("pepd_terminal_sha256") == payload["pepd_terminal_sha256"]
        and summary.get("pepd_import_receipt_sha256")
        == payload["pepd_import_receipt_sha256"]
        and summary.get("pepd_model_state_sha256") == payload["pepd_model_state_sha256"]
        and summary.get("frozen_pepd_state_sha256") == frozen_pepd_digest
        and Path(str(summary.get("journal"))).resolve(strict=True) == journal.resolve(strict=True)
        and summary.get("journal_sha256") == sha256_file(journal)
        and Path(str(summary.get("terminal"))).resolve(strict=True)
        == terminal.resolve(strict=True)
        and summary.get("terminal_sha256") == sha256_file(terminal)
        and summary.get("model_state_sha256") == payload["model_state_sha256"]
        and summary.get("publication_state_machine")
        == "EMPTY->JOURNALED->TERMINAL_PUBLISHED->SUMMARY_COMPLETE"
        and isinstance(summary.get("recovered_terminal_without_summary"), bool)
        and math.isfinite(float(summary.get("elapsed_training_seconds")))
        and float(summary.get("elapsed_training_seconds"))
        == float(payload["elapsed_training_seconds"])
        and summary.get("device_name") == "NVIDIA GeForce RTX 4060"
        and summary.get("protocol_snapshot_sha256") == canonical_sha256(protocol)
        and summary.get("source_manifest_sha256")
        == protocol["source_manifest"]["manifest_sha256"],
        "parent summary identity/alignment drift",
    )
    _require(summary.get("inner_train_ids_sha256") == discovery.identity["inner_train_ids_sha256"],
             "parent terminal roster drift")
    _require(summary.get("inner_screen_images_read") == 0
             and summary.get("outer_holdout_images_read") == 0,
             "parent summary data firewall drift")
    return terminal, payload, summary


def _gain_values(model: CAGHV5SolverGatedResidual) -> dict[str, float]:
    return {
        "pepd_residual_gain": float(model.pepd_residual_gain.detach().cpu()),
        "mask_residual_gain": float(model.mask_residual_gain.detach().cpu()),
        "pepd_residual_scale": float(torch.tanh(model.pepd_residual_gain.detach()).cpu()),
        "mask_residual_scale": float(torch.tanh(model.mask_residual_gain.detach()).cpu()),
    }


def _load_gain_state(model: CAGHV5SolverGatedResidual, value: Mapping[str, Any]) -> None:
    expected = {"pepd_residual_gain", "mask_residual_gain"}
    _require(set(value) == expected, "gain state schema drift")
    with torch.no_grad():
        model.pepd_residual_gain.fill_(float(value["pepd_residual_gain"]))
        model.mask_residual_gain.fill_(float(value["mask_residual_gain"]))
    _require(
        bool(torch.isfinite(model.pepd_residual_gain))
        and bool(torch.isfinite(model.mask_residual_gain)),
        "gain state is non-finite",
    )


GAIN_CHECKPOINT_KEYS = frozenset({
    "schema_version", "protocol", "role", "arm", "seed", "terminal_epoch",
    "completed_epoch", "global_step", "checkpoint_selection", "gradient_policy",
    "terminal_state_all_finite", "optimizer_state_all_finite",
    "protocol_snapshot_sha256", "source_manifest_sha256", "preflight_sha256",
    "inner_train_content_receipt_sha256", "parent_terminal",
    "parent_terminal_sha256", "parent_journal_sha256", "parent_summary_sha256",
    "parent_state_sha256", "inner_train_ids_sha256", "fresh_zero_initialization",
    "warm_started_from_full", "trainable_parameters", "gain_state",
    "optimizer_state", "history", "history_sha256", "elapsed_training_seconds",
    "inner_screen_images_read", "outer_holdout_images_read",
})
GAIN_SUMMARY_KEYS = frozenset({
    "schema_version", "protocol", "status", "role", "arm", "seed",
    "terminal_epoch", "optimizer_steps", "checkpoint_selection", "gradient_policy",
    "terminal_state_all_finite", "optimizer_state_all_finite",
    "fresh_zero_initialization", "warm_started_from_full", "trainable_parameters",
    "history", "history_sha256", "gains", "preflight_sha256",
    "inner_train_content_receipt_sha256", "parent_terminal",
    "parent_terminal_sha256", "parent_journal_sha256", "parent_summary_sha256",
    "parent_state_sha256", "journal", "journal_sha256", "terminal",
    "terminal_sha256", "inner_train_ids_sha256", "inner_screen_images_read",
    "outer_holdout_images_read", "elapsed_training_seconds", "device_name",
    "protocol_snapshot_sha256", "source_manifest_sha256",
    "publication_state_machine", "recovered_terminal_without_summary",
})


def _expected_gain_trainable(arm: str) -> tuple[str, ...]:
    _require(arm in ARM_NAMES, f"unknown gain arm: {arm}")
    return {
        FULL_GATED: ("pepd_residual_gain", "mask_residual_gain"),
        NO_PEPD_RESIDUAL: ("mask_residual_gain",),
        NO_MASK_RESIDUAL: ("pepd_residual_gain",),
        SOLVER_CORE: (),
    }[arm]


def _validate_gain_state(value: Any, *, arm: str, label: str) -> dict[str, float]:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"pepd_residual_gain", "mask_residual_gain"},
        f"{label}: gain state schema drift",
    )
    result = {name: float(raw) for name, raw in value.items()}
    _require(all(math.isfinite(raw) for raw in result.values()),
             f"{label}: non-finite gain state")
    if arm in {NO_PEPD_RESIDUAL, SOLVER_CORE}:
        _require(result["pepd_residual_gain"] == 0.0,
                 f"{label}: disabled PEPD gain is nonzero")
    if arm in {NO_MASK_RESIDUAL, SOLVER_CORE}:
        _require(result["mask_residual_gain"] == 0.0,
                 f"{label}: disabled mask gain is nonzero")
    return result


def _gain_telemetry_from_state(value: Mapping[str, Any]) -> dict[str, float]:
    state = {name: float(raw) for name, raw in value.items()}
    return {
        **state,
        "pepd_residual_scale": float(torch.tanh(
            torch.tensor(state["pepd_residual_gain"], dtype=torch.float32)
        )),
        "mask_residual_scale": float(torch.tanh(
            torch.tensor(state["mask_residual_gain"], dtype=torch.float32)
        )),
    }


def _validate_gain_telemetry(value: Any, *, arm: str, label: str) -> dict[str, float]:
    _require(
        isinstance(value, Mapping)
        and set(value) == {
            "pepd_residual_gain", "mask_residual_gain",
            "pepd_residual_scale", "mask_residual_scale",
        },
        f"{label}: gain telemetry schema drift",
    )
    state = _validate_gain_state(
        {name: value[name] for name in ("pepd_residual_gain", "mask_residual_gain")},
        arm=arm,
        label=label,
    )
    expected = _gain_telemetry_from_state(state)
    actual = {name: float(raw) for name, raw in value.items()}
    _require(
        all(math.isfinite(raw) for raw in actual.values())
        and actual["pepd_residual_gain"] == expected["pepd_residual_gain"]
        and actual["mask_residual_gain"] == expected["mask_residual_gain"]
        and math.isclose(
            actual["pepd_residual_scale"], expected["pepd_residual_scale"],
            rel_tol=0.0, abs_tol=1e-7,
        )
        and math.isclose(
            actual["mask_residual_scale"], expected["mask_residual_scale"],
            rel_tol=0.0, abs_tol=1e-7,
        ),
        f"{label}: gain telemetry value drift",
    )
    return actual


def _validate_gain_history(
    history: Any,
    *,
    arm: str,
    completed_epoch: int,
    terminal_epoch: int,
    batch_size: int,
    parent_state_sha256: str,
    trainable: Sequence[str],
) -> int:
    _require(
        isinstance(history, list) and len(history) == completed_epoch,
        f"{arm}: gain history length drift",
    )
    expected_keys = {
        "epoch", "terminal_epoch", "samples", "optimizer_steps", "global_step",
        "loss", "augmented_fraction", "gradient_absolute_sum", "gains",
        "parent_state_sha256",
    }
    expected_steps = math.ceil(EXPECTED_INNER_TRAIN[0] / int(batch_size))
    cumulative = 0
    for epoch, row in enumerate(history, start=1):
        _require(
            isinstance(row, Mapping) and set(row) == expected_keys,
            f"{arm}: gain history schema drift: epoch{epoch}",
        )
        cumulative += expected_steps
        gradients = row["gradient_absolute_sum"]
        _require(
            isinstance(gradients, Mapping)
            and tuple(gradients) == tuple(trainable)
            and all(math.isfinite(float(raw)) and float(raw) >= 0.0
                    for raw in gradients.values()),
            f"{arm}: gain gradient receipt drift: epoch{epoch}",
        )
        _validate_gain_telemetry(
            row["gains"], arm=arm, label=f"{arm} history epoch{epoch}"
        )
        _require(
            row.get("epoch") == epoch
            and row.get("terminal_epoch") == terminal_epoch
            and row.get("samples") == EXPECTED_INNER_TRAIN[0]
            and row.get("optimizer_steps") == expected_steps
            and row.get("global_step") == cumulative
            and math.isfinite(float(row.get("loss")))
            and 0.0 <= float(row.get("augmented_fraction")) <= 1.0
            and row.get("parent_state_sha256") == parent_state_sha256,
            f"{arm}: gain history semantic drift: epoch{epoch}",
        )
    return cumulative


def _gain_parent_artifacts(
    protocol: Mapping[str, Any], root: Path
) -> tuple[Path, Path]:
    parent_stage = f"solver_parent_seed_{int(protocol['training']['parent']['seed'])}"
    parent_directory, parent_journal, _ = _stage_paths(root, parent_stage)
    return parent_journal, parent_directory / "summary.json"


def _validate_gain_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    root: Path,
    arm: str,
    parent_terminal: Path,
    parent_summary: Mapping[str, Any],
    require_terminal: bool,
) -> None:
    _require(set(payload) == GAIN_CHECKPOINT_KEYS, f"{arm}: gain checkpoint schema drift")
    config = protocol["training"]["gains"]
    fixed = arm == SOLVER_CORE
    role = "fixed_solver_core_control" if fixed else "independently_trained_gain_arm"
    terminal_epoch = 0 if fixed else int(config["epochs"])
    completed_epoch = int(payload.get("completed_epoch", -1))
    if fixed:
        _require(completed_epoch == 0, "solver core completed epoch drift")
    else:
        _require(1 <= completed_epoch <= terminal_epoch,
                 f"{arm}: gain completed epoch drift")
        if require_terminal:
            _require(completed_epoch == terminal_epoch,
                     f"{arm}: gain terminal epoch incomplete")
    preflight_path = root / "preflight.json"
    preflight = strict_json(preflight_path)
    parent_journal, parent_summary_path = _gain_parent_artifacts(protocol, root)
    expected_trainable = _expected_gain_trainable(arm)
    _require(
        payload.get("schema_version") == 2
        and payload.get("protocol") == PROTOCOL
        and payload.get("role") == role
        and payload.get("arm") == arm
        and int(payload.get("seed", -1)) == int(config["seed"])
        and int(payload.get("terminal_epoch", -1)) == terminal_epoch
        and payload.get("checkpoint_selection")
        == ("none; fixed control" if fixed else "none; fixed terminal")
        and payload.get("gradient_policy")
        == ("not applicable; fixed control" if fixed else GAIN_GRADIENT_POLICY)
        and payload.get("terminal_state_all_finite") is True
        and payload.get("optimizer_state_all_finite") is (None if fixed else True)
        and payload.get("protocol_snapshot_sha256") == canonical_sha256(protocol)
        and payload.get("source_manifest_sha256")
        == protocol["source_manifest"]["manifest_sha256"]
        and payload.get("preflight_sha256") == sha256_file(preflight_path)
        and payload.get("inner_train_content_receipt_sha256")
        == preflight["inner_train_content_verification"]["receipt_sha256"]
        and Path(str(payload.get("parent_terminal"))).resolve(strict=True)
        == parent_terminal.resolve(strict=True)
        and payload.get("parent_terminal_sha256") == sha256_file(parent_terminal)
        and payload.get("parent_journal_sha256") == sha256_file(parent_journal)
        and payload.get("parent_summary_sha256") == sha256_file(parent_summary_path)
        and payload.get("parent_state_sha256") == parent_summary["model_state_sha256"]
        and payload.get("inner_train_ids_sha256")
        == discovery.identity["inner_train_ids_sha256"]
        and payload.get("fresh_zero_initialization") is True
        and payload.get("warm_started_from_full") is False
        and tuple(payload.get("trainable_parameters") or ()) == expected_trainable
        and payload.get("inner_screen_images_read") == 0
        and payload.get("outer_holdout_images_read") == 0
        and math.isfinite(float(payload.get("elapsed_training_seconds")))
        and float(payload.get("elapsed_training_seconds")) >= 0.0,
        f"{arm}: gain checkpoint identity/provenance drift",
    )
    gain_state = _validate_gain_state(payload["gain_state"], arm=arm, label=arm)
    if fixed:
        _require(
            payload.get("global_step") == 0
            and payload.get("optimizer_state") is None
            and payload.get("history") == []
            and payload.get("history_sha256") == canonical_sha256([])
            and float(payload.get("elapsed_training_seconds")) == 0.0,
            "solver core fixed-control state drift",
        )
        return
    global_step = _validate_gain_history(
        payload["history"],
        arm=arm,
        completed_epoch=completed_epoch,
        terminal_epoch=terminal_epoch,
        batch_size=int(config["batch_size"]),
        parent_state_sha256=parent_summary["model_state_sha256"],
        trainable=expected_trainable,
    )
    _require(
        payload.get("global_step") == global_step
        and payload.get("history_sha256") == canonical_sha256(payload["history"]),
        f"{arm}: gain history/global-step hash drift",
    )
    last = payload["history"][-1]["gains"]
    _require(
        gain_state == {
            "pepd_residual_gain": float(last["pepd_residual_gain"]),
            "mask_residual_gain": float(last["mask_residual_gain"]),
        },
        f"{arm}: terminal gain/history drift",
    )
    gain_model_state = {
        name: torch.tensor(float(gain_state[name]), dtype=torch.float32)
        for name in ("pepd_residual_gain", "mask_residual_gain")
    }
    _validate_adamw_optimizer_state(
        payload["optimizer_state"],
        model_state=gain_model_state,
        trainable=expected_trainable,
        global_step=global_step,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        param_group_contract=config["optimizer_param_group"],
        label=f"{arm} optimizer state",
    )


def _gain_summary_value(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    root: Path,
    arm: str,
    journal_path: Path,
    terminal_path: Path,
    device: torch.device,
    recovered: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "status": "complete",
        "role": payload["role"],
        "arm": arm,
        "seed": payload["seed"],
        "terminal_epoch": payload["terminal_epoch"],
        "optimizer_steps": payload["global_step"],
        "checkpoint_selection": (
            "none; fixed control" if arm == SOLVER_CORE else "none; terminal only"
        ),
        "gradient_policy": payload["gradient_policy"],
        "terminal_state_all_finite": True,
        "optimizer_state_all_finite": payload["optimizer_state_all_finite"],
        "fresh_zero_initialization": True,
        "warm_started_from_full": False,
        "trainable_parameters": list(payload["trainable_parameters"]),
        "history": list(payload["history"]),
        "history_sha256": payload["history_sha256"],
        "gains": _gain_telemetry_from_state(payload["gain_state"]),
        "preflight_sha256": payload["preflight_sha256"],
        "inner_train_content_receipt_sha256": payload[
            "inner_train_content_receipt_sha256"
        ],
        "parent_terminal": payload["parent_terminal"],
        "parent_terminal_sha256": payload["parent_terminal_sha256"],
        "parent_journal_sha256": payload["parent_journal_sha256"],
        "parent_summary_sha256": payload["parent_summary_sha256"],
        "parent_state_sha256": payload["parent_state_sha256"],
        "journal": str(journal_path),
        "journal_sha256": sha256_file(journal_path),
        "terminal": str(terminal_path),
        "terminal_sha256": sha256_file(terminal_path),
        "inner_train_ids_sha256": discovery.identity["inner_train_ids_sha256"],
        "inner_screen_images_read": 0,
        "outer_holdout_images_read": 0,
        "elapsed_training_seconds": payload["elapsed_training_seconds"],
        "device_name": torch.cuda.get_device_name(device),
        "protocol_snapshot_sha256": canonical_sha256(protocol),
        "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
        "publication_state_machine": (
            "EMPTY->JOURNALED->TERMINAL_PUBLISHED->SUMMARY_COMPLETE"
        ),
        "recovered_terminal_without_summary": bool(recovered),
    }


def train_gain_arm(
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    *,
    arm: str,
    root: Path,
    workers: int,
    device: torch.device,
    resume: bool,
) -> Path:
    _require(arm in ARM_NAMES, f"unknown gain arm: {arm}")
    preflight = _require_preflight(
        protocol, root, discovery=discovery, workers=max(1, int(workers))
    )
    parent_terminal, parent_payload, parent_summary = _validated_parent_terminal(
        protocol, discovery, root
    )
    config = protocol["training"]["gains"]
    stage = f"gain_{arm}_seed_{int(config['seed'])}"
    directory, journal_path, terminal_path = _stage_paths(root, stage)
    summary_path = directory / "summary.json"
    stale_lock_evidence = (root / "locks" / f"{stage}.lock").is_file()
    lock = _acquire_lock(root, stage, resume=resume)
    completed = False
    try:
        directory.mkdir(parents=True, exist_ok=True)
        publication_state = _gain_stage_state(directory)
        configure_determinism(int(config["seed"]))
        model = CAGHV5SolverGatedResidual(progress_bins=72, dropout=0.10)
        model.load_parent_state_dict(parent_payload["model_state"])
        with torch.no_grad():
            model.pepd_residual_gain.zero_()
            model.mask_residual_gain.zero_()
        trainable = model.configure_trainable_arm(arm)
        expected_trainable = _expected_gain_trainable(arm)
        _require(tuple(trainable) == expected_trainable, f"{arm}: trainable roster drift")
        model = model.to(device)
        parent_digest = _module_state_sha256(model.parent)
        _require(parent_digest == parent_summary["model_state_sha256"],
                 f"{arm}: loaded parent digest drift")
        parent_journal, parent_summary_path = _gain_parent_artifacts(protocol, root)

        if publication_state == "SUMMARY_COMPLETE":
            _require(resume, f"{arm}: complete gain stage requires explicit --resume")
            validated = _validated_gain_terminal(protocol, discovery, root, arm=arm)
            completed = True
            return Path(validated["terminal"])
        if not resume:
            _require(publication_state == "EMPTY",
                     f"{arm}: fresh gain run refuses existing artifacts")
        else:
            _require(
                publication_state in {"JOURNALED", "TERMINAL_PUBLISHED"}
                or (publication_state == "EMPTY" and stale_lock_evidence),
                     f"{arm}: --resume requires a valid incomplete gain state")

        if arm == SOLVER_CORE:
            if journal_path.is_file():
                payload = torch.load(journal_path, map_location="cpu", weights_only=False)
                _validate_gain_checkpoint_payload(
                    payload,
                    protocol=protocol,
                    discovery=discovery,
                    root=root,
                    arm=arm,
                    parent_terminal=parent_terminal,
                    parent_summary=parent_summary,
                    require_terminal=terminal_path.is_file(),
                )
            else:
                payload = {
                    "schema_version": 2,
                    "protocol": PROTOCOL,
                    "role": "fixed_solver_core_control",
                    "arm": arm,
                    "seed": int(config["seed"]),
                    "terminal_epoch": 0,
                    "completed_epoch": 0,
                    "global_step": 0,
                    "checkpoint_selection": "none; fixed control",
                    "gradient_policy": "not applicable; fixed control",
                    "terminal_state_all_finite": True,
                    "optimizer_state_all_finite": None,
                    "protocol_snapshot_sha256": canonical_sha256(protocol),
                    "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
                    "preflight_sha256": sha256_file(root / "preflight.json"),
                    "inner_train_content_receipt_sha256": preflight[
                        "inner_train_content_verification"
                    ]["receipt_sha256"],
                    "parent_terminal": str(parent_terminal),
                    "parent_terminal_sha256": sha256_file(parent_terminal),
                    "parent_journal_sha256": sha256_file(parent_journal),
                    "parent_summary_sha256": sha256_file(parent_summary_path),
                    "parent_state_sha256": parent_digest,
                    "inner_train_ids_sha256": discovery.identity["inner_train_ids_sha256"],
                    "fresh_zero_initialization": True,
                    "warm_started_from_full": False,
                    "trainable_parameters": [],
                    "gain_state": {
                        "pepd_residual_gain": 0.0,
                        "mask_residual_gain": 0.0,
                    },
                    "optimizer_state": None,
                    "history": [],
                    "history_sha256": canonical_sha256([]),
                    "elapsed_training_seconds": 0.0,
                    "inner_screen_images_read": 0,
                    "outer_holdout_images_read": 0,
                }
                _validate_gain_checkpoint_payload(
                    payload,
                    protocol=protocol,
                    discovery=discovery,
                    root=root,
                    arm=arm,
                    parent_terminal=parent_terminal,
                    parent_summary=parent_summary,
                    require_terminal=True,
                )
                torch_new(journal_path, payload)
            if terminal_path.is_file():
                terminal_payload = torch.load(
                    terminal_path, map_location="cpu", weights_only=False
                )
                _require(_nested_equal(payload, terminal_payload),
                         "solver core journal/terminal semantic drift")
                _validate_gain_checkpoint_payload(
                    terminal_payload,
                    protocol=protocol,
                    discovery=discovery,
                    root=root,
                    arm=arm,
                    parent_terminal=parent_terminal,
                    parent_summary=parent_summary,
                    require_terminal=True,
                )
            else:
                torch_new(terminal_path, payload)
                terminal_payload = torch.load(
                    terminal_path, map_location="cpu", weights_only=False
                )
                _require(_nested_equal(payload, terminal_payload),
                         "published solver core terminal semantic drift")
            atomic_json_new(
                summary_path,
                _gain_summary_value(
                    terminal_payload,
                    protocol=protocol,
                    discovery=discovery,
                    root=root,
                    arm=arm,
                    journal_path=journal_path,
                    terminal_path=terminal_path,
                    device=device,
                    recovered=publication_state == "TERMINAL_PUBLISHED",
                ),
            )
            completed = True
            return terminal_path

        dataset = CanonicalTightROIDataset(
            discovery.inner_train,
            training=True,
            seed=int(config["seed"]),
            augmentation=_augmentation(protocol, enabled=True),
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        history: list[dict[str, Any]] = []
        start_epoch = 1
        global_step = 0
        elapsed_before = 0.0
        if journal_path.is_file():
            _require(resume, f"{arm}: orphan journal requires --resume")
            checkpoint = torch.load(journal_path, map_location="cpu", weights_only=False)
            _validate_gain_checkpoint_payload(
                checkpoint,
                protocol=protocol,
                discovery=discovery,
                root=root,
                arm=arm,
                parent_terminal=parent_terminal,
                parent_summary=parent_summary,
                require_terminal=terminal_path.is_file(),
            )
            _load_gain_state(model, checkpoint["gain_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _validate_loaded_adamw_optimizer(
                optimizer,
                module=model,
                trainable=expected_trainable,
                global_step=int(checkpoint["global_step"]),
                label=f"{arm} resume optimizer",
            )
            history = list(checkpoint["history"])
            start_epoch = int(checkpoint["completed_epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            elapsed_before = float(checkpoint["elapsed_training_seconds"])
            _require(_module_state_sha256(model.parent) == parent_digest,
                     f"{arm}: resume mutated frozen parent")
            if terminal_path.is_file():
                terminal_payload = torch.load(
                    terminal_path, map_location="cpu", weights_only=False
                )
                _require(_nested_equal(checkpoint, terminal_payload),
                         f"{arm}: journal/terminal semantic drift")
                _validate_gain_checkpoint_payload(
                    terminal_payload,
                    protocol=protocol,
                    discovery=discovery,
                    root=root,
                    arm=arm,
                    parent_terminal=parent_terminal,
                    parent_summary=parent_summary,
                    require_terminal=True,
                )
                atomic_json_new(
                    summary_path,
                    _gain_summary_value(
                        terminal_payload,
                        protocol=protocol,
                        discovery=discovery,
                        root=root,
                        arm=arm,
                        journal_path=journal_path,
                        terminal_path=terminal_path,
                        device=device,
                        recovered=True,
                    ),
                )
                completed = True
                return terminal_path
        started = time.time()
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            configure_determinism(int(config["seed"]) + epoch * 1_000_003)
            dataset.set_epoch(epoch)
            loader = _loader(
                dataset,
                batch_size=int(config["batch_size"]),
                workers=workers,
                shuffle=True,
                seed=int(config["seed"]) + epoch,
                device=device,
            )
            model.train()
            total = 0.0
            samples = 0
            augmented = 0
            steps = 0
            gradient_sum = Counter()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    batch["image"].to(device, non_blocking=True),
                    batch["final_to_isotropic"].to(device, non_blocking=True),
                    batch["crop_affine"].to(device, non_blocking=True),
                    arm=arm,
                )
                loss, _ = solver_gated_progress_loss(
                    output,
                    target_progress=batch["target_progress"].to(device, non_blocking=True),
                    group_weight=batch["group_weight"].to(device, non_blocking=True),
                )
                _require(bool(torch.isfinite(loss)), f"{arm}: non-finite gain loss")
                loss.backward()
                _require_trainable_gradients_finite(
                    model,
                    expected_trainable,
                    label=f"{arm} epoch{epoch} step{steps + 1}",
                )
                named_parameters = dict(model.named_parameters())
                for name in expected_trainable:
                    gradient_sum[name] += float(
                        named_parameters[name].grad.detach().abs().sum()
                    )
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    float(protocol["training"]["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                _require_named_parameters_finite(
                    model,
                    expected_trainable,
                    label=f"{arm} epoch{epoch} step{steps + 1}",
                )
                _nested_tensors_finite(
                    optimizer.state, label=f"{arm} epoch{epoch} optimizer state"
                )
                gains = _gain_values(model)
                _require(all(math.isfinite(value) for value in gains.values()),
                         f"{arm}: non-finite terminal gain candidate")
                count = int(output.expected_progress.shape[0])
                total += float(loss.detach()) * count
                samples += count
                steps += 1
                augmented += int((batch["augmentation_code"] != 0).sum())
            _require(samples == len(dataset), f"{arm}/epoch{epoch}: sample inventory drift")
            _require(_module_state_sha256(model.parent) == parent_digest,
                     f"{arm}/epoch{epoch}: frozen parent state drift")
            global_step += steps
            row = {
                "epoch": epoch,
                "terminal_epoch": int(config["epochs"]),
                "samples": samples,
                "optimizer_steps": steps,
                "global_step": global_step,
                "loss": total / samples,
                "augmented_fraction": augmented / samples,
                "gradient_absolute_sum": dict(gradient_sum),
                "gains": _gain_values(model),
                "parent_state_sha256": parent_digest,
            }
            history.append(row)
            gain_state = {
                "pepd_residual_gain": float(model.pepd_residual_gain.detach().cpu()),
                "mask_residual_gain": float(model.mask_residual_gain.detach().cpu()),
            }
            optimizer_state = _cpu_nested_state(optimizer.state_dict())
            checkpoint = {
                "schema_version": 2,
                "protocol": PROTOCOL,
                "role": "independently_trained_gain_arm",
                "arm": arm,
                "seed": int(config["seed"]),
                "terminal_epoch": int(config["epochs"]),
                "completed_epoch": epoch,
                "global_step": global_step,
                "checkpoint_selection": "none; fixed terminal",
                "gradient_policy": GAIN_GRADIENT_POLICY,
                "terminal_state_all_finite": True,
                "optimizer_state_all_finite": True,
                "protocol_snapshot_sha256": canonical_sha256(protocol),
                "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
                "preflight_sha256": sha256_file(root / "preflight.json"),
                "inner_train_content_receipt_sha256": preflight[
                    "inner_train_content_verification"
                ]["receipt_sha256"],
                "parent_terminal": str(parent_terminal),
                "parent_terminal_sha256": sha256_file(parent_terminal),
                "parent_journal_sha256": sha256_file(parent_journal),
                "parent_summary_sha256": sha256_file(parent_summary_path),
                "parent_state_sha256": parent_digest,
                "inner_train_ids_sha256": discovery.identity["inner_train_ids_sha256"],
                "fresh_zero_initialization": True,
                "warm_started_from_full": False,
                "trainable_parameters": list(trainable),
                "gain_state": gain_state,
                "optimizer_state": optimizer_state,
                "history": history,
                "history_sha256": canonical_sha256(history),
                "elapsed_training_seconds": elapsed_before + time.time() - started,
                "inner_screen_images_read": 0,
                "outer_holdout_images_read": 0,
            }
            _validate_gain_checkpoint_payload(
                checkpoint,
                protocol=protocol,
                discovery=discovery,
                root=root,
                arm=arm,
                parent_terminal=parent_terminal,
                parent_summary=parent_summary,
                require_terminal=epoch == int(config["epochs"]),
            )
            atomic_torch(journal_path, checkpoint)
            print(
                f"gain arm={arm} epoch={epoch}/{config['epochs']} "
                f"loss={row['loss']:.6f} gains={row['gains']}",
                flush=True,
            )
        _require(len(history) == int(config["epochs"]), f"{arm}: history incomplete")
        payload = torch.load(journal_path, map_location="cpu", weights_only=False)
        _validate_gain_checkpoint_payload(
            payload,
            protocol=protocol,
            discovery=discovery,
            root=root,
            arm=arm,
            parent_terminal=parent_terminal,
            parent_summary=parent_summary,
            require_terminal=True,
        )
        torch_new(terminal_path, payload)
        terminal_payload = torch.load(terminal_path, map_location="cpu", weights_only=False)
        _require(_nested_equal(payload, terminal_payload),
                 f"{arm}: published gain terminal semantic drift")
        atomic_json_new(
            summary_path,
            _gain_summary_value(
                terminal_payload,
                protocol=protocol,
                discovery=discovery,
                root=root,
                arm=arm,
                journal_path=journal_path,
                terminal_path=terminal_path,
                device=device,
                recovered=False,
            ),
        )
        completed = True
        return terminal_path
    finally:
        if completed:
            _complete_lock(lock)


def train_all_gain_arms(
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    *,
    root: Path,
    workers: int,
    device: torch.device,
    resume: bool,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    config = protocol["training"]["gains"]
    for arm in ARM_NAMES:
        stage = f"gain_{arm}_seed_{int(config['seed'])}"
        directory, _, _ = _stage_paths(root, stage)
        paths.append(train_gain_arm(
            protocol,
            discovery,
            arm=arm,
            root=root,
            workers=workers,
            device=device,
            resume=bool(resume and directory.exists()),
        ))
    return tuple(paths)


def _validated_gain_terminal(
    protocol: Mapping[str, Any],
    discovery: ScreenDiscovery,
    root: Path,
    *,
    arm: str,
) -> dict[str, Any]:
    _require(arm in ARM_NAMES, f"unknown gain arm: {arm}")
    config = protocol["training"]["gains"]
    parent_terminal, _, parent_summary = _validated_parent_terminal(
        protocol, discovery, root
    )
    stage = f"gain_{arm}_seed_{int(config['seed'])}"
    directory, journal, terminal = _stage_paths(root, stage)
    summary_path = directory / "summary.json"
    _require(
        _gain_stage_state(directory) == "SUMMARY_COMPLETE",
        f"{arm}: gain stage is not complete",
    )
    for path in (journal, terminal, summary_path):
        _require(path.is_file() and not path.is_symlink() and not _is_reparse(path),
                 f"{arm}: invalid completed gain artifact: {path.name}")
    journal_payload = torch.load(journal, map_location="cpu", weights_only=False)
    payload = torch.load(terminal, map_location="cpu", weights_only=False)
    _require(_nested_equal(journal_payload, payload),
             f"{arm}: gain journal/terminal semantic drift")
    _validate_gain_checkpoint_payload(
        payload,
        protocol=protocol,
        discovery=discovery,
        root=root,
        arm=arm,
        parent_terminal=parent_terminal,
        parent_summary=parent_summary,
        require_terminal=True,
    )
    summary = strict_json(summary_path)
    _require(set(summary) == GAIN_SUMMARY_KEYS, f"{arm}: gain summary schema drift")
    fixed = arm == SOLVER_CORE
    expected_role = "fixed_solver_core_control" if fixed else "independently_trained_gain_arm"
    expected_trainable = _expected_gain_trainable(arm)
    expected_gains = _gain_telemetry_from_state(payload["gain_state"])
    _validate_gain_telemetry(summary["gains"], arm=arm, label=f"{arm} summary")
    parent_journal, parent_summary_path = _gain_parent_artifacts(protocol, root)
    _require(
        summary.get("schema_version") == 2
        and summary.get("protocol") == PROTOCOL
        and summary.get("status") == "complete"
        and summary.get("role") == expected_role
        and summary.get("arm") == arm
        and int(summary.get("seed", -1)) == int(config["seed"])
        and int(summary.get("terminal_epoch", -1)) == (0 if fixed else int(config["epochs"]))
        and summary.get("optimizer_steps") == payload["global_step"]
        and summary.get("checkpoint_selection")
        == ("none; fixed control" if fixed else "none; terminal only")
        and summary.get("gradient_policy") == payload["gradient_policy"]
        and summary.get("terminal_state_all_finite") is True
        and summary.get("optimizer_state_all_finite")
        is payload["optimizer_state_all_finite"]
        and summary.get("fresh_zero_initialization") is True
        and summary.get("warm_started_from_full") is False
        and tuple(summary.get("trainable_parameters") or ()) == expected_trainable
        and summary.get("history") == payload["history"]
        and summary.get("history_sha256") == payload["history_sha256"]
        and summary.get("gains") == expected_gains
        and summary.get("preflight_sha256") == payload["preflight_sha256"]
        and summary.get("inner_train_content_receipt_sha256")
        == payload["inner_train_content_receipt_sha256"]
        and Path(str(summary.get("parent_terminal"))).resolve(strict=True)
        == parent_terminal.resolve(strict=True)
        and summary.get("parent_terminal_sha256") == sha256_file(parent_terminal)
        and summary.get("parent_journal_sha256") == sha256_file(parent_journal)
        and summary.get("parent_summary_sha256") == sha256_file(parent_summary_path)
        and summary.get("parent_state_sha256") == parent_summary["model_state_sha256"]
        and Path(str(summary.get("journal"))).resolve(strict=True) == journal.resolve(strict=True)
        and summary.get("journal_sha256") == sha256_file(journal)
        and Path(str(summary.get("terminal"))).resolve(strict=True)
        == terminal.resolve(strict=True)
        and summary.get("terminal_sha256") == sha256_file(terminal)
        and summary.get("inner_train_ids_sha256")
        == discovery.identity["inner_train_ids_sha256"]
        and summary.get("inner_screen_images_read") == 0
        and summary.get("outer_holdout_images_read") == 0
        and math.isfinite(float(summary.get("elapsed_training_seconds")))
        and float(summary.get("elapsed_training_seconds"))
        == float(payload["elapsed_training_seconds"])
        and summary.get("device_name") == "NVIDIA GeForce RTX 4060"
        and summary.get("protocol_snapshot_sha256") == canonical_sha256(protocol)
        and summary.get("source_manifest_sha256")
        == protocol["source_manifest"]["manifest_sha256"]
        and summary.get("publication_state_machine")
        == "EMPTY->JOURNALED->TERMINAL_PUBLISHED->SUMMARY_COMPLETE"
        and isinstance(summary.get("recovered_terminal_without_summary"), bool),
        f"{arm}: gain summary identity/alignment drift",
    )
    return {
        "terminal": terminal,
        "terminal_sha256": sha256_file(terminal),
        "summary": summary_path,
        "summary_sha256": sha256_file(summary_path),
        "journal": journal,
        "journal_sha256": sha256_file(journal),
        "gain_state": dict(payload["gain_state"]),
    }


def _validated_gain_terminals(
    protocol: Mapping[str, Any], discovery: ScreenDiscovery, root: Path
) -> dict[str, dict[str, Any]]:
    return {
        arm: _validated_gain_terminal(protocol, discovery, root, arm=arm)
        for arm in ARM_NAMES
    }


def _array_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _assert_label_free(rows: Sequence[Mapping[str, Any]]) -> None:
    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = str(key).casefold()
                _require(
                    not any(token in normalized for token in LABEL_TOKENS),
                    f"label-like prediction key at {location}.{key}",
                )
                walk(nested, f"{location}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, nested in enumerate(value):
                walk(nested, f"{location}[{index}]")
        elif isinstance(value, float):
            _require(math.isfinite(value), f"non-finite prediction value at {location}")

    for index, row in enumerate(rows):
        walk(row, f"prediction[{index}]")


def _screen_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "metric rows empty")
    errors = np.asarray([float(row["absolute_progress_error"]) for row in rows], dtype=np.float64)
    covered = np.asarray([row["status"] == "ok" for row in rows], dtype=bool)
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    _require(all(sample_ids) and len(set(sample_ids)) == len(sample_ids),
             "metric sample identity drift")
    _require(
        all(row.get("status") in {"ok", "failure"} for row in rows),
        "metric status drift",
    )
    _require(
        bool(np.isfinite(errors).all())
        and bool(((errors >= 0.0) & (errors <= 1.0)).all()),
        "metric errors non-finite/outside [0,1]",
    )
    _require(
        all(float(row["absolute_progress_error"]) == 1.0
            for row in rows if row["status"] == "failure"),
        "metric failure penalty drift",
    )
    groups: dict[str, list[float]] = {}
    for row, error in zip(rows, errors, strict=True):
        group_id = str(row.get("group_id") or "")
        _require(bool(group_id), "metric group identity empty")
        groups.setdefault(group_id, []).append(float(error))
    return {
        "samples": len(rows),
        "groups": len(groups),
        "coverage": float(covered.mean()),
        "failures": int((~covered).sum()),
        "failure_rate": float((~covered).mean()),
        "full_denominator_nmae": float(errors.mean()),
        "covered_nmae": float(errors[covered].mean()) if bool(covered.any()) else None,
        "p95_absolute_progress_error": float(np.quantile(errors, 0.95, method="linear")),
        "p99_absolute_progress_error": float(np.quantile(errors, 0.99, method="linear")),
        "group_macro_full_denominator_nmae": float(
            np.mean([np.mean(values) for values in groups.values()])
        ),
        "failure_penalty": 1.0,
    }


@torch.inference_mode()
def evaluate_screen(
    protocol: Mapping[str, Any],
    *,
    root: Path,
    workers: int,
    device: torch.device,
) -> Path:
    _require_preflight(protocol, root)
    identity_only = ScreenDiscovery(
        inner_train=(), inner_screen=(), identity=dict(protocol["inputs"]["inner_identity"])
    )
    terminals = _validated_gain_terminals(protocol, identity_only, root)
    parent_terminal, parent_payload, parent_summary = _validated_parent_terminal(
        protocol, identity_only, root
    )
    final_output = root / "screen"
    output = root / "screen.inprogress"
    attempt_marker = root / "screen_attempt_started.json"
    _require(not final_output.exists(), f"screen output already exists: {final_output}")
    _require(not output.exists(), "incomplete screen attempt permanently blocks rerun")
    _require(not attempt_marker.exists(), "screen attempt was already started; rerun forbidden")
    lock = _acquire_lock(root, "screen_evaluation_once", resume=False)
    completed = False
    try:
        atomic_json_new(
            attempt_marker,
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "screen_attempt_started_irreversible",
                "protocol_snapshot_sha256": canonical_sha256(protocol),
                "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
                "gain_terminal_sha256": {
                    arm: terminals[arm]["terminal_sha256"] for arm in ARM_NAMES
                },
                "started_at_unix": time.time(),
                "retry_authorized": False,
            },
        )
        output.mkdir(parents=True, exist_ok=False)
        summary_path = output / "summary.json"
        discovery = load_fit_discovery(protocol, scope="inner_screen")
        screen_content_verification = _verify_partition_content(
            protocol,
            discovery.inner_screen,
            scope="inner_screen",
            workers=max(1, int(workers)),
        )
        configure_determinism(EVALUATION_SEED)
        model = CAGHV5SolverGatedResidual(progress_bins=72, dropout=0.10)
        model.load_parent_state_dict(parent_payload["model_state"])
        model = model.to(device).eval()
        _require(_module_state_sha256(model.parent) == parent_summary["model_state_sha256"],
                 "screen parent state drift")
        predictions: list[dict[str, Any]] = []
        scores: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        metric_rows: dict[tuple[str, str], list[dict[str, Any]]] = {
            (arm, condition): []
            for arm in ARM_NAMES
            for condition in protocol["screen_evaluation"]["conditions"]
        }
        started = time.time()
        for condition_value in protocol["screen_evaluation"]["conditions"]:
            condition = str(condition_value)
            stress = condition == "frozen_stress"
            dataset = CanonicalTightROIDataset(
                discovery.inner_screen,
                training=stress,
                seed=EVALUATION_SEED,
                augmentation=_augmentation(protocol, enabled=stress),
            )
            dataset.set_epoch(STRESS_EPOCH if stress else 0)
            loader = _loader(
                dataset,
                batch_size=int(protocol["training"]["parent"]["batch_size"]),
                workers=workers,
                shuffle=False,
                seed=EVALUATION_SEED,
                device=device,
            )
            seen = 0
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                final_to_isotropic = batch["final_to_isotropic"].to(device, non_blocking=True)
                crop_affine = batch["crop_affine"].to(device, non_blocking=True)
                sample_ids = list(map(str, batch["sample_id"]))
                group_ids = list(map(str, batch["group_id"]))
                target = batch["target_progress"].detach().cpu().numpy().astype(np.float64)
                batch_receipt_hashes: list[str] = []
                for index, sample_id in enumerate(sample_ids):
                    receipt_row = {
                            "schema_version": 1,
                            "protocol": PROTOCOL,
                            "sample_id": sample_id,
                            "condition": condition,
                            "image_tensor_sha256": _array_hash(batch["image"][index]),
                            "final_to_isotropic_sha256": _array_hash(batch["final_to_isotropic"][index]),
                            "crop_affine_sha256": _array_hash(batch["crop_affine"][index]),
                            "augmentation_code": int(batch["augmentation_code"][index]),
                    }
                    receipts.append(receipt_row)
                    batch_receipt_hashes.append(canonical_sha256(receipt_row))
                for arm in ARM_NAMES:
                    _load_gain_state(model, terminals[arm]["gain_state"])
                    output_value = model(
                        images,
                        final_to_isotropic,
                        crop_affine,
                        arm=arm,
                    )
                    predicted = output_value.expected_progress.detach().cpu().numpy().astype(np.float64)
                    valid = output_value.valid.detach().cpu().numpy().astype(bool)
                    pointer_valid = output_value.pointer_valid.detach().cpu().numpy().astype(bool)
                    reference_valid = output_value.reference_valid.detach().cpu().numpy().astype(bool)
                    geometry_valid = output_value.geometry_solver_valid.detach().cpu().numpy().astype(bool)
                    fusion_valid = output_value.fusion_solver_valid.detach().cpu().numpy().astype(bool)
                    pepd_gate = output_value.pepd_reliability_gate.detach().cpu().numpy().astype(np.float64)
                    mask_gate = output_value.mask_reliability_gate.detach().cpu().numpy().astype(np.float64)
                    for index, sample_id in enumerate(sample_ids):
                        success = bool(
                            valid[index]
                            and math.isfinite(float(predicted[index]))
                            and 0.0 <= float(predicted[index]) <= 1.0
                        )
                        prediction = float(predicted[index]) if success else None
                        prediction_row = {
                            "schema_version": 1,
                            "protocol": PROTOCOL,
                            "sample_id": sample_id,
                            "arm": arm,
                            "condition": condition,
                            "checkpoint_sha256": terminals[arm]["terminal_sha256"],
                            "input_receipt_sha256": batch_receipt_hashes[index],
                            "status": "ok" if success else "failure",
                            "failure_code": None if success else "invalid_geometry_or_fusion",
                            "predicted_progress": prediction,
                            "validity": {
                                "combined": bool(valid[index]),
                                "pointer": bool(pointer_valid[index]),
                                "reference": bool(reference_valid[index]),
                                "geometry_solver": bool(geometry_valid[index]),
                                "fusion_solver": bool(fusion_valid[index]),
                            },
                            "pepd_reliability": float(pepd_gate[index]),
                            "mask_reliability": float(mask_gate[index]),
                            "runtime_inputs": [
                                "canonical_roi_pixels",
                                "preprocessing_final_to_isotropic",
                                "preprocessing_crop_affine",
                            ],
                        }
                        error = abs(float(prediction) - float(target[index])) if success else 1.0
                        score_row = {
                            "schema_version": 1,
                            "protocol": PROTOCOL,
                            "sample_id": sample_id,
                            "group_id": group_ids[index],
                            "arm": arm,
                            "condition": condition,
                            "checkpoint_sha256": terminals[arm]["terminal_sha256"],
                            "input_receipt_sha256": batch_receipt_hashes[index],
                            "status": prediction_row["status"],
                            "predicted_progress": prediction,
                            "target_progress": float(target[index]),
                            "absolute_progress_error": float(error),
                            "failure_penalty": 1.0,
                        }
                        predictions.append(prediction_row)
                        scores.append(score_row)
                        metric_rows[(arm, condition)].append(score_row)
                seen += len(sample_ids)
            _require(seen == len(discovery.inner_screen), f"{condition}: screen roster drift")

        _assert_label_free(predictions)
        _assert_label_free(receipts)
        _require(len(predictions) == len(scores) == len(ARM_NAMES) * 2 * len(discovery.inner_screen),
                 "screen bundle row count drift")
        _require(len(receipts) == 2 * len(discovery.inner_screen), "screen receipt count drift")
        _require(
            len({(row["sample_id"], row["arm"], row["condition"]) for row in predictions})
            == len(predictions)
            and len({(row["sample_id"], row["arm"], row["condition"]) for row in scores})
            == len(scores)
            and len({(row["sample_id"], row["condition"]) for row in receipts})
            == len(receipts),
            "screen bundle contains duplicate identities",
        )
        predictions_path = output / "predictions.label_free.jsonl"
        scores_path = output / "scores.labels_joined.jsonl"
        receipts_path = output / "input_receipts.label_free.jsonl"
        atomic_jsonl(predictions_path, predictions)
        atomic_jsonl(scores_path, scores)
        atomic_jsonl(receipts_path, receipts)
        metrics = {
            arm: {
                condition: _screen_metrics(metric_rows[(arm, condition)])
                for condition in protocol["screen_evaluation"]["conditions"]
            }
            for arm in ARM_NAMES
        }
        target_by_sample: dict[str, tuple[str, float]] = {}
        for row in scores:
            sample_id = str(row["sample_id"])
            candidate = (str(row["group_id"]), float(row["target_progress"]))
            previous = target_by_sample.setdefault(sample_id, candidate)
            _require(previous == candidate, f"screen target roster drift: {sample_id}")
        _require(len(target_by_sample) == len(discovery.inner_screen),
                 "screen target roster incomplete")
        summary = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "scope": "public_inner_screen_only",
            "inner_screen_samples": len(discovery.inner_screen),
            "inner_screen_groups": EXPECTED_INNER_SCREEN[1],
            "inner_screen_ids_sha256": discovery.identity["inner_screen_ids_sha256"],
            "inner_screen_groups_sha256": discovery.identity["inner_screen_groups_sha256"],
            "outer_holdout_images_read": 0,
            "restricted_images_read": 0,
            "shared_batch_inputs_across_arms": True,
            "conditions": list(protocol["screen_evaluation"]["conditions"]),
            "metrics": metrics,
            "gain_terminals": {
                arm: {
                    "terminal": str(terminals[arm]["terminal"]),
                    "terminal_sha256": terminals[arm]["terminal_sha256"],
                    "summary": str(terminals[arm]["summary"]),
                    "summary_sha256": terminals[arm]["summary_sha256"],
                    "gain_state": terminals[arm]["gain_state"],
                }
                for arm in ARM_NAMES
            },
            "parent_terminal": str(parent_terminal),
            "parent_terminal_sha256": sha256_file(parent_terminal),
            "parent_state_sha256": parent_summary["model_state_sha256"],
            "input_receipt_roots": {
                condition: canonical_sha256(
                    [row for row in receipts if row["condition"] == condition]
                )
                for condition in protocol["screen_evaluation"]["conditions"]
            },
            "score_target_roster_sha256": canonical_sha256(
                [[sample_id, *target_by_sample[sample_id]] for sample_id in sorted(target_by_sample)]
            ),
            "artifacts": {
                "predictions": str(final_output / predictions_path.name),
                "predictions_sha256": sha256_file(predictions_path),
                "scores": str(final_output / scores_path.name),
                "scores_sha256": sha256_file(scores_path),
                "input_receipts": str(final_output / receipts_path.name),
                "input_receipts_sha256": sha256_file(receipts_path),
            },
            "elapsed_seconds": time.time() - started,
            "device_name": torch.cuda.get_device_name(device),
            "protocol_snapshot_sha256": canonical_sha256(protocol),
            "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
            "screen_content_verification": screen_content_verification,
            "irreversible_attempt_marker": str(attempt_marker),
            "irreversible_attempt_marker_sha256": sha256_file(attempt_marker),
        }
        atomic_json(summary_path, summary)
        os.replace(output, final_output)
        atomic_json_new(
            root / "screen_complete.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "screen_complete",
                "attempt_marker_sha256": sha256_file(attempt_marker),
                "summary": str(final_output / "summary.json"),
                "summary_sha256": sha256_file(final_output / "summary.json"),
            },
        )
        completed = True
        return final_output / "summary.json"
    finally:
        if completed:
            _complete_lock(lock)


def _formal_device(value: str) -> torch.device:
    _require(str(value).casefold() == "cuda", "formal runner accepts only --device cuda")
    device = torch.device("cuda:0")
    _require_cuda_runtime(device)
    return device


def _require_current_disk_space(root: Path) -> int:
    free = int(shutil.disk_usage(root).free)
    _require(free >= 20 * 1024**3, "current C: free space below 20 GiB")
    return free


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen public inner-screen for solver-gated CAGH-V5"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "emit-protocol-candidate",
            "freeze-protocol-candidate",
            "validate-only",
            "import-pepd-p0",
            "preflight",
            "train-parent",
            "train-gain",
            "evaluate-screen",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arm", choices=ARM_NAMES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _guard_output_root(
        args.output_root,
        allow_create=args.mode == "emit-protocol-candidate",
    )
    _require(
        root == DEFAULT_OUTPUT_ROOT.resolve(strict=False),
        "formal output root differs from frozen C: root",
    )
    _require(int(args.workers) in (0, 1, 2, 3, 4), "workers must be in [0,4]")
    if args.mode == "emit-protocol-candidate":
        _require(args.arm is None and not args.resume, "candidate mode has extra run flags")
        value = build_protocol_candidate(root)
        print(json.dumps({
            "status": "protocol_candidate_emitted",
            "path": str(root / "protocol_candidate.json"),
            "canonical_sha256": canonical_sha256(value),
        }, sort_keys=True), flush=True)
        return 0
    if args.mode == "freeze-protocol-candidate":
        _require(args.arm is None and not args.resume, "freeze mode has extra run flags")
        path = freeze_protocol_candidate(root)
        print(json.dumps({
            "status": "protocol_frozen",
            "path": str(path),
            "sha256": sha256_file(path),
        }, sort_keys=True), flush=True)
        return 0

    protocol_scope = (
        "p0_import"
        if args.mode == "import-pepd-p0"
        else ("no_partition" if args.mode == "evaluate-screen" else "inner_train")
    )
    protocol = load_protocol(scope=protocol_scope)
    if args.mode == "validate-only":
        _require(args.arm is None and not args.resume, "validate mode has extra run flags")
        discovery = load_fit_discovery(protocol, scope="inner_train")
        print(json.dumps({
            "status": "validation_pass",
            "protocol_snapshot_sha256": canonical_sha256(protocol),
            "source_manifest_sha256": protocol["source_manifest"]["manifest_sha256"],
            "inner_train_samples": len(discovery.inner_train),
            "inner_train_groups": discovery.identity["inner_train_groups"],
            "inner_screen_manifest_rows_read": 0,
            "outer_holdout_images_read": 0,
            "restricted_images_read": 0,
        }, sort_keys=True), flush=True)
        return 0
    if args.mode == "import-pepd-p0":
        _require(args.arm is None and not args.resume, "P0 import has extra run flags")
        path = import_pepd_p0(protocol, root=root)
        print(json.dumps({
            "status": "p0_pepd_import_complete",
            "path": str(path),
            "receipt_sha256": sha256_file(path),
            "optimizer_steps": 0,
            "images_read": 0,
            "annotations_read": 0,
        }, sort_keys=True), flush=True)
        return 0

    device = _formal_device(args.device)
    current_disk_free = _require_current_disk_space(root)
    if args.mode == "preflight":
        _require(args.arm is None and not args.resume, "preflight mode has extra run flags")
        discovery = load_fit_discovery(protocol, scope="inner_train")
        path = run_preflight(
            protocol, discovery, root=root, device=device, workers=args.workers
        )
    elif args.mode == "train-parent":
        _require(args.arm is None, "train-parent does not accept --arm")
        discovery = load_fit_discovery(protocol, scope="inner_train")
        path = train_parent(
            protocol, discovery, root=root, workers=args.workers,
            device=device, resume=bool(args.resume),
        )
    elif args.mode == "train-gain":
        _require(args.arm is not None, "train-gain requires --arm")
        discovery = load_fit_discovery(protocol, scope="inner_train")
        path = train_gain_arm(
            protocol, discovery, arm=str(args.arm), root=root,
            workers=args.workers, device=device, resume=bool(args.resume),
        )
    elif args.mode == "evaluate-screen":
        _require(args.arm is None, "evaluate-screen does not accept --arm")
        _require(not args.resume, "screen evaluation never permits resume/retry")
        path = evaluate_screen(
            protocol, root=root, workers=args.workers, device=device,
        )
    else:  # pragma: no cover - argparse owns the mode space.
        raise AssertionError(args.mode)
    print(json.dumps({"status": "complete", "mode": args.mode, "artifact": str(path),
                      "disk_free_bytes_at_start": current_disk_free},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
