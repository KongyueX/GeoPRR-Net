"""Fetch and authenticate the frozen public YOLO11n initialization checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


SAFE_OUTPUT_ROOT = Path(r"C:\pointer_read").resolve()
DEFAULT_OUTPUT = (
    SAFE_OUTPUT_ROOT / "public_pretrained/yolo11n-ultralytics-assets-v8.3.0.pt"
)
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
EXPECTED_BYTES = 5_613_764
EXPECTED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(path: Path) -> dict[str, object]:
    output = Path(path).resolve()
    try:
        output.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"pretrained checkpoint must stay below {SAFE_OUTPUT_ROOT}") from error
    if output.is_file():
        if output.stat().st_size != EXPECTED_BYTES or sha256_file(output) != EXPECTED_SHA256:
            raise ValueError("existing public pretrained checkpoint differs from the frozen identity")
        return {
            "status": "already_present_verified",
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": EXPECTED_SHA256,
            "public_url": URL,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.download.{os.getpid()}")
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size != EXPECTED_BYTES:
            raise ValueError("downloaded public checkpoint byte-size drift")
        if sha256_file(temporary) != EXPECTED_SHA256:
            raise ValueError("downloaded public checkpoint SHA-256 drift")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "downloaded_verified",
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": EXPECTED_SHA256,
        "public_url": URL,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(fetch(args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
