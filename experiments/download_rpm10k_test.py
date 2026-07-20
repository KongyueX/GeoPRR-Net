"""Download only the frozen RPM-10K single-pointer external-test subset.

Google Drive's anonymous ``embeddedfolderview`` truncates this release at
5,500 files.  This downloader opens the public folder in an isolated headless
Chrome context, captures Drive's anonymous paginated list request, and then
uses that request to enumerate the complete official image folder.  No Google
account, cookie, or private API credential is used.

Run from the repository root:

    python -m experiments.download_rpm10k_test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gdown
import requests
from playwright.sync_api import sync_playwright

from experiments.datasets import (
    RPM10K_SINGLE_POINTER_ROWS,
    RPM10K_TEST_SHA256,
    select_rpm10k_single_pointer_rows,
)


RPM10K_ROOT_FOLDER_ID = "1uw2FkX89XZseDcw_P-BSP-_AYZP5-W10"
RPM10K_IMAGES_FOLDER_ID = "1z2aXffH8A7PAoQ4X-gRnp3h4BpjaEMLc"
RPM10K_TEST_LABEL_FILE_ID = "1UAOmOfj9gdJ6YXNp9FTfJwl54U8LRS23"
RPM10K_RELEASE_IMAGE_COUNT = 10_730
DRIVE_LIST_ENDPOINT_FRAGMENT = "drivefrontend-pa.clients6.google.com/v1/items:list"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proxy_map(proxy: str) -> dict[str, str] | None:
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _playwright_proxy(proxy: str) -> dict[str, str] | None:
    if not proxy:
        return None
    # Chromium accepts socks5:// but not requests' DNS-explicit socks5h://.
    return {"server": proxy.replace("socks5h://", "socks5://", 1)}


def _find_chrome(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Chrome executable not found: {explicit}")
        return explicit.resolve()

    candidates = [
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", ""))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", ""))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", ""))
        / "Microsoft/Edge/Application/msedge.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Chrome/Edge was not found; pass --chrome-path to a Chromium executable"
    )


def _download_pinned_labels(output: Path, proxy: str) -> Path:
    labels_dir = output / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    destination = labels_dir / "test.json"
    if destination.is_file() and _sha256(destination) == RPM10K_TEST_SHA256:
        print(f"verified, skipping {destination}")
        return destination

    temporary = destination.with_name(destination.name + ".download")
    temporary.unlink(missing_ok=True)
    result = gdown.download(
        id=RPM10K_TEST_LABEL_FILE_ID,
        output=str(temporary),
        quiet=False,
        proxy=proxy or None,
        resume=True,
    )
    if not result or not temporary.is_file():
        raise RuntimeError("gdown did not download RPM-10K test.json")
    actual = _sha256(temporary)
    if actual != RPM10K_TEST_SHA256:
        raise RuntimeError(
            f"RPM-10K test.json SHA-256 is {actual}, expected {RPM10K_TEST_SHA256}"
        )
    os.replace(temporary, destination)
    return destination


def _capture_anonymous_list_request(
    chrome_path: Path,
    proxy: str,
) -> tuple[str, dict[str, str], list[Any]]:
    folder_url = (
        "https://drive.google.com/drive/folders/"
        f"{RPM10K_IMAGES_FOLDER_ID}?usp=sharing"
    )
    captured = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=str(chrome_path),
            proxy=_playwright_proxy(proxy),
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on(
                "request",
                lambda request: captured.append(request)
                if DRIVE_LIST_ENDPOINT_FRAGMENT in request.url
                else None,
            )
            page.goto(folder_url, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_selector(
                '[data-id][data-tooltip*="_img.jpg"]',
                timeout=60_000,
            )
            page.wait_for_timeout(4_000)

            scroll = page.locator("c-wiz.PEfnhb.v4kGBb.NtyuW")
            box = scroll.bounding_box()
            if box is None:
                raise RuntimeError("could not locate the Google Drive file list")
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            last_scroll_state = None
            for _ in range(30):
                scroll.evaluate(
                    """element => {
                        element.scrollTop = element.scrollHeight;
                        element.dispatchEvent(new Event("scroll", {bubbles: true}));
                    }"""
                )
                page.mouse.wheel(0, 2400)
                page.wait_for_timeout(600)
                if captured:
                    break
                last_scroll_state = scroll.evaluate(
                    "element => [element.scrollTop, element.scrollHeight, element.clientHeight]"
                )
            if not captured:
                raise RuntimeError(
                    "Google Drive did not issue a paginated list request; "
                    f"last scroll state={last_scroll_state}"
                )

            request = captured[0]
            url = request.url
            all_headers = request.all_headers()
            post_data = request.post_data
            if not post_data:
                raise RuntimeError("captured Google Drive request has no payload")
            body = json.loads(post_data)
        finally:
            browser.close()

    allowed_headers = {
        "content-type",
        "x-goog-fieldmask",
        "x-goog-drive-client-version",
        "x-goog-ext-472780938-jspb",
        "x-goog-ext-477772811-jspb",
        "user-agent",
        "accept-language",
    }
    headers = {
        key: value
        for key, value in all_headers.items()
        if key.lower() in allowed_headers
    }
    headers.update(
        {
            "origin": "https://drive.google.com",
            "referer": "https://drive.google.com/",
        }
    )
    return url, headers, body


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    retries: int,
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.request(method, url, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}",
                    response=response,
                )
            response.raise_for_status()
            return response
        except (requests.RequestException, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from last_error


def _list_complete_image_folder(
    chrome_path: Path,
    proxy: str,
    retries: int,
) -> dict[str, dict[str, Any]]:
    url, headers, body = _capture_anonymous_list_request(chrome_path, proxy)
    if (
        not isinstance(body, list)
        or len(body) < 2
        or not isinstance(body[1], list)
        or len(body[1]) < 2
    ):
        raise RuntimeError("unexpected Google Drive list request schema")

    body[1][0] = 1000
    body[1][1] = ""
    session = requests.Session()
    if proxy:
        session.proxies.update(_proxy_map(proxy) or {})

    image_index: dict[str, dict[str, Any]] = {}
    seen_tokens: set[str] = set()
    page_number = 0
    while True:
        page_number += 1
        response = _request_with_retry(
            session,
            "POST",
            url,
            retries=retries,
            headers=headers,
            json=body,
            timeout=90,
        )
        payload = response.json()
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
            raise RuntimeError("unexpected Google Drive list response schema")

        for item in payload[0]:
            if not isinstance(item, list) or len(item) <= 13:
                continue
            file_id, name, file_size = item[0], item[2], item[13]
            if (
                isinstance(file_id, str)
                and isinstance(name, str)
                and name.lower().endswith(".jpg")
                and file_size is not None
            ):
                image_index[name] = {"id": file_id, "size": int(file_size)}

        token = payload[1] if len(payload) > 1 else ""
        print(
            f"indexed Drive page {page_number}: "
            f"{len(payload[0])} rows, {len(image_index)} images"
        )
        if not token:
            break
        if not isinstance(token, str) or token in seen_tokens:
            raise RuntimeError("invalid/repeated Google Drive continuation token")
        seen_tokens.add(token)
        body[1][1] = token
        if page_number > 100:
            raise RuntimeError("Google Drive pagination exceeded the safety limit")

    if len(image_index) != RPM10K_RELEASE_IMAGE_COUNT:
        raise RuntimeError(
            f"Drive contains {len(image_index)} images, "
            f"expected {RPM10K_RELEASE_IMAGE_COUNT}; release drift or partial listing"
        )
    return image_index


_thread_local = threading.local()


def _thread_session(proxy: str) -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        if proxy:
            session.proxies.update(_proxy_map(proxy) or {})
        _thread_local.session = session
    return session


def _download_image(
    name: str,
    item: dict[str, Any],
    image_dir: Path,
    proxy: str,
    retries: int,
) -> str:
    destination = image_dir / name
    expected_size = int(item["size"])
    if destination.is_file() and destination.stat().st_size == expected_size:
        return "skipped"

    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > expected_size:
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            response = _thread_session(proxy).get(
                DOWNLOAD_URL,
                params={
                    "id": item["id"],
                    "export": "download",
                    "confirm": "t",
                },
                headers=headers,
                stream=True,
                timeout=(30, 120),
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}",
                    response=response,
                )
            response.raise_for_status()

            append = bool(offset and response.status_code == 206)
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            if partial.stat().st_size != expected_size:
                raise RuntimeError(
                    f"{name}: got {partial.stat().st_size} bytes, "
                    f"expected {expected_size}"
                )
            os.replace(partial, destination)
            return "downloaded"
        except (OSError, requests.RequestException, RuntimeError) as exc:
            if attempt + 1 >= retries:
                raise RuntimeError(f"failed to download {name}") from exc
            time.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/RPM10K"))
    parser.add_argument(
        "--proxy",
        default="socks5h://127.0.0.1:7890",
        help='HTTP/SOCKS proxy; pass "" for a direct connection',
    )
    parser.add_argument("--chrome-path", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, help="diagnostic download limit")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="download labels and enumerate Drive without downloading images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1:
        raise ValueError("--workers and --retries must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = _download_pinned_labels(output, args.proxy)
    label_rows = json.loads(labels.read_text(encoding="utf-8"))
    selected, audit = select_rpm10k_single_pointer_rows(label_rows)
    if len(selected) != RPM10K_SINGLE_POINTER_ROWS:
        raise RuntimeError(
            f"protocol selected {len(selected)} rows, "
            f"expected {RPM10K_SINGLE_POINTER_ROWS}"
        )
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    target_names = [str(row["image"]) for row in selected]
    index_path = output / "labels" / "test_drive_index.json"
    image_index: dict[str, dict[str, Any]] | None = None
    if index_path.is_file():
        try:
            cached = json.loads(index_path.read_text(encoding="utf-8"))
            cached_targets = cached.get("targets")
            if (
                cached.get("images_folder_id") == RPM10K_IMAGES_FOLDER_ID
                and cached.get("release_image_count") == RPM10K_RELEASE_IMAGE_COUNT
                and isinstance(cached_targets, dict)
                and set(target_names).issubset(cached_targets)
            ):
                image_index = {
                    name: cached_targets[name]
                    for name in target_names
                }
                print(f"verified, reusing Drive index {index_path}")
        except (OSError, ValueError, TypeError):
            image_index = None

    if image_index is None:
        chrome_path = _find_chrome(args.chrome_path)
        complete_index = _list_complete_image_folder(
            chrome_path,
            args.proxy,
            args.retries,
        )
        image_index = {
            name: complete_index[name]
            for name in target_names
            if name in complete_index
        }
        release_image_count = len(complete_index)
    else:
        release_image_count = RPM10K_RELEASE_IMAGE_COUNT

    missing = sorted(set(target_names) - set(image_index))
    if missing:
        raise RuntimeError(
            f"{len(missing)} protocol images are missing from Drive; "
            f"examples: {missing[:10]}"
        )

    index_path.write_text(
        json.dumps(
            {
                "root_folder_id": RPM10K_ROOT_FOLDER_ID,
                "images_folder_id": RPM10K_IMAGES_FOLDER_ID,
                "release_image_count": release_image_count,
                "protocol": audit,
                "targets": {name: image_index[name] for name in target_names},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.list_only:
        print(f"listed {len(target_names)} target images in {index_path}")
        return

    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    downloaded = 0
    skipped = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _download_image,
                name,
                image_index[name],
                image_dir,
                args.proxy,
                args.retries,
            ): name
            for name in target_names
        }
        for future in as_completed(futures):
            outcome = future.result()
            completed += 1
            if outcome == "downloaded":
                downloaded += 1
            else:
                skipped += 1
            if completed % 50 == 0 or completed == len(futures):
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"{completed}/{len(futures)} images; "
                    f"downloaded={downloaded}, skipped={skipped}, "
                    f"{completed / elapsed:.1f} files/s"
                )

    print(
        f"RPM-10K single-pointer subset ready: {len(target_names)} images below "
        f"{image_dir}"
    )


if __name__ == "__main__":
    main()
