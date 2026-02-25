#!/usr/bin/env python3
"""Install the NCI GDC Data Transfer Tool (`gdc-client`) locally.

This repo uses `gdc-client` for downloading TCGA/IMPACT artifacts from the GDC.
Rather than making users manually hunt for the right binary, this script downloads
the official release zip for your platform and installs it into `bin/`.

Notes:
- The download source is the official GDC "Data Transfer Tool" page.
- This installs *only* the executable. Do NOT commit tokens to git.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import requests


GDC_TOOL_PAGE = "https://gdc.cancer.gov/access-data/gdc-data-transfer-tool"


def _iter_candidate_urls(html: str) -> Iterable[str]:
    for match in re.finditer(r'href="([^"]+)"', html):
        href = match.group(1)
        if "gdc-client" not in href.lower():
            continue
        if not href.lower().endswith(".zip"):
            continue
        yield href


def _default_platform() -> str:
    plat = sys.platform.lower()
    if plat.startswith("linux"):
        return "linux"
    if plat.startswith("darwin"):
        return "macos-14"
    if plat.startswith("win"):
        return "windows"
    return "linux"


def _select_url(urls: list[str], platform_key: str) -> str:
    key = (platform_key or "").strip().lower()
    if not key:
        key = _default_platform()

    def _first(pred) -> Optional[str]:
        for url in urls:
            if pred(url.lower()):
                return url
        return None

    if key in {"linux", "ubuntu"}:
        selected = _first(lambda u: "ubuntu" in u and "x64" in u)
    elif key in {"windows", "win"}:
        selected = _first(lambda u: "windows" in u and "x64" in u)
    elif key in {"macos", "osx"}:
        selected = _first(lambda u: "osx" in u and "x64" in u)
    elif key in {"macos-14", "macos14"}:
        selected = _first(lambda u: "osx" in u and "macos-14" in u)
    elif key in {"macos-12", "macos12"}:
        selected = _first(lambda u: "osx" in u and "macos-12" in u)
    else:
        raise ValueError(f"Unknown --platform '{platform_key}'. Expected linux|windows|macos-12|macos-14.")

    if not selected:
        preview = "\n".join(urls[:10])
        raise RuntimeError(
            f"Could not find a matching gdc-client zip for platform '{key}'. "
            f"Found URLs (first 10):\n{preview}"
        )
    return selected


def _download(url: str) -> bytes:
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    return resp.content


def _extract_gdc_client(zip_bytes: bytes) -> tuple[bytes, str]:
    """Return (binary_bytes, suggested_filename). Handles nested zip structure."""

    outer = zipfile.ZipFile(io.BytesIO(zip_bytes))
    outer_names = [name for name in outer.namelist() if not name.endswith("/")]
    if not outer_names:
        raise RuntimeError("Downloaded zip is empty.")

    # Newer GDC releases ship a zip that contains an inner zip.
    inner_zip_name = next((name for name in outer_names if name.lower().endswith(".zip")), None)
    if inner_zip_name:
        inner_bytes = outer.read(inner_zip_name)
        inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
        candidates = [name for name in inner.namelist() if not name.endswith("/")]
        target = next(
            (name for name in candidates if Path(name).name.lower() in {"gdc-client", "gdc-client.exe"}),
            None,
        )
        if not target:
            raise RuntimeError(f"Inner zip missing gdc-client binary. Members: {candidates[:20]}")
        return inner.read(target), Path(target).name

    # Fallback: outer zip contains the binary directly.
    target = next(
        (name for name in outer_names if Path(name).name.lower() in {"gdc-client", "gdc-client.exe"}),
        None,
    )
    if not target:
        raise RuntimeError(f"Zip missing gdc-client binary. Members: {outer_names[:20]}")
    return outer.read(target), Path(target).name


def _make_executable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and install gdc-client into this repo.")
    parser.add_argument("--dest", default="bin/gdc-client", help="Install path (default: bin/gdc-client)")
    parser.add_argument(
        "--platform",
        default="",
        help="Override platform selection: linux|windows|macos-12|macos-14 (default: auto)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing destination file")
    parser.add_argument("--verify", action="store_true", help="Run '--version' after install")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not args.force:
        print(f"[gdc-client] Already installed: {dest} (use --force to overwrite)")
        return 0

    print(f"[gdc-client] Resolving download URL from {GDC_TOOL_PAGE}")
    html = requests.get(GDC_TOOL_PAGE, timeout=60).text
    urls = sorted(set(_iter_candidate_urls(html)))
    if not urls:
        raise RuntimeError("No gdc-client download URLs found on the GDC tool page.")

    url = _select_url(urls, args.platform)
    print(f"[gdc-client] Downloading: {url}")
    payload = _download(url)
    binary, suggested_name = _extract_gdc_client(payload)

    final_path = dest
    if suggested_name.lower().endswith(".exe") and dest.suffix.lower() != ".exe":
        final_path = dest.with_suffix(".exe")

    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    tmp.write_bytes(binary)
    tmp.replace(final_path)
    if os.name != "nt":
        _make_executable(final_path)

    print(f"[gdc-client] Installed -> {final_path}")
    if args.verify:
        try:
            result = subprocess.run([str(final_path), "--version"], check=False, capture_output=True, text=True)
            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()
            if out:
                print(out)
            if err:
                print(err, file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[gdc-client] Verification failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

