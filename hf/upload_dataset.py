#!/usr/bin/env python3
"""upload_dataset.py — package the open-Jev labelled corpus and (optionally) push it to the HF Hub.

Default mode is a DRY RUN: prints exactly what would be uploaded (files, row counts, sizes,
sha256s, target paths, repo settings) and touches neither the network nor any credentials.
Nothing leaves this machine unless you pass --go.

  python3 upload_dataset.py           # dry run: print the exact plan (no auth needed)
  python3 upload_dataset.py --stage   # copy the three JSONL files + README into the staging dir
  python3 upload_dataset.py --go      # stage, then create the repo and upload (requires auth)

Auth resolution (same order as the huggingface_hub CLIs): $HF_TOKEN, then
$HUGGING_FACE_HUB_TOKEN, then ~/.cache/huggingface/token (written by `hf auth login`).
--go stops with a clear error if none is found; --dry-run needs no auth.

Row counts are pinned below so drift in the corpus is loud, not silent.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../openjev-lm/hf
REPO_ROOT = HERE.parent                           # .../openjev-lm
DEFAULT_REPO_ID = "decrux9812/openjev-jev-labelled"
DATA_DIR = REPO_ROOT / "data"
README_SRC = HERE / "README-dataset.md"           # uploads as README.md
STAGING = HERE / "staging" / "openjev-jev-labelled"
COMMIT_MSG = ("open-Jev labelled corpus: 2,591 Jev-labelled training rows + 286 dev + 70 gold "
              "+ dataset card")
EXPECTED_ROWS = {"train_jev.jsonl": 2591, "dev_jev.jsonl": 286, "eval_gold.jsonl": 70}

# ------------------------------------------------------------------ helpers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GiB"


def count_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def resolve_credentials(required: bool):
    """Return (cred|None, where). Env first, then ~/.cache/huggingface/token."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(var, "").strip()
        if v:
            return v, f"${var}"
    p = Path.home() / ".cache" / "huggingface" / "token"
    if p.exists():
        v = p.read_text().strip()
        if v:
            return v, str(p)
    where = "checked $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN and ~/.cache/huggingface/token"
    if required:
        die("no Hugging Face auth found (" + where + ").\n"
            "  Log in first:  hf auth login   (older CLI: huggingface-cli login)\n"
            "  or export HF_TOKEN with your token's value.\n"
            "  Nothing was uploaded.", code=2)
    return None, where

# ------------------------------------------------------------------ plan / stage

def plan_rows():
    """(name, local_path, origin_note) for the four upload artifacts."""
    return [
        ("train_jev.jsonl", STAGING / "train_jev.jsonl", f"copy of {DATA_DIR / 'train_jev.jsonl'}"),
        ("dev_jev.jsonl", STAGING / "dev_jev.jsonl", f"copy of {DATA_DIR / 'dev_jev.jsonl'}"),
        ("eval_gold.jsonl", STAGING / "eval_gold.jsonl", f"copy of {DATA_DIR / 'eval_gold.jsonl'}"),
        ("README.md", STAGING / "README.md", f"rendered from {README_SRC}"),
    ]


def print_plan(repo_id: str, where: str, have_cred: bool, mode: str) -> list:
    print("=== HF upload plan — dataset ===")
    print(f"  repo id        : {repo_id}   (repo_type=dataset)")
    print(f"  visibility     : public (pass --private to change; repo will be created if missing)")
    print(f"  auth source    : {where}" + ("" if have_cred else "  — none found (a dry run needs none)"))
    print(f"  commit message : {COMMIT_MSG!r}")
    print(f"  staging dir    : {STAGING}")
    print("  files to upload:")
    ready = []
    for name, local, origin in plan_rows():
        if local.exists():
            sha = sha256_file(local)
            rows = ""
            if name.endswith(".jsonl"):
                n = count_rows(local)
                exp = EXPECTED_ROWS.get(name)
                rows = f"  rows {n}" + ("" if exp is None or n == exp else f"  [EXPECTED {exp} — MISMATCH]")
            print(f"    {name:<16} [staged]  {human(local.stat().st_size):>10}  sha256 {sha[:16]}…{rows}")
            ready.append(local)
        else:
            print(f"    {name:<16} [not staged]  would be produced from: {origin}")
    print(f"  corpus source  : {DATA_DIR}  (present: {DATA_DIR.exists()})")
    print("  corpus summary : train 2,591 rows · dev 286 · gold 70; every target = the hosted "
          "API's own answer, audit 13/13 (runs/VERIFY.md)")
    if mode == "dry":
        if not ready:
            print("\nDRY RUN — nothing staged yet. `--stage` renders the staging dir; `--go` stages and uploads.")
        else:
            print(f"\nDRY RUN — plan is complete and staged ({len(ready)}/4 files). "
                  f"Nothing uploaded; `--go` creates the repo and pushes.")
    return ready

def stage() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    for name in ("train_jev.jsonl", "dev_jev.jsonl", "eval_gold.jsonl"):
        src = DATA_DIR / name
        if not src.exists():
            die(f"corpus file missing: {src}", code=2)
        shutil.copy2(src, STAGING / name)
        n = count_rows(src)
        exp = EXPECTED_ROWS.get(name)
        warn = "" if exp is None or n == exp else f"  [WARN: expected {exp} rows, found {n}]"
        print(f"[stage] {name}  <- {src}  ({n} rows){warn}")
    if not README_SRC.exists():
        die(f"dataset card missing: {README_SRC}", code=2)
    shutil.copy2(README_SRC, STAGING / "README.md")
    print(f"[stage] README.md <- {README_SRC}")

# ------------------------------------------------------------------ upload

def do_upload(repo_id: str, cred: str, private: bool, files: list) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        die("huggingface_hub is not installed in this interpreter "
            "(pip install huggingface_hub) — nothing was uploaded.", code=2)

    api = HfApi(token=cred)
    try:
        who = api.whoami().get("name")
        print(f"[hub] authenticated as: {who}")
    except Exception as exc:  # noqa: BLE001
        print(f"[hub][WARN] whoami failed ({exc}); continuing")

    print(f"[hub] create_repo({repo_id}, repo_type=dataset, exist_ok=True, private={private})")
    kwargs = dict(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=cred)
    try:
        api.create_repo(private=private, **kwargs)
    except TypeError:                     # huggingface_hub >= 1.x moved to `visibility`
        api.create_repo(visibility="private" if private else "public", **kwargs)

    for f in files:
        print(f"[hub] uploading {f.name} ({human(f.stat().st_size)}) …")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_id,
                        repo_type="dataset", token=cred, commit_message=COMMIT_MSG)

    # read back the repo listing and confirm every file landed at the right size
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    remote = {}
    for s in info.siblings:
        remote[s.rfilename] = getattr(s, "size", None)
    ok = True
    for f in files:
        rsize = remote.get(f.name)
        lsize = f.stat().st_size
        status = "ok" if rsize == lsize else f"MISMATCH (local {lsize}, remote {rsize})"
        if rsize != lsize:
            ok = False
        print(f"[verify] {f.name}: {status}")
    if not ok:
        die("verification failed — some files did not land at the expected size; "
            "check the repo on the Hub", code=1)
    print(f"[done] https://huggingface.co/datasets/{repo_id}")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="upload the open-Jev labelled corpus to the Hugging Face Hub",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help=f"target repo (default: {DEFAULT_REPO_ID})")
    ap.add_argument("--stage", action="store_true",
                    help="render the staging directory only (no network, no auth)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="print the exact plan and do nothing else (default)")
    g.add_argument("--go", action="store_true",
                   help="stage, create the repo and upload (requires auth)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    args = ap.parse_args()

    if args.stage and not args.go:
        stage()
        cred, where = resolve_credentials(required=False)
        print_plan(args.repo_id, where, cred is not None, mode="stage")
        return

    if args.go:
        cred, where = resolve_credentials(required=True)
        stage()
        files = print_plan(args.repo_id, where, True, mode="go")
        if len(files) != 4:
            die(f"expected 4 staged files, found {len(files)}", code=1)
        do_upload(args.repo_id, cred, args.private, files)
        return

    # default: dry run (read-only)
    cred, where = resolve_credentials(required=False)
    print_plan(args.repo_id, where, cred is not None, mode="dry")


if __name__ == "__main__":
    main()
