#!/usr/bin/env python3
"""upload_model.py — package the open-Jev 0.5B LoRA adapter and (optionally) push it to the HF Hub.

Default mode is a DRY RUN. It prints exactly what would be uploaded — files, sizes, sha256s,
target paths, repo settings, commit message — and touches neither the network nor any token.
Nothing leaves this machine unless you pass --go.

  python3 upload_model.py            # dry run: print the exact plan (no token needed)
  python3 upload_model.py --stage    # render the staging dir only (copies adapter, writes
                                     #   README.md + MODEL_INFO.json); no network, no token
  python3 upload_model.py --go       # stage, then create the repo and upload (requires a token)

Token resolution (same order as the huggingface_hub CLIs):
  1. $HF_TOKEN   2. $HUGGING_FACE_HUB_TOKEN   3. ~/.cache/huggingface/token  (written by
  `hf auth login` / `huggingface-cli login`).  --go stops with a clear error if none is found;
  --dry-run needs no token and never reads one beyond reporting where one would come from.

Every number in MODEL_INFO.json / README.md is frozen from the source repo's runs/FACTS.json
(generated 2026-09-18) and the run receipts; the adapter file's digest is pinned below so drift
in either the weights or the facts is loud, not silent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../openjev-lm/hf
REPO_ROOT = HERE.parent                           # .../openjev-lm
DEFAULT_REPO_ID = "decrux9812/openjev-0.5b"
ADAPTER_SRC = Path("/home/decrux/Code/jev-repro-test/openjev/runs/run-final/adapter.pt")
# sha256 of the staged adapter, split so it cannot be mistaken for a secret; compare with
# `sha256sum /home/decrux/Code/jev-repro-test/openjev/runs/run-final/adapter.pt`
ADAPTER_SHA256 = ("e49b717438fa54ea2b" "f03dd102ea7229045a" "bf9e514db17b0ef5a9" "3aefa027fe")
README_SRC = HERE / "README-model.md"              # uploads as README.md
STAGING = HERE / "staging" / "openjev-0.5b"
COMMIT_MSG = "open-Jev 0.5B: LoRA adapter (400 steps, CPU) + model card + MODEL_INFO.json"

# ------------------------------------------------------------------ frozen facts (runs/FACTS.json)

def model_info(adapter_sha256: str) -> dict:
    """MODEL_INFO.json for the repo. Numbers pinned from runs/FACTS.json (2026-09-18)."""
    return {
        "artifact": "open-Jev 0.5B — LoRA adapter for Qwen2.5-0.5B-Instruct (job-posting decision task)",
        "file": "adapter.pt",
        "sha256": adapter_sha256,
        "format": "torch.save zip checkpoint (PyTorch pickle); dict keys: format=2, lora, opt, step",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora": {
            "rank": 16,
            "alpha": 32,
            "scale": 2.0,
            "dropout_train": 0.05,
            "dropout_inference": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "wrapped_modules": 96,
            "tensors": 192,
            "trainable_params": 2162688,
            "base_params_frozen": True,
        },
        "training": {
            "run": "run-final",
            "seed": 7,
            "dataset": "decrux9812/openjev-jev-labelled",
            "train_rows": 2591,
            "train_file": "data/train_jev.jsonl",
            "steps": 400,
            "batch": 6,
            "max_len": 192,
            "lr": 1.5e-4,
            "schedule": "OneCycleLR (max_lr=1.5e-4, pct_start=0.05, cos)",
            "loss": {"first": 0.6792, "final": 0.0468, "mean_last10": 0.0253},
            "wall_min": 89.3,
            "chunks": 3,
            "host": "6 vCPU CPU-only (no CUDA)",
        },
        "evaluation": {
            "gold_rows": 70,
            "bucket_correct": 65,
            "bucket_accuracy": 0.9286,
            "per_class_recall": {"service_lead": "1/2", "staff_role": "12/14", "generic_job": "52/54"},
            "mean_bucket_confidence": 0.945,
            "harness_a": "eval_openjev.py (parallel constrained decoding)",
            "harness_b": "eval_likelihood.py (staged whole-answer likelihood)",
            "cross_harness": "both 92.9% (65/70), row-for-row bucket agreement",
        },
        "reference_points": {
            "jev_hosted_bucket_accuracy": 0.9714,
            "jev_hosted_detail": "68/70 on the same gold rows",
            "untrained_qwen2_5_0_5b_instruct": 0.7286,
            "stock_qwen2_5_1_5b_plus_published_decoder": 0.771,
        },
        "provenance": {
            "author": "Ritesh Patel (GitHub DECRUX9812)",
            "license": "mit",
            "note": ("Independent workalike. Trained only on input/output pairs obtained by calling "
                     "the hosted Jev API; no Jev weights or internals used; no affiliation with TypeSafe."),
            "receipts": ["runs/FACTS.json", "runs/VERIFY.md", "runs/eval_A_run-final.json",
                          "runs/eval_B_run-final.json"],
        },
    }

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


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def resolve_token(required: bool):
    """Return (token|None, where). Env first, then ~/.cache/huggingface/token."""
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
        die("no Hugging Face token found (" + where + ").\n"
            "  Log in first:  hf auth login   (older CLI: huggingface-cli login)\n"
            "  or export HF_TOKEN with your token's value.\n"
            "  Nothing was uploaded.", code=2)
    return None, where

# ------------------------------------------------------------------ plan / stage

def plan_rows():
    """(name, local_path, origin_note) for the three upload artifacts."""
    return [
        ("adapter.pt", STAGING / "adapter.pt", f"copy of {ADAPTER_SRC}"),
        ("README.md", STAGING / "README.md", f"rendered from {README_SRC}"),
        ("MODEL_INFO.json", STAGING / "MODEL_INFO.json",
         "generated from the frozen facts inside this script + the adapter's sha256"),
    ]


def print_plan(repo_id: str, token_where: str, have_token: bool, mode: str) -> list:
    print("=== HF upload plan — model ===")
    print(f"  repo id        : {repo_id}   (repo_type=model)")
    print(f"  visibility     : public (pass --private to change; repo will be created if missing)")
    print(f"  auth source    : {token_where}" + ("" if have_token else "  — none found (a dry run needs none)"))
    print(f"  commit message : {COMMIT_MSG!r}")
    print(f"  staging dir    : {STAGING}")
    print("  files to upload:")
    ready = []
    for name, local, origin in plan_rows():
        if local.exists():
            sha = sha256_file(local)
            note = f"sha256 {sha[:16]}…"
            if name == "adapter.pt":
                match = "match" if sha == ADAPTER_SHA256 else "MISMATCH — expected " + ADAPTER_SHA256[:16] + "…"
                note += f"  (vs pinned source sha256: {match})"
            print(f"    {name:<16} [staged]  {human(local.stat().st_size):>10}  {note}")
            ready.append(local)
        else:
            print(f"    {name:<16} [not staged]  would be produced from: {origin}")
    src_state = "present" if ADAPTER_SRC.exists() else "MISSING"
    print(f"  adapter source : {ADAPTER_SRC} ({src_state})")
    print(f"  adapter config : r=16, alpha=32, q/k/v/o, 2,162,688 trainable params, step=400, "
          f"base Qwen/Qwen2.5-0.5B-Instruct")
    if mode == "dry":
        if not ready:
            print("\nDRY RUN — nothing staged yet. `--stage` renders the staging dir; `--go` stages and uploads.")
        else:
            print(f"\nDRY RUN — plan is complete and staged ({len(ready)}/3 files). "
                  f"Nothing uploaded; `--go` creates the repo and pushes.")
    return ready

def stage() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    dst = STAGING / "adapter.pt"
    if ADAPTER_SRC.exists():
        shutil.copy2(ADAPTER_SRC, dst)
        print(f"[stage] adapter.pt  <- {ADAPTER_SRC}")
    elif not dst.exists():
        die(f"adapter not found at {ADAPTER_SRC} and no staged copy at {dst}", code=2)
    sha = sha256_file(dst)
    if sha != ADAPTER_SHA256:
        print(f"[stage][WARN] staged adapter sha256 {sha} != pinned {ADAPTER_SHA256}; "
              f"MODEL_INFO.json will carry the staged hash — re-check the facts before uploading")
    if not README_SRC.exists():
        die(f"model card missing: {README_SRC}", code=2)
    shutil.copy2(README_SRC, STAGING / "README.md")
    (STAGING / "MODEL_INFO.json").write_text(json.dumps(model_info(sha), indent=2) + "\n")
    print(f"[stage] README.md, MODEL_INFO.json -> {STAGING}")
    print(f"[stage] adapter sha256 {sha}" + (" (matches pinned value)" if sha == ADAPTER_SHA256 else ""))

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

    print(f"[hub] create_repo({repo_id}, repo_type=model, exist_ok=True, private={private})")
    kwargs = dict(repo_id=repo_id, repo_type="model", exist_ok=True, token=cred)
    try:
        api.create_repo(private=private, **kwargs)
    except TypeError:                     # huggingface_hub >= 1.x moved to `visibility`
        api.create_repo(visibility="private" if private else "public", **kwargs)

    for f in files:
        print(f"[hub] uploading {f.name} ({human(f.stat().st_size)}) …")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_id,
                        repo_type="model", token=cred, commit_message=COMMIT_MSG)

    # read back the repo listing and confirm every file landed at the right size
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
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
    print(f"[done] https://huggingface.co/{repo_id}")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="upload the open-Jev 0.5B adapter to the Hugging Face Hub",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help=f"target repo (default: {DEFAULT_REPO_ID})")
    ap.add_argument("--stage", action="store_true",
                    help="render the staging directory only (no network, no credentials)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="print the exact plan and do nothing else (default)")
    g.add_argument("--go", action="store_true",
                   help="stage, create the repo and upload (requires auth)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    args = ap.parse_args()

    if args.stage and not args.go:
        stage()
        cred, where = resolve_token(required=False)
        print_plan(args.repo_id, where, cred is not None, mode="stage")
        return

    if args.go:
        cred, where = resolve_token(required=True)
        stage()
        files = print_plan(args.repo_id, where, True, mode="go")
        if len(files) != 3:
            die(f"expected 3 staged files, found {len(files)}", code=1)
        do_upload(args.repo_id, cred, args.private, files)
        return

    # default: dry run (read-only)
    cred, where = resolve_token(required=False)
    print_plan(args.repo_id, where, cred is not None, mode="dry")


if __name__ == "__main__":
    main()
