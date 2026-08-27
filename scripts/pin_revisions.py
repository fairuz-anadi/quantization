"""Pin exact commit SHAs for the model(s) and the benchmark dataset.

A tag or branch name is not a pin: `main` moves, and a re-run months later can
silently evaluate different weights. Every run records the 40-char commit SHA,
and `config.require()` refuses to start a run while any revision is null.

This talks to the Hugging Face Hub only. Kaggle credentials are not involved.

Run:  python scripts/pin_revisions.py          # pin anything still null
      python scripts/pin_revisions.py --check  # verify pins, change nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi  # noqa: E402

from quantlang import config  # noqa: E402

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _fatal(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def _check_sha(sha: str | None, what: str) -> str:
    if not sha or not SHA_RE.match(sha):
        _fatal(
            f"{what} resolved to {sha!r}, which is not a 40-char commit SHA. "
            f"Refusing to record a value that is not a real pin."
        )
    return sha


def resolve(api: HfApi, cfg: dict) -> dict:
    bench = cfg["benchmark"]
    out: dict = {"models": {}}

    ds_id = bench["hf_dataset"]
    info = api.dataset_info(ds_id)
    out["dataset"] = {
        "hf_id": ds_id,
        "sha": _check_sha(info.sha, f"dataset {ds_id}"),
        "last_modified": str(getattr(info, "last_modified", "")),
    }
    print(f"dataset {ds_id}\n  sha = {out['dataset']['sha']}")

    for m in cfg["models"]:
        mid = m["hf_id"]
        mi = api.model_info(mid)
        sha = _check_sha(mi.sha, f"model {mid}")
        out["models"][m["alias"]] = {
            "hf_id": mid,
            "sha": sha,
            "last_modified": str(getattr(mi, "last_modified", "")),
            "gated": bool(getattr(mi, "gated", False)),
        }
        print(f"model {mid}\n  sha = {sha}\n  gated = {out['models'][m['alias']]['gated']}")
    return out


def write_back(resolved: dict) -> None:
    """Surgically replace the null revisions, preserving comments."""
    path = config.CONFIG_PATH
    lines = path.read_text(encoding="utf-8").splitlines()

    ds_sha = resolved["dataset"]["sha"]
    replaced_ds = False
    for i, line in enumerate(lines):
        if re.match(r"^\s*hf_revision:\s*null", line):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f'{indent}hf_revision: "{ds_sha}"   # pinned by scripts/pin_revisions.py'
            replaced_ds = True
            break
    if not replaced_ds:
        print("  (benchmark.hf_revision already pinned; left untouched)")

    # For each model, find its hf_id line then the first `revision:` after it.
    for alias, info in resolved["models"].items():
        target = None
        for i, line in enumerate(lines):
            if re.match(rf"^\s*hf_id:\s*{re.escape(info['hf_id'])}\s*$", line):
                target = i
                break
        if target is None:
            _fatal(f"could not locate hf_id line for {info['hf_id']} in the config")
        for j in range(target, min(target + 6, len(lines))):
            if re.match(r"^\s*revision:\s*null", lines[j]):
                indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
                lines[j] = (
                    f'{indent}revision: "{info["sha"]}"   '
                    f"# commit SHA, pinned by scripts/pin_revisions.py"
                )
                break
        else:
            print(f"  ({alias}.revision already pinned; left untouched)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = ap.parse_args()

    cfg = config.load()
    api = HfApi()
    resolved = resolve(api, cfg)

    if args.check:
        cfg = config.load()
        ds = config.require(cfg, "benchmark.hf_revision")
        if ds != resolved["dataset"]["sha"]:
            _fatal(
                f"benchmark.hf_revision pinned to {ds} but the Hub now reports "
                f"{resolved['dataset']['sha']}. The upstream dataset moved; "
                f"existing results are tied to the pinned SHA."
            )
        for m in cfg["models"]:
            if m.get("revision") != resolved["models"][m["alias"]]["sha"]:
                _fatal(
                    f"{m['alias']} pinned to {m.get('revision')} but the Hub now "
                    f"reports {resolved['models'][m['alias']]['sha']}."
                )
        print("\n--check: all pins match the Hub.")
        return 0

    write_back(resolved)

    prov = config.REPO_ROOT / "configs" / "revisions.json"
    prov.write_text(
        json.dumps(
            {"pinned_at_utc": datetime.now(timezone.utc).isoformat(), **resolved},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {prov.relative_to(config.REPO_ROOT)}")

    cfg = config.load()
    print(f"benchmark.hf_revision = {config.require(cfg, 'benchmark.hf_revision')}")
    for m in cfg["models"]:
        if m.get("revision") is None:
            _fatal(f"{m['alias']}.revision still null after write-back")
        print(f"{m['alias']}.revision = {m['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
