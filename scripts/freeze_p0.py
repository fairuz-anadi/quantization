"""Freeze the P0 record so P1 work cannot change it silently.

P0 is complete and its numbers are in the paper's chain of custody. Everything
P1 adds sits alongside P0, never on top of it, and this script is what makes
that checkable rather than merely intended.

Three kinds of entry are tracked, and the distinction matters:

  STRICT      Bytes define P0. P1 must never touch them. A mismatch is a
              failure.
  ADDITIVE    Section 11 of the P1 brief permits an optional local-checkpoint
              argument on `model.load`, so these files are allowed to grow.
              Their P0 baseline digest is recorded for the record, and their
              P0-relevant BEHAVIOUR is pinned by tests/test_p1_freeze.py
              instead of by bytes.
  PROVENANCE  The per-item raw output of the real P0 Kaggle run is NOT in this
              repository -- results/ALL_P0_RESULTS/raw/ holds only its README,
              so tidy.csv currently has no committed source. Those entries are
              registered as null, which in this repo means NOT YET KNOWN and is
              fatal at read. They are never fabricated. Drop the real files
              into results/ALL_P0_RESULTS/raw/ and re-run --register.

Digests are taken over newline-normalised bytes. This checkout has CRLF in
configs/item_id_manifest.json and LF in results/, so a raw-byte digest would
depend on how git happened to check a file out rather than on its content.

    python scripts/freeze_p0.py --register   # write configs/p0_freeze.json
    python scripts/freeze_p0.py              # verify, change nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang.config import REPO_ROOT  # noqa: E402

FREEZE_PATH = REPO_ROOT / "configs" / "p0_freeze.json"

# The P0 subtree of experiment.yaml. Adding a NEW top-level key (`finetune`)
# leaves this digest untouched; editing any P0 value breaks it. That is exactly
# the boundary P1 has to respect.
P0_CONFIG_KEYS = ("benchmark", "models", "precisions", "scoring", "stats")

STRICT_FILES = (
    "configs/item_id_manifest.json",
    "configs/revisions.json",
    "quantlang/data.py",
    "quantlang/schema.py",
    "quantlang/statistics.py",
    "results/ALL_P0_RESULTS/tables/tidy.csv",
    "results/ALL_P0_RESULTS/tables/accuracy.csv",
    "results/ALL_P0_RESULTS/tables/degradation.csv",
    "results/ALL_P0_RESULTS/tables/interaction.csv",
    "results/ALL_P0_RESULTS/tables/latency.csv",
    "results/ALL_P0_RESULTS/tables/summary.txt",
)

ADDITIVE_FILES = (
    "quantlang/config.py",
    "quantlang/model.py",
    "quantlang/evaluate.py",
)

# Where the real P0 run's per-item output belongs, and what it is called.
# evaluate.py writes {tag}__{alias}__{lang}__{precision}.{jsonl,meta.json};
# the P0 run used tag "p0".
P0_RAW_DIR = "results/ALL_P0_RESULTS/raw"
P0_TAG = "p0"


class FreezeError(RuntimeError):
    """Raised when the P0 record does not match what was registered."""


def sha256_text(path: Path) -> str:
    """Digest over newline-normalised bytes, so the value is checkout-portable."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def config_p0_digest(cfg: dict | None = None) -> str:
    """Digest the P0 subtree of experiment.yaml, canonicalised.

    Canonical JSON with sorted keys, so the digest tracks P0's VALUES and is
    indifferent to comment edits, key order, or a new top-level P1 block.
    """
    cfg = cfg if cfg is not None else cfg_mod.load()
    missing = [k for k in P0_CONFIG_KEYS if k not in cfg]
    if missing:
        raise FreezeError(f"experiment.yaml is missing P0 key(s): {missing}")
    subtree = {k: cfg[k] for k in P0_CONFIG_KEYS}
    blob = json.dumps(subtree, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def expected_p0_raw_files(cfg: dict | None = None) -> list[str]:
    """The 30 files the real P0 run produced, named as evaluate.py names them."""
    cfg = cfg if cfg is not None else cfg_mod.load()
    langs = cfg_mod.require(cfg, "benchmark.languages")
    precisions = cfg_mod.require(cfg, "precisions")
    models = cfg_mod.require(cfg, "models")
    primary = [m for m in models if m.get("role") == "primary"]
    if len(primary) != 1:
        raise FreezeError(
            f"expected exactly one role=primary model, found {len(primary)}")
    alias = primary[0]["alias"]
    out: list[str] = []
    for prec in precisions:
        for lang in langs:
            stem = f"{P0_TAG}__{alias}__{lang}__{prec}"
            out.append(f"{P0_RAW_DIR}/{stem}.jsonl")
            out.append(f"{P0_RAW_DIR}/{stem}.meta.json")
    return out


def _scan(rels) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for rel in rels:
        path = REPO_ROOT / rel
        if path.exists():
            entries[rel] = {"sha256": sha256_text(path), "status": "present"}
        else:
            # null == NOT YET KNOWN, per configs/README.md. Never a placeholder.
            entries[rel] = {"sha256": None, "status": "MISSING"}
    return entries


def build_registry() -> dict:
    cfg = cfg_mod.load()
    return {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_by": "scripts/freeze_p0.py",
        "digest_note": (
            "sha256 over newline-normalised bytes (CRLF -> LF). Raw-byte digests "
            "would depend on git's checkout, not on content."
        ),
        "config_p0_subtree": {
            "keys": list(P0_CONFIG_KEYS),
            "canonicalisation": "json.dumps(sort_keys=True, separators=(',',':'))",
            "sha256": config_p0_digest(cfg),
            "note": (
                "Covers P0's VALUES only. A new top-level key such as `finetune` "
                "does not change this digest; editing a P0 value does."
            ),
        },
        "strict_files": _scan(STRICT_FILES),
        "additive_files": {
            "note": (
                "P0 baseline digests, recorded for the record. P1 may extend "
                "these additively (brief section 11). Byte drift here is NOT a "
                "guard failure; tests/test_p1_freeze.py pins the P0-relevant "
                "behaviour instead."
            ),
            "files": _scan(ADDITIVE_FILES),
        },
        "p0_raw_provenance": {
            "note": (
                "The real P0 Kaggle run's per-item output is not committed: "
                f"{P0_RAW_DIR}/ holds only its README, so "
                "results/ALL_P0_RESULTS/tables/tidy.csv has no committed source. "
                "Entries below are null (NOT YET KNOWN) until the actual files "
                "are restored. Re-run `python scripts/freeze_p0.py --register` "
                "to register them. Nothing here is ever fabricated."
            ),
            "tag": P0_TAG,
            "files": _scan(expected_p0_raw_files(cfg)),
        },
    }


def load_registry(path: Path | None = None) -> dict:
    path = path or FREEZE_PATH
    if not path.exists():
        raise FreezeError(
            f"P0 freeze registry missing: {path}\n"
            f"Run `python scripts/freeze_p0.py --register` before starting P1 work."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def check(registry: dict | None = None) -> list[str]:
    """Return a list of violations. Empty means P0 is intact."""
    reg = registry or load_registry()
    problems: list[str] = []

    want = reg["config_p0_subtree"]["sha256"]
    got = config_p0_digest()
    if got != want:
        problems.append(
            f"experiment.yaml P0 subtree changed.\n"
            f"    registered: {want}\n"
            f"    current:    {got}\n"
            f"    Keys covered: {reg['config_p0_subtree']['keys']}. P1 must add "
            f"a NEW top-level key, never edit a P0 value."
        )

    for rel, entry in reg["strict_files"].items():
        path = REPO_ROOT / rel
        if entry["sha256"] is None:
            problems.append(
                f"{rel}: registered as MISSING; nothing to verify against.")
            continue
        if not path.exists():
            problems.append(f"{rel}: registered but now absent from the repository.")
            continue
        got = sha256_text(path)
        if got != entry["sha256"]:
            problems.append(
                f"{rel}: content changed.\n"
                f"    registered: {entry['sha256']}\n"
                f"    current:    {got}"
            )
    return problems


def unregistered_provenance(registry: dict | None = None) -> list[str]:
    """P0 raw files that are still absent. Reported, never invented."""
    reg = registry or load_registry()
    return sorted(
        rel for rel, e in reg["p0_raw_provenance"]["files"].items()
        if e["sha256"] is None
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true",
                    help="write configs/p0_freeze.json from the current tree")
    args = ap.parse_args()

    if args.register:
        if FREEZE_PATH.exists():
            problems = check()
            if problems:
                print("Refusing to re-register while P0 differs from the "
                      "existing registry. Resolve these first, or delete the "
                      "registry deliberately:\n")
                for p in problems:
                    print(f"  - {p}")
                return 2
        reg = build_registry()
        FREEZE_PATH.write_text(
            json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        n_strict = sum(1 for e in reg["strict_files"].values() if e["sha256"])
        n_raw = len(reg["p0_raw_provenance"]["files"])
        missing = unregistered_provenance(reg)
        print(f"wrote {FREEZE_PATH.relative_to(REPO_ROOT)}")
        print(f"  config P0 subtree : {reg['config_p0_subtree']['sha256']}")
        print(f"  strict files      : {n_strict}/{len(reg['strict_files'])} registered")
        print(f"  P0 raw provenance : {n_raw - len(missing)}/{n_raw} registered")
        if missing:
            print(f"\n  {len(missing)} P0 raw file(s) still MISSING (null, not "
                  f"fabricated), e.g.:")
            for rel in missing[:4]:
                print(f"    {rel}")
            print(f"  Restore the real Kaggle output into {P0_RAW_DIR}/ and "
                  f"re-run --register.")
        return 0

    problems = check()
    missing = unregistered_provenance()
    if problems:
        print("P0 FREEZE VIOLATED:\n")
        for p in problems:
            print(f"  - {p}")
        return 2
    print("P0 freeze intact: config subtree and every registered strict file match.")
    if missing:
        print(f"\nNOTE: {len(missing)} P0 raw provenance file(s) are still "
              f"unregistered (null). This is a pre-existing gap, not a P1 change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
