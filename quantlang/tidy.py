"""Build tidy.csv from raw run output.

This is the only bridge between what a Kaggle run emits and what the analysis
reads, and it is deliberately unforgiving: `schema.validate_tidy` runs on the
result and any violation raises. An incomplete cell is never padded and never
partially reported -- it is reported as missing, and the run is repeated.

Latency lives in a separate frame because it is a per-run efficiency metric with
its own comparability rules (within-session, same GPU), not a per-item outcome
that belongs in the paired accuracy analysis.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .schema import TIDY_COLUMNS, SchemaError, load_manifest, validate_tidy


def read_records(indir: Path) -> list[dict]:
    """Read every per-item JSONL under `indir`. Raw files are only ever read."""
    records: list[dict] = []
    files = sorted(Path(indir).rglob("*.jsonl"))
    if not files:
        raise SchemaError(
            f"No .jsonl found under {indir}. Nothing has been measured yet; "
            f"there is no number to report."
        )
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SchemaError(
                        f"{path.name}:{lineno} is not valid JSON ({exc}). A "
                        f"truncated raw file means the run died mid-write; "
                        f"rerun that cell rather than salvaging the partial one."
                    ) from exc
    return records


def build_tidy(indir: Path, manifest: dict | None = None) -> pd.DataFrame:
    """Records -> validated tidy frame with exactly TIDY_COLUMNS."""
    manifest = manifest or load_manifest()
    records = read_records(indir)

    df = pd.DataFrame([{
        "model": r["model_alias"],
        "model_revision": r["model_revision"],
        "precision": r["precision"],
        "lang": r["lang"],
        "item_id": r["item_id"],
        "pred": int(r["pred"]),
        "gold": int(r["gold"]),
        "correct": int(r["correct"]),
    } for r in records])[list(TIDY_COLUMNS)]

    validate_tidy(df, manifest)
    return df


def cell_inventory(indir: Path) -> pd.DataFrame:
    """What has been measured so far, complete or not. Safe to call any time.

    Use this to see where a run stands without triggering the strict validation
    that build_tidy applies -- it reports shortfalls instead of raising on them.
    """
    manifest = load_manifest()
    expected = int(manifest["n_items"])
    counts: dict[tuple, set] = defaultdict(set)
    for r in read_records(indir):
        counts[(r["model_alias"], r["lang"], r["precision"])].add(r["item_id"])

    rows = [{
        "model": m, "lang": lang, "precision": prec,
        "n_items": len(ids), "expected": expected,
        "complete": len(ids) == expected,
    } for (m, lang, prec), ids in sorted(counts.items())]
    return pd.DataFrame(rows)


def build_latency(indir: Path) -> pd.DataFrame:
    """Per-run latency summary, read from the meta manifests.

    Session identity is carried through explicitly: latency is only comparable
    within one session on one GPU, so gpu_name and run timestamp travel with
    every row and any cross-session comparison has to be a deliberate act.
    """
    rows = []
    for path in sorted(Path(indir).rglob("*.meta.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": m["model_alias"],
            "model_revision": m["model_revision"],
            "precision": m["precision"],
            "lang": m["lang"],
            "tag": m["tag"],
            "n_items": m["n_items"],
            "median_latency_ms": m["median_latency_ms"],
            "mean_latency_ms": m["mean_latency_ms"],
            "p25_latency_ms": m.get("p25_latency_ms"),
            "p75_latency_ms": m.get("p75_latency_ms"),
            "median_input_tokens": m["median_input_tokens"],
            "peak_memory_allocated_gb": m["peak_memory_allocated_gb"],
            "peak_memory_reserved_gb": m["peak_memory_reserved_gb"],
            "repeats": m["repeats"],
            "warmup": m["warmup"],
            "gpu_name": m.get("env", {}).get("gpu_name"),
            "compute_capability": m.get("env", {}).get("compute_capability"),
            "timestamp_utc": m["timestamp_utc"],
        })
    if not rows:
        raise SchemaError(f"No .meta.json found under {indir}.")
    return pd.DataFrame(rows)
