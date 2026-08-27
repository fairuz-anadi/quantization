"""Turn raw run output into the validated tidy frame the analysis reads.

Raw files are read, never written. If any measured cell is short of the frozen
900 items, this script refuses to emit tidy.csv and tells you which cell to
rerun -- it does not drop the cell, trim the others to match, or report a
partial accuracy.

    python scripts/build_tidy.py                     # from results/raw
    python scripts/build_tidy.py --inventory         # what exists so far
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang.config import REPO_ROOT  # noqa: E402
from quantlang.tidy import build_latency, build_tidy, cell_inventory  # noqa: E402

DEFAULT_IN = REPO_ROOT / "results" / "raw"
DEFAULT_OUT = REPO_ROOT / "results" / "tables"


def _rel(p: Path) -> str:
    """Display path relative to the repo when possible; absolute otherwise."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(DEFAULT_IN))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--inventory", action="store_true",
                    help="report which cells exist and stop; validates nothing")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    inv = cell_inventory(indir)
    print(inv.to_string(index=False))
    incomplete = inv[~inv["complete"]]
    if len(incomplete):
        print(f"\n{len(incomplete)} incomplete cell(s):")
        for _, r in incomplete.iterrows():
            print(f"  {r['model']} / {r['lang']} / {r['precision']}: "
                  f"{r['n_items']}/{r['expected']}")

    if args.inventory:
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    df = build_tidy(indir)
    tidy_path = outdir / "tidy.csv"
    df.to_csv(tidy_path, index=False)
    print(f"\nwrote {_rel(tidy_path)}  ({len(df)} rows)")

    lat = build_latency(indir)
    lat_path = outdir / "latency.csv"
    lat.to_csv(lat_path, index=False)
    print(f"wrote {_rel(lat_path)}  ({len(lat)} runs)")

    gpus = sorted({g for g in lat["gpu_name"].dropna().unique()})
    if len(gpus) > 1:
        print(
            f"\nWARNING: latency rows span more than one GPU type ({gpus}). "
            f"Latency is only comparable within a session on one GPU; do not "
            f"put these in one table without saying so."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
