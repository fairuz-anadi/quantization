"""Run FIRST in every Kaggle session, before anything else.

Answers three questions that decide whether the session is usable at all:

  * Can bitsandbytes INT8 and NF4 run here? Both need compute capability >= 7.5.
    A P100 is sm_60 and cannot run two of the three precisions, so a session
    that lands on one must be restarted rather than partially used.
  * Is genuine FP8 W8A8 possible? That needs sm_89+. On T4 it is not, which is
    why the paper reports FP16/INT8/NF4 and treats FP8 as future work. This is
    checked rather than assumed, so the limitation is evidenced.
  * What exactly is this environment? Recorded for the reproducibility log.

Exits non-zero if the session cannot run the experiment.

    python scripts/probe_env.py --outdir /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

MIN_CAP_BNB = 75    # Turing: bitsandbytes INT8 + NF4
MIN_CAP_FP8 = 89    # Ada/Hopper: genuine FP8 W8A8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    out: dict = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if not torch.cuda.is_available():
        out.update(ok_for_experiment=False, reason="no CUDA device visible")
        print(json.dumps(out, indent=2))
        return 2

    caps = []
    out["gpus"] = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        caps.append(major * 10 + minor)
        props = torch.cuda.get_device_properties(i)
        out["gpus"].append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "compute_capability": f"{major}.{minor}",
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
        })
    min_cap = min(caps)
    out["n_gpus"] = len(caps)
    out["min_compute_capability"] = min_cap / 10
    out["bnb_int8_supported"] = min_cap >= MIN_CAP_BNB
    out["bnb_nf4_supported"] = min_cap >= MIN_CAP_BNB
    out["fp8_w8a8_supported"] = min_cap >= MIN_CAP_FP8
    out["bf16_supported"] = torch.cuda.is_bf16_supported()

    for lib in ("transformers", "bitsandbytes", "accelerate", "datasets", "peft"):
        try:
            out[f"{lib}_version"] = getattr(__import__(lib), "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            out[f"{lib}_version"] = f"NOT INSTALLED ({type(exc).__name__})"

    # Does bitsandbytes actually build a working 4-bit linear on THIS box?
    # Version strings do not prove the kernels load.
    try:
        import bitsandbytes as bnb
        lin = bnb.nn.Linear4bit(64, 64, quant_type="nf4",
                                compute_dtype=torch.float16).cuda()
        y = lin(torch.randn(2, 64, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
        out["bnb_nf4_smoke_test"] = "pass" if y.shape == (2, 64) else f"bad shape {tuple(y.shape)}"
    except Exception as exc:  # noqa: BLE001
        out["bnb_nf4_smoke_test"] = f"fail: {type(exc).__name__}: {exc}"

    try:
        out["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception:  # noqa: BLE001
        out["nvidia_smi"] = "unavailable"

    out["ok_for_experiment"] = bool(
        out["bnb_int8_supported"] and out["bnb_nf4_supported"]
        and out["bnb_nf4_smoke_test"] == "pass"
    )
    if not out["ok_for_experiment"]:
        out["reason"] = (
            f"compute capability {min_cap / 10} (need >= {MIN_CAP_BNB / 10}) or the "
            f"bitsandbytes NF4 smoke test failed. Set the Kaggle accelerator to "
            f"T4 x2 and restart the session. Do NOT run on a P100."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"env_probe_{int(time.time())}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print(f"\nwritten -> {path}")

    if not out["ok_for_experiment"]:
        print("\n*** STOP. This session cannot run the experiment. ***")
        return 2
    if out["fp8_w8a8_supported"]:
        print("\n*** This GPU DOES support FP8 W8A8. The FP8 decision can be reopened. ***")
    else:
        print("\nFP8 W8A8 unavailable on this hardware (needs sm_8.9+). "
              "Reporting FP16/INT8/NF4; FP8 stays future work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
