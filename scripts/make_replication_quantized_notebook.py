"""Phase 2 for a replication model: INT8 and NF4, gate-passing languages only.

FP16 was measured by the gate run and is not re-run here.

`--floored` is REQUIRED to be stated explicitly rather than inferred. A language
excluded from the quantized pass must be excluded because its FP16 Wilson lower
bound failed the pre-registered 0.30 threshold -- never because its result was
inconvenient -- and the notebook records which languages were dropped and why,
so the exclusion is auditable from the artefact itself.

    python scripts/make_replication_quantized_notebook.py --alias bloomz-3b \\
        --langs eng_Latn ben_Beng asm_Beng npi_Deva --floored sin_Sinh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang.config import REPO_ROOT  # noqa: E402

REPO_URL = "https://github.com/fairuz-anadi/quantization.git"
OUT_DIR = REPO_ROOT / "notebooks"

GATE = (
    "import subprocess, sys\n"
    "def gate(*cmd):\n"
    "    print('$', ' '.join(cmd), flush=True)\n"
    "    if subprocess.run([sys.executable, *cmd]).returncode != 0:\n"
    "        raise SystemExit('GATE FAILED: ' + ' '.join(cmd) + '. Stop here -- '\n"
    "                         'the design is not adjusted to make a check pass.')\n"
)


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {},
             "source": src.splitlines(keepends=True)}
        if kind == "code":
            c["execution_count"] = None
            c["outputs"] = []
        out.append(c)
    return {"cells": out,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


CLONE = '''# 1. Get the code.
REPO_URL = "{repo}"
REF      = "main"

import os, subprocess, sys
SRC = "/kaggle/working/quantlang"
if not os.path.exists(SRC):
    subprocess.run(["git", "clone", REPO_URL, SRC], check=True)
    subprocess.run(["git", "-C", SRC, "checkout", "--quiet", REF], check=True)
print(subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip())
os.chdir(SRC); sys.path.insert(0, SRC)'''

AUTH = """# 1b. Hugging Face auth, for gated repositories.
#
# The FP16 gate notebook has always had this. The quantized notebook did NOT,
# so a gated model cleared the gate and then died on a 401 in phase 2 -- the
# same model, the same session settings, a different notebook. Both generators
# emit it now.
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as exc:
    print(f"no HF_TOKEN secret ({type(exc).__name__}). Fine for an open model; "
          f"a gated one will fail below with a 401.")"""


DEPS = ('!pip install -q -U "transformers>=4.45" "bitsandbytes>=0.43" '
        '"peft>=0.13" accelerate datasets pyyaml\n'
        '!pip uninstall -q -y torchao')

REPORT = '''import glob, json
rows = []
for p in sorted(glob.glob("/kaggle/working/rep/results/*.meta.json")):
    m = json.load(open(p, encoding="utf-8"))
    assert m["arm"] == "base", "a replication cell must be a base cell"
    assert m["n_items"] == 900, m["n_items"]
    rows.append(m)

print(f"{"lang":10}{"precision":16}{"acc":>8}{"items":>7}{"tok":>7}{"trunc":>7}")
for m in sorted(rows, key=lambda m: (m["lang"], m["precision"])):
    print(f"{m['lang']:10}{m['precision']:16}{m['accuracy']:8.4f}"
          f"{m['n_items']:7d}{m['median_input_tokens']:7.0f}{m['n_truncated']:7d}")
print("\\nBring these back with the FP16 gate numbers for the paired analysis.")'''

PACK = '''import os, shutil
KEEP = "/kaggle/working/repq_{alias}_keep"
os.makedirs(KEEP, exist_ok=True)
shutil.copytree("/kaggle/working/rep/results", KEEP + "/results", dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/repq_{alias}", "zip", KEEP)
print(sorted(os.listdir("/kaggle/working")))'''


def build(alias: str, entry: dict, langs: list[str], floored: list[str],
          precisions: list[str]) -> dict:
    lang_args = ",\n         ".join(repr(l) for l in langs)
    prec_label = " and ".join(p.replace("int8_llmint8", "INT8").replace("nf4", "NF4")
                              for p in precisions)
    excl = ""
    if floored:
        excl = (
            "\n\n**Excluded: `" + "`, `".join(floored) + "`** -- floored at the "
            "FP16 gate. A model at chance cannot exhibit quantization "
            "degradation, so a small delta there would mean *already broken*, "
            "not *robust*. Per `docs/H5_PREREGISTRATION.md` these cells are "
            "reported as floored and carry no degradation claim in either "
            "direction.")

    cells = [
        ("markdown", f"""# Replication quantized pass - {alias} ({prec_label})

{prec_label} on **{entry['hf_id']}** @ `{entry['revision'][:12]}`, on the
languages that cleared the FP16 floor gate. FP16 is NOT re-run; those cells
already exist from the gate run.{excl}

Tests pre-registered prediction 1: languages this model tokenizes cheaply
should show smaller FP16-to-NF4 degradation than Qwen showed for them.

**Settings: Accelerator `GPU T4 x2`, Internet `ON`.**"""),
        ("code", CLONE.format(repo=REPO_URL)),
        ("code", AUTH),
        ("code", DEPS),
        ("code", GATE + '\ngate("scripts/probe_env.py", "--outdir", "/kaggle/working")'),
        ("markdown", "## P0 is still frozen, and the suite still passes"),
        ("code", GATE + '\ngate("scripts/freeze_p0.py")\ngate("-m", "pytest", "-q")'),
        ("markdown", f"""## {prec_label}

Same evaluator, same frozen 900-item manifest, same `letter_logit` scoring as
P0 and as the FP16 gate run. Languages: `{'`, `'.join(langs)}`."""),
        ("code", GATE + f'''
for prec in {precisions!r}:
    gate("scripts/run_eval.py", "--precision", prec,
         "--langs",
         {lang_args},
         "--model-alias", "{alias}",
         "--outdir", "/kaggle/working/rep/results",
         "--tag", "repq")'''),
        ("markdown", "## Cells written"),
        ("code", REPORT),
        ("code", PACK.format(alias=alias)),
    ]
    return nb(cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True)
    ap.add_argument("--langs", nargs="+", required=True,
                    help="languages that PASSED the FP16 floor gate")
    ap.add_argument("--precisions", nargs="+", default=["int8_llmint8", "nf4"],
                    choices=["int8_llmint8", "nf4"],
                    help="which quantized precisions to run. H5 concerns FP16 "
                         "-> NF4 only, so --precisions nf4 is the cheap path "
                         "when INT8 is not needed for table symmetry.")
    ap.add_argument("--floored", nargs="*", default=[],
                    help="languages excluded by the gate; recorded in the notebook")
    args = ap.parse_args()

    cfg = cfg_mod.load()
    rep = cfg.get("replication_models") or []
    match = [m for m in rep if m["alias"] == args.alias]
    if not match:
        raise SystemExit(f"FATAL: {args.alias!r} is not in replication_models. "
                         f"Available: {[m['alias'] for m in rep]}")
    entry = match[0]
    if not entry.get("revision"):
        raise SystemExit(f"FATAL: {args.alias} has no pinned revision.")

    known = cfg_mod.require(cfg, "benchmark.languages")
    unknown = [l for l in list(args.langs) + list(args.floored) if l not in known]
    if unknown:
        raise SystemExit(f"FATAL: {unknown} not in the frozen language set {known}")
    overlap = set(args.langs) & set(args.floored)
    if overlap:
        raise SystemExit(f"FATAL: {sorted(overlap)} cannot both pass and be floored")

    path = OUT_DIR / f"kaggle_rep_{args.alias}_{'_'.join(args.precisions)}.ipynb"
    path.write_text(json.dumps(build(args.alias, entry, args.langs,
                                     args.floored, args.precisions),
                               indent=1), encoding="utf-8")
    print(f"wrote {path}")
    print(f"  model     : {entry['hf_id']} @ {entry['revision'][:12]}")
    print(f"  precisions: {', '.join(args.precisions)}")
    print(f"  languages : {args.langs}")
    if args.floored:
        print(f"  FLOORED   : {args.floored} -- excluded, and recorded in the "
              f"notebook so the exclusion is auditable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
