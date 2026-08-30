"""Generate the Kaggle FP16 gate notebook for one replication model.

FP16 ONLY, and deliberately so. A model at or near chance in a language cannot
exhibit quantization degradation -- there is nothing left to lose, and a small
delta then means "already broken" rather than "robust". Spending GPU on INT8 and
NF4 for such a language buys an uninterpretable number.

So the protocol is: FP16 across all five languages first (~a third of the cost),
apply the floor gate fixed in docs/H5_PREREGISTRATION.md, and only then decide.
This notebook stops at the gate. It does not launch INT8 or NF4.

    python scripts/make_replication_notebook.py --alias bloomz-3b
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
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
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


def build(alias: str, entry: dict, langs: list[str]) -> dict:
    langs_arg = " ".join(langs)
    cells = [
        ("markdown", f"""# Replication FP16 gate - {alias}

Evaluates **{entry['hf_id']}** at **FP16 only**, on the same frozen 900-item
BELEBELE manifest, the same prompt template and the same `letter_logit` scoring
that produced P0.

**This notebook does not run INT8 or NF4.** A model at chance in a language
cannot show quantization degradation, so the floor gate in
`docs/H5_PREREGISTRATION.md` is applied to FP16 first. Bring the results back
before any quantized cell is authorised.

P0 is untouched by this notebook. `{alias}` is resolved from the
`replication_models` config key, which sits outside the frozen P0 subtree; the
freeze gate below proves that.

**Settings: Accelerator `GPU T4 x2`, Internet `ON`.**"""),

        ("code", f'''# 1. Get the code.
REPO_URL = "{REPO_URL}"
REF      = "main"          # branch, tag or commit SHA -- all three work

import os, subprocess, sys
SRC = "/kaggle/working/quantlang"
if not os.path.exists(SRC):
    subprocess.run(["git", "clone", REPO_URL, SRC], check=True)
    subprocess.run(["git", "-C", SRC, "checkout", "--quiet", REF], check=True)
print(subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip())
os.chdir(SRC); sys.path.insert(0, SRC)'''),

        ("code", '# 2. Dependencies. Kaggle\'s torch is CUDA-matched -- never reinstall it.\n'
                 '!pip install -q -U "transformers>=4.45" "bitsandbytes>=0.43" "peft>=0.13" '
                 'accelerate datasets pyyaml\n'
                 '!pip uninstall -q -y torchao'),

        ("code", "# 3. Environment probe.\n" + GATE +
         '\ngate("scripts/probe_env.py", "--outdir", "/kaggle/working")'),

        ("markdown", """## P0 is still frozen

This is the check that matters for a replication run: adding a model must not
have disturbed the results P0 already produced. `replication_models` is a new
top-level key precisely so this passes."""),

        ("code", "# 4. Contracts.\n" + GATE +
         '\ngate("scripts/freeze_p0.py")\ngate("-m", "pytest", "-q")'),

        ("markdown", f"""## Tokenizer pre-flight

Decided from the tokenizer alone, before any weights download. `letter_logit`
reads the logit of `" A"`..`" D"` and requires each to be exactly one distinct
token; a model that fails this cannot be scored by the P0 procedure at all, and
the answer is a different model -- never a different scoring method, which would
make the numbers incomparable with P0.

Also reports tokens per item per language on the frozen items, which is the
quantity H5 is about."""),

        ("code", GATE + f'\ngate("scripts/probe_model_compat.py", "--hf-id", '
                        f'"{entry["hf_id"]}", "--revision", "{entry["revision"]}", '
                        f'"--outdir", "/kaggle/working/rep")'),

        ("markdown", f"""## FP16 across all five languages

Same evaluator, same frozen manifest, same scoring as P0. `--model-alias
{alias}` resolves from `replication_models`; P0's own selection path is
untouched.

Roughly 35-50 minutes -- Sinhala dominates, at about 7x the tokens per item of
the other four under this tokenizer."""),

        ("code", GATE + f'\ngate("scripts/run_eval.py", "--precision", "fp16",\n'
                        f'     "--langs", {", ".join(repr(l) for l in langs)},\n'
                        f'     "--model-alias", "{alias}",\n'
                        f'     "--outdir", "/kaggle/working/rep/results", "--tag", "repfp16")'),

        ("markdown", """## The FP16 gate report

The floor gate is fixed in `docs/H5_PREREGISTRATION.md`: a language is
interpretable for this model when its **Wilson 95% lower bound exceeds 0.30**,
against a four-option chance level of 0.25. It is applied uniformly. A floored
language is reported as floored, not dropped."""),

        ("code", f'''import glob, json, math
rows = []
for p in sorted(glob.glob("/kaggle/working/rep/results/*.meta.json")):
    m = json.load(open(p, encoding="utf-8"))
    n, k = m["n_items"], m["n_correct"]
    acc = k / n
    z = 1.959963985
    den = 1 + z*z/n
    centre = (acc + z*z/(2*n)) / den
    half = z*math.sqrt(acc*(1-acc)/n + z*z/(4*n*n)) / den
    lo, hi = centre - half, centre + half
    rows.append(dict(lang=m["lang"], n=n, correct=k, acc=acc, lo=lo, hi=hi,
                     tokens=m["median_input_tokens"], trunc=m["n_truncated"],
                     alias=m["model_alias"], arm=m["arm"],
                     passes=lo > 0.30))

print(f"model: {{rows[0]['alias'] if rows else '?'}}   chance = 0.25   "
      f"gate: Wilson lower bound > 0.30\\n")
print(f"{{'lang':10}}{{'acc':>8}}{{'95% CI':>18}}{{'items':>8}}{{'tok/item':>10}}"
      f"{{'trunc':>7}}   gate")
for r in sorted(rows, key=lambda r: r["tokens"]):
    print(f"{{r['lang']:10}}{{r['acc']:8.4f}}"
          f"{{'[' + format(r['lo'], '.3f') + ', ' + format(r['hi'], '.3f') + ']':>18}}"
          f"{{r['n']:8d}}{{r['tokens']:10.0f}}{{r['trunc']:7d}}   "
          f"{{'PASS' if r['passes'] else 'FLOORED'}}")

bad = [r for r in rows if not r["passes"]]
print()
if bad:
    print(f"FLOORED: {{[r['lang'] for r in bad]}} -- no degradation claim will be "
          f"made for these cells. They are reported as floored, not omitted.")
else:
    print("All five languages clear the floor gate.")
assert all(r["arm"] == "base" for r in rows), "a replication cell must be a base cell"
assert all(r["n"] == 900 for r in rows), "every cell must be the full 900 items"
json.dump(rows, open("/kaggle/working/rep/fp16_gate.json", "w"), indent=2)
print("\\nwrote /kaggle/working/rep/fp16_gate.json")
print("\\nSTOP HERE. Bring these numbers back before any INT8 or NF4 cell is run.")'''),

        ("code", f'''import os, shutil
KEEP = "/kaggle/working/rep_{alias}_keep"
os.makedirs(KEEP, exist_ok=True)
shutil.copytree("/kaggle/working/rep/results", f"{{KEEP}}/results", dirs_exist_ok=True)
for f in ["fp16_gate.json"]:
    if os.path.exists(f"/kaggle/working/rep/{{f}}"):
        shutil.copy(f"/kaggle/working/rep/{{f}}", KEEP)
for f in os.listdir("/kaggle/working/rep"):
    if f.startswith("model_compat_"):
        shutil.copy(f"/kaggle/working/rep/{{f}}", KEEP)
shutil.make_archive("/kaggle/working/rep_{alias}", "zip", KEEP)
print(sorted(os.listdir("/kaggle/working")))'''),
    ]
    return nb(cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True)
    args = ap.parse_args()

    cfg = cfg_mod.load()
    rep = cfg.get("replication_models") or []
    match = [m for m in rep if m["alias"] == args.alias]
    if not match:
        raise SystemExit(
            f"FATAL: {args.alias!r} is not in replication_models. "
            f"Available: {[m['alias'] for m in rep]}")
    entry = match[0]
    if not entry.get("revision"):
        raise SystemExit(
            f"FATAL: {args.alias} has no pinned revision. An unpinned run is "
            f"not reproducible and its numbers cannot go in the paper.")

    langs = cfg_mod.require(cfg, "benchmark.languages")
    path = OUT_DIR / f"kaggle_rep_{args.alias}_fp16.ipynb"
    path.write_text(json.dumps(build(args.alias, entry, langs), indent=1),
                    encoding="utf-8")
    print(f"wrote {path}")
    print(f"  model    : {entry['hf_id']} @ {entry['revision'][:12]}")
    print(f"  languages: {langs}")
    print(f"  precision: fp16 ONLY -- the gate stops before INT8/NF4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
