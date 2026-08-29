"""Generate the three Kaggle notebooks for the final P1 run.

The notebooks contain COMMANDS, not logic. Everything that defines the
experiment lives in configs/experiment.yaml and configs/p1_split_manifest.json,
and every cell below invokes a script that is under test in this repo. Nothing
is configured, patched or hand-edited inside Kaggle.

That is not a style preference. The one full P1 evaluation that has been run so
far was driven by ad-hoc notebook code, because `scripts/run_eval.py` had no way
to load a fine-tuned checkpoint; the ad-hoc code scored the base model and the
"fine-tuned" results were bit-identical to the Base arm. `run_eval.py` now takes
--local-checkpoint, so the FT arm has a sanctioned path and these notebooks use
it.

    python scripts/make_p1_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlang import config as cfg_mod  # noqa: E402
from quantlang.config import REPO_ROOT  # noqa: E402

REPO_URL = "https://github.com/fairuz-anadi/quantization.git"
OUT_DIR = REPO_ROOT / "notebooks"


# A `!command` NEVER raises in Jupyter, so a failing gate scrolls past and Run
# All carries straight on into the expensive cells. That happened once: twelve
# failed tests went by and the session continued into a 6 GB model download.
# Gate cells go through this helper instead, which raises and stops the notebook.
GATE = (
    "import subprocess, sys\n"
    "def gate(*cmd):\n"
    "    print('$', ' '.join(cmd), flush=True)\n"
    "    if subprocess.run([sys.executable, *cmd]).returncode != 0:\n"
    "        raise SystemExit('GATE FAILED: ' + ' '.join(cmd) + '. Stop here -- '\n"
    "                         'the design is not adjusted to make a check pass.')\n"
)

SETUP = [
    ("markdown", """# {title}

{intro}

**Settings: Accelerator `GPU T4 x2`, Internet `ON`.**

Nothing in this notebook configures the experiment. Every cell runs a script
from the repo; the design lives in `configs/experiment.yaml` and
`configs/p1_split_manifest.json`. If a check fails, stop and report it -- the
design is not adjusted to make a check pass."""),
    ("code", '''# 1. Get the code.
REPO_URL = "{repo}"
REF      = "main"          # branch, tag or commit SHA -- all three work

import os, subprocess, sys
SRC = "/kaggle/working/quantlang"
if not os.path.exists(SRC):
    # Full clone, then checkout. `-b` takes branch and tag names ONLY, so a
    # commit SHA there fails with exit 128 -- and `--depth 1` fetches just the
    # branch tip, which would not contain the SHA even if -b accepted one.
    subprocess.run(["git", "clone", REPO_URL, SRC], check=True)
    subprocess.run(["git", "-C", SRC, "checkout", "--quiet", REF], check=True)
print(subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip())
os.chdir(SRC); sys.path.insert(0, SRC)'''),
    ("code", '# 2. Dependencies. Kaggle\'s torch is CUDA-matched -- never reinstall it.\n'
             '!pip install -q -U "transformers>=4.45" "bitsandbytes>=0.43" "peft>=0.13" '
             'accelerate datasets pyyaml\n'
             '\n'
             '# torchao is REMOVED, not upgraded. Kaggle ships torchao 0.10.0; a\n'
             '# current PEFT wants >= 0.16.0, and its is_torchao_available() RAISES\n'
             '# on an out-of-range version instead of returning False. PEFT probes it\n'
             '# for every LoRA layer it builds, so with both installed no adapter can\n'
             '# attach at all and fine-tuning cannot run.\n'
             '#\n'
             '# This pipeline never uses torchao -- INT8 and NF4 are both bitsandbytes.\n'
             '# Upgrading it instead could pull a different torch, which is the one\n'
             '# thing on Kaggle that must not move.\n'
             '!pip uninstall -q -y torchao'),
    ("code", "# 3. Environment probe. Raises and STOPS the notebook if this "
             "session cannot\n#    run the experiment: a P100 (no INT8/NF4), or "
             "a PEFT/torchao mismatch\n#    that makes LoRA attachment "
             "impossible.\n" + GATE +
             '\ngate("scripts/probe_env.py", "--outdir", "/kaggle/working")'),
]


def nb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {"cells": out,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"},
                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


def session_a(langs: list[str]) -> dict:
    lang_args = " ".join(langs)
    cells = [(k, v.format(title="P1 session A - validate before spending anything",
                          intro=(
                              "Everything here is cheap. It exists so that a "
                              "session B or C is never started on a corpus or a "
                              "task that cannot support the paper.\\n\\n"
                              "Run this first, read the last cell, and only "
                              "continue if it says the gate passed."),
                          repo=REPO_URL))
             for k, v in SETUP]
    cells += [
        ("markdown", """## The frozen contracts

`freeze_p0.py` proves P0 is byte-identical to what produced the published
results. `pytest` covers the P1 construction, including the checks that version
1 did not have: the substring shortcut is worth exactly a guess, the gold and
the distractors are present at the same rate, and the FT arm is verified to
differ from the Base arm."""),
        ("code", GATE + '\ngate("scripts/freeze_p0.py")\ngate("-m", "pytest", "-q")'),
        ("markdown", """## The corpus reproduces exactly

Re-derives every P1 item from the pinned dataset revision, the frozen
`split_seed` and the pinned tokenizer, and compares against the frozen digests.
Downloads the corpus, so it takes a few minutes."""),
        ("code", GATE + '\ngate("scripts/build_p1_splits.py", "--check")'),
        ("markdown", """## The learnability gate

Two questions, cheapest first.

1. **CPU.** Does the frozen item set still carry a lexical shortcut? Version 1
   scored ~0.96 (English) and ~0.92 (Bangla) on "choose the option that appears
   verbatim in the passage", on a 100% gold-presence rate. Version 2 scores
   exactly 0.25 because all four options are present.
2. **GPU, a few minutes.** How well does the BASE model already do on the P1
   task, scored by the exact P0 evaluator? The threshold is P0's own best
   measured cell, read from `results/ALL_P0_RESULTS/tables/accuracy.csv` -- it
   is not a number chosen here.

   Measured 2026-08-29 on 300 items drawn ONE PER ARTICLE: **0.9567 English
   [0.9273, 0.9745], 0.8500 Bangla [0.8052, 0.8860]**, so English trips it and
   Bangla's whole interval sits below the 0.8956 ceiling. An earlier reading of
   0.970 / 0.900 came from `items[:300]`, which was 300 questions over the 38
   alphabetically-first articles -- clustered, and optimistic on both counts.
   This is a WARNING, passed with `--acknowledge-low-headroom` and recorded in
   the report. It bounds what the FT arm can show on ACCURACY -- RQ3 is expected
   to be null for English -- but it does not make the FT arm vacuous: check 9
   below measures the FT model against the base at matched precision and found a
   1.156 logit delta from three optimizer steps, against 0.000000 for the
   invalid v1 run. Empty headroom and an unchanged model are different claims.
   Borne out by the full runs: English cut its training loss 0.3396 -> 0.1168
   and Bangla 0.4352 -> 0.2099, while neither gained accuracy on BELEBELE."""),
        ("code", GATE + f'\ngate("scripts/check_p1_learnability.py", *"--langs {lang_args}".split(), "--outdir", "/kaggle/working/p1_gate", "--acknowledge-low-headroom")'),
        ("markdown", """## The 20-item smoke test

Nine checks. The ninth is new: it compares the fine-tuned logits against the
base model's. A full English run once passed checks 1-8 while producing logits
bit-identical to the base model at every precision, because nothing compared the
two arms.

Read the **fp16** row of `max_logit_delta_vs_base_fp16`: the baseline is the base
model at FP16, so only that row isolates fine-tuning. The int8 and nf4 rows carry
the quantization effect too and rise for that reason alone."""),
        ("code", GATE + '\ngate("scripts/run_p1_smoke.py", "--outdir", "/kaggle/working/p1_smoke", "--lang", "eng_Latn")'),
        ("code", '''# Read the report.
import json
r = json.load(open("/kaggle/working/p1_smoke/p1_smoke_report.json", encoding="utf-8"))
for name, check in sorted(r["checks"].items()):
    print(f"[{'PASS' if check['pass'] else 'FAIL'}] {name}")

d = r["checks"]["9_ft_arm_differs_from_base_arm"]
print("\\nFT vs base, max logit delta (fp16 row is the comparable one):")
print("   ", d["max_logit_delta_vs_base_fp16"])
print("merge weight delta:", d["merge_weight_delta"])
print("\\nALL CHECKS PASSED:", r["all_checks_passed"])'''),
        ("markdown", """## Gate

Continue to session B only if:

* `freeze_p0.py` reports the P0 freeze intact (the 30 unregistered raw
  provenance files are a known pre-existing gap, not a failure);
* `pytest` is green;
* `--check` reports the split reproduces exactly;
* the learnability gate prints `GATE PASSED`;
* all nine smoke checks read PASS, and check 9's logit delta is **not** 0.0.

Then download `/kaggle/working/p1_gate/p1_learnability_report.json` and
`/kaggle/working/p1_smoke/p1_smoke_report.json`."""),
        ("code", '''import shutil, os
KEEP = "/kaggle/working/p1_sessionA_keep"
os.makedirs(KEEP, exist_ok=True)
for src in ["/kaggle/working/p1_gate/p1_learnability_report.json",
            "/kaggle/working/p1_smoke/p1_smoke_report.json"]:
    if os.path.exists(src):
        shutil.copy(src, KEEP)
if os.path.isdir("/kaggle/working/p1_smoke/adapter"):
    shutil.copytree("/kaggle/working/p1_smoke/adapter", f"{KEEP}/adapter",
                    dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/p1_sessionA", "zip", KEEP)

# The merged checkpoint is ~5.75 GB and is DERIVED -- rebuildable from the base
# model plus the adapter. Freeing it keeps the session inside the 20 GB limit.
if os.path.isdir("/kaggle/working/p1_smoke/merged"):
    shutil.rmtree("/kaggle/working/p1_smoke/merged")
print(sorted(os.listdir("/kaggle/working")))'''),
    ]
    return nb(cells)


def session_lang(lang: str, letter: str) -> dict:
    cells = [(k, v.format(
        title=f"P1 session {letter} - {lang}",
        intro=(
            f"Fine-tunes {lang}, merges the adapter into an FP16 checkpoint, and "
            f"evaluates that checkpoint on BELEBELE at all three precisions.\\n\\n"
            f"**Do not run this before session A passes.**\\n\\n"
            f"Only the three FT cells are produced here. The three Base cells for "
            f"{lang} were measured in P0 and are already in "
            f"`results/ALL_P0_RESULTS/tables/accuracy.csv`; re-running them would "
            f"add nothing and would break the rule that a result comes from "
            f"exactly one run."),
        repo=REPO_URL)) for k, v in SETUP]
    cells += [
        ("code", "# 4. The contracts still hold in THIS session.\n" + GATE +
         '\ngate("scripts/freeze_p0.py")\ngate("-m", "pytest", "-q")'),
        ("markdown", f"""## Fine-tune {lang}

One epoch of LoRA on the FP16 base, then `merge_and_unload` into a plain FP16
checkpoint. The merge is now verified to have moved the weights -- if it has
not, this cell fails rather than shipping a checkpoint that is the base model.

The training partition is trimmed to the size shared by both final-scope
languages, so each arm takes the same number of gradient steps."""),
        ("code", GATE + f'\ngate("scripts/run_finetune.py", "--lang", "{lang}", "--outdir", "/kaggle/working/p1", "--tag", "main")'),
        ("markdown", f"""## Evaluate the FT arm

Three cells: {lang} x FP16 / INT8 / NF4, on the merged checkpoint, through the
same evaluator and the same frozen 900-item BELEBELE manifest that produced P0.

`--local-checkpoint` is what makes this the sanctioned path. `--ft-lang` sets
the result alias to `qwen2.5-3b-instruct-ft-{lang}`, so an FT cell can never
collide with or be mistaken for a Base cell, and every row records
`weights_from` and `arm`."""),
        # One physical line: a `!` magic does not honour backslash continuation.
        ("code", f'''SEED = __import__("yaml").safe_load(
    open("configs/experiment.yaml", encoding="utf-8"))["finetune"]["seeds"]["main"]
MERGED = f"/kaggle/working/p1/merged/{lang}__seed{{SEED}}"
print(MERGED)

!python scripts/run_eval.py --all-precisions --langs {lang} --local-checkpoint {{MERGED}} --ft-lang {lang} --outdir /kaggle/working/p1/results --tag main'''),
        ("markdown", """## Confirm the FT arm is not the Base arm

Read straight off the written cells. `arm` must say `finetuned` and
`weights_from` must be the merged checkpoint path, not `hub`."""),
        ("code", f'''import glob, json
for path in sorted(glob.glob("/kaggle/working/p1/results/*.meta.json")):
    m = json.load(open(path, encoding="utf-8"))
    print(f"{{m['precision']:<14}} {{m['lang']}}  acc={{m['accuracy']:.4f}}  "
          f"arm={{m['arm']}}  alias={{m['model_alias']}}")
    print(f"               weights_from={{m['weights_from']}}")
    assert m["arm"] == "finetuned", "this cell scored the BASE model"
    assert m["weights_from"] != "hub"
    assert m["n_items"] == 900, m["n_items"]
print("\\nall FT cells scored the merged checkpoint on the full 900 items")'''),
        ("markdown", """## Package

The merged FP16 checkpoint is ~5.75 GB and DERIVED -- rebuildable from the base
model plus the adapter -- so it is not shipped. The adapter, the training
metadata and every raw result are."""),
        ("code", f'''import os, shutil
KEEP = "/kaggle/working/p1_{lang}_keep"
os.makedirs(KEEP, exist_ok=True)
shutil.copytree("/kaggle/working/p1/results", f"{{KEEP}}/results", dirs_exist_ok=True)
for d in ["adapters"]:
    if os.path.isdir(f"/kaggle/working/p1/{{d}}"):
        shutil.copytree(f"/kaggle/working/p1/{{d}}", f"{{KEEP}}/{{d}}", dirs_exist_ok=True)
for f in os.listdir("/kaggle/working/p1"):
    if f.endswith("__finetune.json"):
        shutil.copy(f"/kaggle/working/p1/{{f}}", KEEP)
shutil.copy("configs/p1_split_manifest.json", KEEP)

shutil.make_archive("/kaggle/working/p1_{lang}", "zip", KEEP)
if os.path.isdir("/kaggle/working/p1/merged"):
    shutil.rmtree("/kaggle/working/p1/merged")
    print("removed the merged checkpoint (derived, rebuildable)")
print(sorted(os.listdir("/kaggle/working")))'''),
        ("markdown", f"""## Then, locally

```bash
kaggle kernels output <user>/<kernel-slug> -p results/raw/
```

`results/raw/` is append-only and is written by exactly that command. Nothing in
the repo writes to it, and `tests/test_raw_is_append_only.py` fails if something
starts to."""),
    ]
    return nb(cells)


def main() -> int:
    cfg = cfg_mod.load()
    langs = cfg_mod.require(cfg, "finetune.final_scope_languages")

    written = [(OUT_DIR / "kaggle_p1_A_validate.ipynb", session_a(langs))]
    for letter, lang in zip("BCDEF", langs):
        written.append(
            (OUT_DIR / f"kaggle_p1_{letter}_{lang}.ipynb",
             session_lang(lang, letter)))

    for path, doc in written:
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
