# Is Quantization Language-Neutral?

Measuring whether bitsandbytes quantization degrades LLM accuracy more for
low-resource languages than for English, on BELEBELE.

## The rule

No number reaches the paper unless a script generated it from `results/tables/tidy.csv`,
which is built only from files in `results/raw/`, which arrive only via
`kaggle kernels output` from real Kaggle runs.

Unknown values are `null` in `configs/experiment.yaml` and every read goes
through `config.require()`, which raises. Nothing in this repo pads, resamples,
substitutes a default, or reports a partial cell.

## Data flow (one direction only)

    Kaggle run -> results/raw/ -> tidy.csv -> accuracy/degradation/interaction.csv -> paper/

## Frozen design

Locked 2026-08-28. Changing any of it is a scientific decision, not a config edit.

| | | |
|---|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e7253...` | ~6.2 GB in FP16, so all three precisions fit **one** T4. 7B would force FP16 to shard across 2 GPUs while NF4 ran on one, making the latency column a comparison of device topologies. |
| Languages | `eng_Latn` `ben_Beng` `sin_Sinh` `asm_Beng` `npi_Deva` | The brief named the first four. Nepali is carried **in addition** because Aya covers `ben`/`sin`/`eng`/`nep` but not `asm` — without it, one language could never enter the fine-tuning experiment. |
| Precisions | `fp16` `int8_llmint8` `nf4` | FP8 W8A8 needs sm_89; Kaggle gives T4 (sm_75). Verified by `scripts/probe_env.py`, not assumed. FP8 is future work, and no simulated FP8 timing is reported. |
| Scoring | `letter_logit` | One forward pass per item; argmax over the four option-letter token logits. No generation, so no parse failures — which matters because parse failures are language-correlated and would leak into the headline effect. |
| Eval set | full 900 BELEBELE items, untouched | Never split, never trained on, never filtered. |
| Reference | `eng_Latn` | The interaction term is defined relative to it. |

### Two things that are easy to get wrong

**Items are keyed, never positional.** BELEBELE's row order is not aligned
across languages. On the pinned revision, **zero** of Assamese's 900 rows share
an index with their English counterpart, and 580 of Bangla's differ. Pairing by
row position compares unrelated questions while every intermediate number looks
healthy. Everything joins on `link#question_number`, and
`tests/test_statistics.py` fails if positional pairing is reintroduced.

**The scored token is `" A"`, not `"A"`.** The prompt ends with `Answer:` and no
trailing space, so the token the model is actually about to emit is `" A"`
(id 362 for Qwen2.5), a different token from bare `"A"` (id 32). The loader
asserts each prefixed letter is exactly one token and refuses to run otherwise.

## Layout

    configs/    experiment.yaml, item_id_manifest.json, revisions.json
    quantlang/  config, data, model, evaluate, tidy, schema, statistics
    scripts/    probe_env, run_eval, build_tidy, analyze, pin_revisions, verify_item_ids
    notebooks/  kaggle_p0.ipynb -- the runner
    results/    raw/ (append-only) tables/ figures/ smoke/ (gitignored)
    tests/      the invariants, enforced
    paper/      LaTeX

## Running it

On Kaggle (**T4 x2**, internet on) — open `notebooks/kaggle_p0.ipynb` and follow it.
It probes the GPU, runs a 20-item smoke test across every cell, then the full grid.

Locally, once the raw output has been pulled down:

    python scripts/build_tidy.py --inventory   # what exists, complete or not
    python scripts/build_tidy.py               # validated tidy.csv + latency.csv
    python scripts/analyze.py                  # tables, stats, figures

`build_tidy` refuses to emit anything if a measured cell is short of 900 items.
It names the cell to rerun rather than trimming, dropping, or reporting a
partial accuracy.

## What is measured

Per item: `pred`, `gold`, `correct`, the four letter logits, forward-pass
latency, input token count, truncation flag, and a sha256 of the exact prompt.
The prompt text itself is reconstructible from the frozen template plus the
pinned dataset revision plus the item id, so the digest is complete provenance
at a fraction of the size.

Per run: GPU, compute capability, every library version, peak memory, quantized
layer counts, option token ids, prompt-template digest, and the manifest digest.

## Statistics

- **Wilson 95% CI** on every accuracy. Accuracies here sit near the 0.25 chance
  floor, where the normal approximation misbehaves.
- **Exact McNemar**, FP16 vs quantized within a language: is the drop real at all?
- **Paired bootstrap on the interaction** `d_int = d_lang - d_eng`: is the drop
  *larger* here than in English? This, not the raw accuracy table, is the
  paper's actual claim. Item indices are resampled once and reused across
  languages, which is only legitimate because every vector has been reindexed
  onto the same canonical item order first.
- **Holm** correction across contrasts. Where a per-contrast CI and the
  Holm-adjusted p disagree, the adjusted p is the claim that survives.

## Status

**P0 — complete and frozen.** The full 15-cell grid (5 languages × 3
precisions × 900 items) was run on Kaggle T4 and its tables are in
`results/ALL_P0_RESULTS/tables/`. The design is locked: changing it is a
scientific decision, not a config edit. `scripts/freeze_p0.py` digests the P0
subtree of `experiment.yaml` plus the frozen manifest, the pinned revisions,
the untouched P0 modules and the result tables; `tests/test_p1_freeze.py`
fails if any of them moves.

One gap, stated plainly: the per-item raw output of that run is **not** in the
repository — `results/ALL_P0_RESULTS/raw/` holds only its README, so `tidy.csv`
currently has no committed source. Those 30 files are registered in
`configs/p0_freeze.json` as `null`, meaning NOT YET KNOWN, and are never
fabricated. Restoring them and re-running `freeze_p0.py --register` pins them.

**P1 — rebuilt at algorithm_version 2, not yet executed.** Language-specific
LoRA fine-tuning × quantization, reduced to **English + Bangla** (a 2 × 2 × 3
grid: Base/FT × eng/ben × FP16/INT8/NF4). Six of those twelve cells — the Base
arm — are the P0 results above and need no further GPU time.

**Version 1 of the P1 corpus was invalid and every result built on it is
excluded.** It centred the passage window on the gold answer span and drew
distractors from *other* articles, so the gold was in the passage 100% of the
time and a distractor essentially never was. "Choose the option that appears
verbatim in the passage" scored ~0.96 (English) and ~0.92 (Bangla): the task was
solvable without reading, and solvable to a *language-dependent* degree, sitting
directly on the interaction P1 exists to measure. Version 2 draws distractors
from the item's **own** article and places the window over the span covering all
four options, so presence is constant across options and the substring heuristic
scores exactly **0.25 in both languages**.

Raising `max_seq_tokens` from 1024 to 2048 was required to make that
language-neutral — at 1024 the construction succeeds for 97.9% of English rows
against 74.8% of Bangla's — and costs ~nothing (1.00× / 1.02× total training
tokens, because batch size is 1 and only the overflowing tail gets longer). The
two training partitions are trimmed to a common 3,732 items so neither arm sees
more data than the other. `configs/p1_split_manifest.json` reproduces exactly
from its pinned dataset revision, seed and tokenizer.

Two pipeline gaps were closed alongside it. `evaluate_cell` had no way to load a
merged fine-tuned checkpoint, so the one full P1 evaluation ever run was driven
by ad-hoc notebook code that scored the *base* model — its "fine-tuned" logits
were bit-identical to the base model's at every precision. `run_eval.py` now
takes `--local-checkpoint`/`--ft-lang`, every result row records `arm` and
`weights_from`, `merge_and_save` proves the merge moved the weights, and the
smoke test compares the two arms directly.

**246 tests pass, 14 skipped** (the skips are opt-in network and CUDA-gated
tests). P0's original 53 pass unchanged and `freeze_p0.py` reports the P0 config
subtree and every strict file byte-identical. The full non-GPU chain is verified
end to end.

**No P1 GPU run has been made.** Run `notebooks/kaggle_p1_A_validate.ipynb`
first: it re-derives the corpus, runs the learnability gate — which stops the
project if the base model already solves the P1 task, against a ceiling read
from P0's own best cell rather than chosen by hand — and runs the nine-check
smoke test. Only then are sessions B and C worth starting. No P1 number exists
anywhere in this repository.

The pilot numbers in the three Kaggle notebooks at `fairuz-anadi/quantization`
are 120-item exploratory runs with a different prompt and a different scored
token — they are not comparable to what this pipeline produces and are not
paper data.

Not yet built: FLORES tokenizer fertility (P5; FLORES is a gated dataset and
needs an HF token with the terms accepted).
