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

Phase 0 (scaffold, config enforcement, frozen manifest, invariants) and Phase 1
(evaluation, tidy, statistics, analysis, Kaggle runner) complete. 53 tests pass.
The full non-GPU chain is verified end to end on synthetic input.

**No model has been run under this pipeline yet**; `results/raw/` is empty by
design. The pilot numbers in the three Kaggle notebooks at
`fairuz-anadi/quantization` are 120-item exploratory runs with a different
prompt and a different scored token — they are not comparable to what this
pipeline produces and are not paper data.

Not yet built: FLORES tokenizer fertility (P5; FLORES is a gated dataset and
needs an HF token with the terms accepted) and the LoRA fine-tuning experiment
(P2; starts only once P0 is complete and clean).
