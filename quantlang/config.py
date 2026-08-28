"""Configuration loading with mandatory-value enforcement.

The central invariant of this repo is that unknown values are never guessed.
`configs/experiment.yaml` encodes "not yet known" as YAML null, and every read
path goes through `require()`, which raises rather than returning None. A
missing pin therefore stops a run at startup instead of silently producing
numbers that look fine and mean nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"

VALID_PRECISIONS = ("fp16", "int8_llmint8", "nf4")
VALID_SCORING_METHODS = ("letter_logit", "option_loglik")


class ConfigError(RuntimeError):
    """Raised when configuration is missing, unpinned, or self-inconsistent."""


def load(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _validate_static(cfg)
    return cfg


def require(cfg: dict[str, Any], dotted: str) -> Any:
    """Fetch `dotted` (e.g. 'benchmark.item_id_key'), raising if it is null.

    This is the only sanctioned way to read a pinnable value. Callers must not
    fall back to a default when it raises -- that is precisely the failure mode
    this function exists to prevent.
    """
    node: Any = cfg
    walked: list[str] = []
    for part in dotted.split("."):
        walked.append(part)
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"Missing config key: {'.'.join(walked)}")
        node = node[part]
    if node is None:
        raise ConfigError(
            f"Config value '{dotted}' is null, meaning NOT YET KNOWN.\n"
            f"Resolve it with the script that owns it -- do not fill it in by "
            f"hand and do not substitute a default.\n"
            f"  benchmark.item_id_key      -> scripts/verify_item_ids.py\n"
            f"  benchmark.hf_revision      -> scripts/pin_revisions.py\n"
            f"  models[].revision          -> scripts/pin_revisions.py"
        )
    return node


def _validate_static(cfg: dict[str, Any]) -> None:
    """Check the parts of the config that are knowable without any pinning."""
    precisions = cfg.get("precisions") or []
    unknown = [p for p in precisions if p not in VALID_PRECISIONS]
    if unknown:
        raise ConfigError(
            f"Unknown precision(s) {unknown}. Allowed: {list(VALID_PRECISIONS)}. "
            f"Note the naming rule: bitsandbytes NF4 is 'nf4' and LLM.int8() is "
            f"'int8_llmint8'. Bare 'int4'/'int8' are not valid -- our results "
            f"cover one quantization library and must not imply otherwise."
        )

    method = (cfg.get("scoring") or {}).get("method")
    if method not in VALID_SCORING_METHODS:
        raise ConfigError(
            f"scoring.method={method!r} invalid. "
            f"Allowed: {list(VALID_SCORING_METHODS)}"
        )

    bench = cfg.get("benchmark") or {}
    langs = bench.get("languages") or []
    ref = bench.get("reference_language")
    if ref not in langs:
        raise ConfigError(
            f"reference_language {ref!r} is not in languages {langs}. The "
            f"interaction term is defined relative to it, so it must be present."
        )
    if len(set(langs)) != len(langs):
        raise ConfigError(f"Duplicate entries in languages: {langs}")

    for m in cfg.get("models") or []:
        if "alias" not in m or "hf_id" not in m:
            raise ConfigError(f"Model entry missing alias/hf_id: {m}")
        if "__" in m["alias"]:
            raise ConfigError(
                f"Model alias {m['alias']!r} contains '__', which is the field "
                f"separator in raw result filenames. Choose another alias."
            )

    # P1 only. A config without a `finetune` block is a valid P0 config and is
    # validated exactly as before -- this branch is never entered for one.
    if cfg.get("finetune") is not None:
        _validate_finetune(cfg)


def _validate_finetune(cfg: dict[str, Any]) -> None:
    """Check the P1 block. Additive: never inspects or constrains P0 keys.

    The checks here are the ones that would otherwise fail hours into a Kaggle
    session, or -- worse -- not fail at all and quietly produce a fine-tuning
    corpus that does not match the evaluation contract.
    """
    ft = cfg["finetune"]
    bench_langs = list((cfg.get("benchmark") or {}).get("languages") or [])

    lang_configs = ft.get("lang_configs") or {}
    if set(lang_configs) != set(bench_langs):
        missing = sorted(set(bench_langs) - set(lang_configs))
        extra = sorted(set(lang_configs) - set(bench_langs))
        raise ConfigError(
            f"finetune.lang_configs must cover exactly benchmark.languages.\n"
            f"  missing: {missing}\n  extra: {extra}\n"
            f"A language present in one and absent from the other means a cell "
            f"of the P1 grid can never be filled."
        )
    if len(set(lang_configs.values())) != len(lang_configs):
        raise ConfigError(
            f"finetune.lang_configs maps two languages to the same dataset "
            f"config: {lang_configs}. That would train one language on another's "
            f"text."
        )

    frac = ft.get("train_fraction")
    if not isinstance(frac, (int, float)) or not 0.0 < float(frac) < 1.0:
        raise ConfigError(
            f"finetune.train_fraction={frac!r} must lie strictly between 0 and 1."
        )

    if not ft.get("group_key"):
        raise ConfigError(
            "finetune.group_key is required. The 80/20 split is grouped by "
            "article; splitting rows would leak one article's shared context "
            "across the train/held-out boundary."
        )

    n_options = ft.get("n_options")
    letters = (cfg.get("scoring") or {}).get("option_letters") or []
    if n_options != len(letters):
        raise ConfigError(
            f"finetune.n_options={n_options!r} disagrees with "
            f"scoring.option_letters ({len(letters)}: {letters}). P1 items are "
            f"scored by P0's letter_logit code, so the option count must match."
        )

    cap = ft.get("heldout_eval_cap")
    if not isinstance(cap, int) or cap <= 0:
        raise ConfigError(
            f"finetune.heldout_eval_cap={cap!r} must be a positive integer.")

    train_prec = ft.get("train_precision")
    if train_prec != "fp16":
        raise ConfigError(
            f"finetune.train_precision={train_prec!r}. P1 trains LoRA on the "
            f"FP16 base and quantizes only AFTER merging. Training on a "
            f"quantized base makes 'the fine-tuned model in NF4' a "
            f"mixed-precision object whose quantization effect is a different "
            f"estimand from P0's."
        )

    seeds = ft.get("seeds") or {}
    for key in ("main", "sensitivity"):
        if not isinstance(seeds.get(key), int):
            raise ConfigError(f"finetune.seeds.{key} must be an integer seed.")
    if seeds["main"] == seeds["sensitivity"]:
        raise ConfigError(
            f"finetune.seeds.main and .sensitivity are both {seeds['main']}. "
            f"The sensitivity probe would then repeat the main run and measure "
            f"nothing."
        )
    sens_langs = seeds.get("sensitivity_languages") or []
    unknown = [l for l in sens_langs if l not in bench_langs]
    if unknown:
        raise ConfigError(
            f"finetune.seeds.sensitivity_languages contains {unknown}, which "
            f"are not in benchmark.languages {bench_langs}."
        )

    if not isinstance(ft.get("split_seed"), int):
        raise ConfigError(
            "finetune.split_seed must be an integer. It is frozen once and "
            "shared by both training seeds, so the sensitivity probe varies the "
            "training run and never the data."
        )

    max_seq = (ft.get("training") or {}).get("max_seq_tokens")
    max_in = (cfg.get("scoring") or {}).get("max_input_tokens")
    if not isinstance(max_seq, int) or max_seq <= 0:
        raise ConfigError(
            f"finetune.training.max_seq_tokens={max_seq!r} must be a positive "
            f"integer.")
    if max_in is not None and max_seq > max_in:
        raise ConfigError(
            f"finetune.training.max_seq_tokens={max_seq} exceeds "
            f"scoring.max_input_tokens={max_in}. Training on longer inputs than "
            f"evaluation ever sees is a train/eval mismatch in the wrong "
            f"direction."
        )
