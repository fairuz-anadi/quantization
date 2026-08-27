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
