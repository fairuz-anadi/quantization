"""Model loading, one function per precision condition.

Three things this module refuses to do, because each of them would quietly
invalidate a headline comparison:

* Load an unpinned revision. Every load passes the commit SHA from the config.
* Shard across GPUs. Kaggle gives 2xT4; if FP16 sharded and NF4 did not, the
  latency column would compare device topologies rather than precisions. All
  precisions are pinned to cuda:0 and a model that does not fit there is a
  hard failure, not a silent fallback.
* Fall back to FP16 when a quantizer is unavailable. `assert_precision_applied`
  inspects the loaded modules and raises if the requested quantization is not
  actually present in the weights.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import config as cfg_mod

PRECISIONS = ("fp16", "int8_llmint8", "nf4")


class PrecisionError(RuntimeError):
    """Raised when the loaded model is not in the precision that was requested."""


def _quant_kwargs(precision: str) -> dict[str, Any]:
    if precision == "fp16":
        return {"dtype": torch.float16}
    if precision == "int8_llmint8":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_threshold=6.0
            ),
            "dtype": torch.float16,
        }
    if precision == "nf4":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
        }
    raise ValueError(
        f"unknown precision {precision!r}; allowed {list(PRECISIONS)}. "
        f"Bare 'int8'/'int4' are never valid names here."
    )


def assert_precision_applied(model, precision: str) -> dict[str, int]:
    """Prove the quantizer actually took effect. A silent FP16 fallback would
    produce a perfectly plausible table of numbers that means nothing."""
    counts = {"Linear8bitLt": 0, "Linear4bit": 0, "Linear": 0}
    for m in model.modules():
        name = type(m).__name__
        if name in counts:
            counts[name] += 1

    if precision == "int8_llmint8" and counts["Linear8bitLt"] == 0:
        raise PrecisionError(
            "int8_llmint8 requested but no bitsandbytes Linear8bitLt layer was "
            "found. The model silently loaded in another precision."
        )
    if precision == "nf4" and counts["Linear4bit"] == 0:
        raise PrecisionError(
            "nf4 requested but no bitsandbytes Linear4bit layer was found. "
            "The model silently loaded in another precision."
        )
    if precision == "fp16":
        dtypes = {p.dtype for p in model.parameters()}
        if counts["Linear8bitLt"] or counts["Linear4bit"]:
            raise PrecisionError(f"fp16 requested but quantized layers present: {counts}")
        if dtypes != {torch.float16}:
            raise PrecisionError(f"fp16 requested but parameter dtypes are {dtypes}")
    return counts


def load(cfg: dict[str, Any], hf_id: str, revision: str, precision: str,
         device: str = "cuda:0"):
    """Load tokenizer + model at a pinned revision in the requested precision."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")

    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    tok.truncation_side = cfg_mod.require(cfg, "scoring.truncation_side")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = _quant_kwargs(precision)
    # device_map pins every layer to ONE device. See module docstring.
    kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision,
                                                 low_cpu_mem_usage=True, **kwargs)
    model.eval()
    layer_counts = assert_precision_applied(model, precision)
    return tok, model, layer_counts


def option_token_ids(cfg: dict[str, Any], tok) -> list[int]:
    """Token ids for the four answer letters as the model would emit them.

    The prompt ends with `Answer:` and no trailing space, so the next token is
    " A", not "A" -- those are different tokens in a BPE vocabulary, and reading
    the logit of the wrong one measures something the model was never about to
    say. Each prefixed letter must be exactly one token or we stop.
    """
    letters = cfg_mod.require(cfg, "scoring.option_letters")
    prefix = cfg_mod.require(cfg, "scoring.option_prefix")
    ids: list[int] = []
    for letter in letters:
        enc = tok.encode(f"{prefix}{letter}", add_special_tokens=False)
        if len(enc) != 1:
            raise PrecisionError(
                f"option {prefix!r}+{letter!r} encodes to {len(enc)} tokens "
                f"({enc}), not 1. letter_logit scoring requires a single token "
                f"per option; this tokenizer needs a different option_prefix."
            )
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise PrecisionError(f"option letters collide to the same token id: {ids}")
    return ids
