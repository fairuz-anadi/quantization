"""BELEBELE letter-logit evaluation, one (model, language, precision) cell.

Scoring: a single forward pass over the frozen prompt; the prediction is the
argmax over the four option-letter token logits at the final position. One
forward per item, no generation, therefore no parse failures -- which matters
because parse failures are language-correlated and would leak straight into the
headline cross-language effect.

Timing (per the brief's protocol): model load, tokenizer init and dataset load
sit outside every timer. Only the forward pass is timed, with an explicit CUDA
synchronize on both sides so GPU work is actually inside the measured window.
Warm-up iterations are run and discarded.

Output goes to --outdir, which on Kaggle is /kaggle/working. It is NEVER written
directly into the repo's committed raw directory: that is populated only by
`kaggle kernels output`, so the provenance chain stays one-directional.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from . import config as cfg_mod
from . import data as data_mod
from . import model as model_mod


def _env_metadata() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    # torchao is recorded despite being unused: PEFT's LoRA dispatcher calls
    # is_torchao_available() unconditionally, and a torchao older than PEFT
    # expects makes that RAISE rather than return False, which breaks adapter
    # attachment outright. Its version belongs in the provenance record.
    for lib in ("transformers", "bitsandbytes", "accelerate", "datasets", "peft",
                "torchao"):
        try:
            env[f"{lib}_version"] = getattr(__import__(lib), "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            env[f"{lib}_version"] = f"NOT INSTALLED ({type(exc).__name__})"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        env.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{major}.{minor}",
            "n_gpus_visible": torch.cuda.device_count(),
            "total_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        })
    return env


@torch.no_grad()
def _forward_letter_logits(model, input_ids, option_ids) -> list[float]:
    logits = model(input_ids).logits[0, -1, :].float()
    return [logits[i].item() for i in option_ids]


def evaluate_cell(
    cfg: dict[str, Any],
    *,
    hf_id: str,
    model_alias: str,
    revision: str,
    precision: str,
    lang: str,
    outdir: Path,
    tag: str,
    limit: int | None = None,
    warmup: int = 5,
    repeats: int = 1,
    store_prompts: bool = False,
    device: str = "cuda:0",
    local_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Run one cell end to end and write per-item JSONL + a meta manifest.

    `local_checkpoint` (P1 only) scores a merged fine-tuned checkpoint from disk
    instead of the pinned Hub model. Everything else is identical -- same items,
    same prompt, same letter_logit scoring, same quantization kwargs -- so the
    only thing that differs between the Base and FT arms is the weights.

    Leaving it None reproduces P0's behaviour exactly.

    It is a REQUIRED parameter of the FT arm, not an optional nicety. Before it
    existed there was no sanctioned way to evaluate a fine-tuned checkpoint at
    all, the one full P1 evaluation was run by ad-hoc code outside the repo, and
    it silently scored the base model -- see `finetune.assert_merge_moved`.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    max_len = cfg_mod.require(cfg, "scoring.max_input_tokens")
    method = cfg_mod.require(cfg, "scoring.method")
    if method != "letter_logit":
        raise ValueError(
            f"scoring.method is {method!r}; this evaluator implements "
            f"letter_logit only. Changing method changes the numbers, so it "
            f"must be an explicit config decision, not an implicit one."
        )

    # Checked before anything expensive: a missing FT checkpoint must stop the
    # run outright, because loading the base model instead would produce an
    # "FT" cell identical to the Base arm -- and that has already happened once.
    if local_checkpoint is not None and not Path(local_checkpoint).is_dir():
        raise FileNotFoundError(
            f"local_checkpoint {local_checkpoint!r} is not a directory. A "
            f"missing FT checkpoint must stop the run: loading the base model "
            f"instead would produce an 'FT' cell identical to the Base arm."
        )

    rows = data_mod.load_language(cfg, lang)
    if limit is not None:
        rows = rows[:limit]

    load_t0 = time.perf_counter()
    tok, model, layer_counts = model_mod.load(cfg, hf_id, revision, precision,
                                              device,
                                              local_checkpoint=local_checkpoint)
    load_seconds = time.perf_counter() - load_t0
    option_ids = model_mod.option_token_ids(cfg, tok)

    dev = torch.device(device)
    torch.cuda.reset_peak_memory_stats(dev)

    def encode(row: dict):
        prompt = data_mod.build_prompt(cfg, row)
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(dev)
        return prompt, ids

    # ---- warm-up, discarded -------------------------------------------------
    for row in rows[:warmup]:
        _, ids = encode(row)
        _forward_letter_logits(model, ids, option_ids)
    torch.cuda.synchronize(dev)

    run_id = f"{tag}__{model_alias}__{lang}__{precision}"
    jsonl_path = outdir / f"{run_id}.jsonl"

    n_correct = 0
    latencies: list[float] = []
    input_tokens: list[int] = []
    started = time.time()

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            prompt, ids = encode(row)
            reps: list[float] = []
            scores: list[float] = []
            for _ in range(repeats):
                torch.cuda.synchronize(dev)
                t = time.perf_counter()
                scores = _forward_letter_logits(model, ids, option_ids)
                torch.cuda.synchronize(dev)
                reps.append((time.perf_counter() - t) * 1000.0)
            latency_ms = statistics.median(reps)

            pred = int(max(range(4), key=lambda i: scores[i])) + 1   # 1-indexed
            gold = int(row["gold"])
            correct = int(pred == gold)
            n_correct += correct
            latencies.append(latency_ms)
            input_tokens.append(int(ids.shape[1]))

            record = {
                "run_id": run_id,
                "model": hf_id,
                # Where the WEIGHTS came from, recorded per item. "hub"
                # means the pinned base model; anything else is a merged FT
                # checkpoint. Without this, a Base row and an FT row are
                # indistinguishable in the raw output.
                "weights_from": local_checkpoint or "hub",
                "arm": "base" if local_checkpoint is None else "finetuned",
                "model_alias": model_alias,
                "model_revision": revision,
                "precision": precision,
                "lang": lang,
                "item_id": row["item_id"],
                "pred": pred,
                "gold": gold,
                "correct": correct,
                "letter_logits": scores,
                "latency_ms": latency_ms,
                "input_tokens": int(ids.shape[1]),
                "truncated": bool(ids.shape[1] >= max_len),
                # The full prompt is exactly reconstructible from the frozen
                # template + pinned dataset revision + item_id, so the digest is
                # a complete and far more compact provenance record.
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            if store_prompts:
                record["prompt"] = prompt
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    quartiles = statistics.quantiles(latencies, n=4) if len(latencies) > 3 else None
    meta = {
        "run_id": run_id,
        "tag": tag,
        "model": hf_id,
        "weights_from": local_checkpoint or "hub",
        "arm": "base" if local_checkpoint is None else "finetuned",
        "model_alias": model_alias,
        "model_revision": revision,
        "precision": precision,
        "lang": lang,
        "n_items": len(rows),
        "n_correct": n_correct,
        "accuracy": n_correct / len(rows),
        "n_truncated": sum(1 for t in input_tokens if t >= max_len),
        "median_latency_ms": statistics.median(latencies),
        "mean_latency_ms": statistics.mean(latencies),
        "p25_latency_ms": quartiles[0] if quartiles else None,
        "p75_latency_ms": quartiles[2] if quartiles else None,
        "median_input_tokens": statistics.median(input_tokens),
        "total_wall_seconds": time.time() - started,
        "load_seconds": load_seconds,
        "peak_memory_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1024**3,
        "peak_memory_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1024**3,
        "quantized_layer_counts": layer_counts,
        "option_token_ids": option_ids,
        "scoring_method": method,
        "prompt_template_sha256": hashlib.sha256(
            cfg_mod.require(cfg, "scoring.prompt_template").encode("utf-8")).hexdigest(),
        "benchmark_revision": cfg_mod.require(cfg, "benchmark.hf_revision"),
        "item_manifest_sha256": data_mod.load_manifest()["sha256"],
        "warmup": warmup,
        "repeats": repeats,
        "device": device,
        "limit": limit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": _env_metadata(),
    }
    (outdir / f"{run_id}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
