"""P1 LoRA fine-tuning: FP16 base -> adapter -> merged FP16 checkpoint.

The precision procedure is the whole point of this module and is not negotiable:

    FP16 base -> LoRA -> merge -> ONE merged FP16 checkpoint
                                        |
                            +-----------+-----------+
                          FP16         NF4         INT8

Training a LoRA on an already-quantized base and calling the result "the
fine-tuned NF4 model" would produce a mixed-precision object -- quantized base,
full-precision adapter -- whose quantization effect is a different quantity from
the one P0 measured. Every P1 precision cell therefore comes from the same
merged FP16 weights, quantized afterwards by exactly the code that quantized the
base model in P0.

The training objective is the second load-bearing decision. Loss covers the
answer-letter token and nothing else:

    labels = [-100, -100, ..., -100, <token id of " A">]

That is precisely the token `letter_logit` reads at evaluation time. Fine-tuning
on free-form answer text would instead teach the model to emit prose, drifting
away from the scored behaviour and pushing accuracy toward the 0.25 chance floor
where a quantization contrast can no longer be measured.

Nothing here writes into results/raw/, and nothing here touches BELEBELE: the
training corpus is the frozen P1 train partition, whose articles are disjoint
from the held-out partition and whose text was verified not to overlap BELEBELE.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from . import config as cfg_mod
from . import model as model_mod
from . import p1data

# Loss is computed on this label only; everything else in the sequence is
# masked. -100 is torch's ignore_index for cross entropy.
IGNORE_INDEX = -100


class FineTuneError(RuntimeError):
    """Raised when fine-tuning cannot be performed as specified."""


# --------------------------------------------------------------------------- #
# provenance helpers
# --------------------------------------------------------------------------- #

def git_commit() -> str:
    """The repo commit that produced a run, or an explicit marker."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=str(cfg_mod.REPO_ROOT), timeout=30)
        sha = out.stdout.strip()
        return sha if sha else "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def directory_digest(path: Path) -> dict[str, Any]:
    """sha256 over a checkpoint's weight files, plus its total size.

    Weights are hashed as raw bytes (they are binary; no newline normalisation
    applies) in sorted-name order, so the digest identifies the checkpoint
    itself rather than the order it happened to be written in.
    """
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    h = hashlib.sha256()
    total = 0
    for f in files:
        h.update(f.relative_to(path).as_posix().encode("utf-8"))
        data = f.read_bytes()
        h.update(data)
        total += len(data)
    return {
        "sha256": h.hexdigest(),
        "n_files": len(files),
        "total_bytes": total,
        "total_gb": round(total / 1024**3, 4),
    }


def env_metadata() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    # torchao: unused by this pipeline, but PEFT's LoRA dispatcher probes it on
    # every adapter attachment and an out-of-range version raises. Recorded so a
    # future breakage is diagnosable from the run manifest alone.
    for lib in ("transformers", "peft", "accelerate", "datasets", "bitsandbytes",
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


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# examples
# --------------------------------------------------------------------------- #

def encode_example(cfg: dict[str, Any], tok, item: dict, option_ids: list[int],
                   max_seq_tokens: int) -> dict[str, list[int]]:
    """One training example: P0's prompt, then the single answer-letter label.

    The prompt is truncated to `max_seq_tokens - 1` so the label token always
    fits. Truncation is left-side, taken from `scoring.truncation_side`, so the
    question and the four options are never cut off -- the same guarantee P0
    evaluation relies on.
    """
    prompt = p1data.build_p1_prompt(cfg, item)
    enc = tok(prompt, truncation=True, max_length=max_seq_tokens - 1)
    input_ids = list(enc["input_ids"])

    gold = int(item["gold"])
    if not 1 <= gold <= len(option_ids):
        raise FineTuneError(
            f"{item['item_id']}: gold {gold} outside 1..{len(option_ids)}")
    label_id = option_ids[gold - 1]

    input_ids.append(label_id)
    labels = [IGNORE_INDEX] * (len(input_ids) - 1) + [label_id]
    return {"input_ids": input_ids, "labels": labels,
            "item_id": item["item_id"], "label_id": label_id}


def build_examples(cfg: dict[str, Any], tok, items: list[dict],
                   option_ids: list[int]) -> list[dict]:
    max_seq = int(cfg_mod.require(cfg, "finetune.training.max_seq_tokens"))
    return [encode_example(cfg, tok, it, option_ids, max_seq) for it in items]


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #

def lora_config(cfg: dict[str, Any]):
    from peft import LoraConfig
    lc = cfg_mod.require(cfg, "finetune.lora")
    return LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        lora_dropout=float(lc["dropout"]),
        target_modules=list(lc["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def assert_peft_dispatch_is_usable() -> None:
    """Fail early, and legibly, on the PEFT/torchao version conflict.

    PEFT builds a fixed list of backend dispatchers for every LoRA layer it
    creates, and `dispatch_torchao` calls `is_torchao_available()` whether or not
    torchao is used. That helper does not merely return False for an old
    torchao -- it RAISES. So an environment carrying torchao below the version
    PEFT expects cannot attach a LoRA adapter at all, and fails once per target
    module with a traceback that points at torchao rather than at the mismatch.

    This project never uses torchao; INT8 and NF4 both come from bitsandbytes.
    The fix is to remove torchao, not to upgrade it -- upgrading it can drag
    torch, and Kaggle's torch is CUDA-matched and must not be reinstalled:

        pip uninstall -y torchao
    """
    try:
        from peft.import_utils import is_torchao_available
    except ImportError:
        return          # older PEFT with no torchao dispatcher at all
    try:
        is_torchao_available()
    except ImportError as exc:
        raise FineTuneError(
            f"PEFT cannot attach a LoRA adapter in this environment: {exc}\n"
            f"torchao is unused by this pipeline -- INT8 and NF4 both come from "
            f"bitsandbytes -- but PEFT probes it for every LoRA layer and an "
            f"out-of-range version raises instead of returning False.\n"
            f"Fix the environment, do not work around it:\n"
            f"    pip uninstall -y torchao\n"
            f"Upgrading torchao instead risks pulling a different torch, and "
            f"Kaggle's torch is CUDA-matched."
        ) from exc


def attach_adapter(cfg: dict[str, Any], base_model):
    """Wrap a base model in a LoRA adapter and report parameter counts.

    LoRA parameters are held in fp32 while the frozen base stays fp16. Adapter
    updates are small relative to fp16's resolution near zero, and letting them
    accumulate in fp16 loses them outright.
    """
    from peft import get_peft_model

    assert_peft_dispatch_is_usable()
    peft_model = get_peft_model(base_model, lora_config(cfg))
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    if trainable == 0:
        raise FineTuneError(
            "LoRA attached but no parameter requires grad. target_modules "
            "probably matched nothing in this architecture."
        )
    return peft_model, {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "base_parameters": total - trainable,
        "trainable_percent": 100.0 * trainable / total,
    }


def adapter_is_active(peft_model) -> dict[str, int]:
    """Count the LoRA layers actually present in the wrapped model.

    A silently inert adapter -- wrong target_modules, a disabled adapter -- would
    train, save, merge and evaluate without error while changing nothing, and
    the FT arm would simply reproduce the base arm.
    """
    from peft.tuners.lora import LoraLayer
    counts = {"lora_layers": 0, "lora_A": 0, "lora_B": 0}
    for module in peft_model.modules():
        if isinstance(module, LoraLayer):
            counts["lora_layers"] += 1
    for name, _ in peft_model.named_parameters():
        if "lora_A" in name:
            counts["lora_A"] += 1
        elif "lora_B" in name:
            counts["lora_B"] += 1
    return counts


def assert_adapter_applied(peft_model) -> dict[str, int]:
    counts = adapter_is_active(peft_model)
    if counts["lora_layers"] == 0 or counts["lora_A"] == 0:
        raise FineTuneError(
            f"No LoRA layer found in the wrapped model: {counts}. The adapter "
            f"was not applied, so training would change nothing."
        )
    return counts


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def _lr_at(step: int, total: int, base_lr: float, warmup_ratio: float,
           schedule: str) -> float:
    warmup = max(1, int(round(warmup_ratio * total)))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule != "cosine":
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_lora(cfg: dict[str, Any], peft_model, examples: list[dict], *,
               seed: int, device: str = "cuda:0",
               log_every: int = 10) -> dict[str, Any]:
    """One epoch-based LoRA training run. Deterministic given `seed`.

    A hand-written loop rather than Trainer: it is short enough to read in one
    sitting, it pins exactly what goes into the recorded metadata (steps, loss,
    schedule), and it does not move under us across transformers releases.
    """
    tcfg = cfg_mod.require(cfg, "finetune.training")
    epochs = int(tcfg["epochs"])
    accum = int(tcfg["gradient_accumulation_steps"])
    base_lr = float(tcfg["learning_rate"])
    warmup_ratio = float(tcfg["warmup_ratio"])
    schedule = str(tcfg["lr_scheduler"])
    max_grad_norm = float(tcfg["max_grad_norm"])
    use_checkpointing = bool(tcfg["gradient_checkpointing"])

    if int(tcfg["per_device_train_batch_size"]) != 1:
        raise FineTuneError(
            "per_device_train_batch_size must be 1: examples have very "
            "different lengths and padding them would put pad tokens in the "
            "attention window of a length-sensitive task."
        )

    set_all_seeds(seed)
    dev = torch.device(device)

    if use_checkpointing and hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()
        if hasattr(peft_model, "enable_input_require_grads"):
            # Without this the checkpointed graph has no grad-requiring input
            # and every LoRA gradient comes back None.
            peft_model.enable_input_require_grads()

    peft_model.train()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=base_lr)

    order = list(range(len(examples)))
    rng = random.Random(seed)

    n_micro = len(examples) * epochs
    total_steps = max(1, math.ceil(n_micro / accum))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    history: list[dict[str, float]] = []
    losses: list[float] = []
    micro = 0
    step = 0
    started = time.time()

    for epoch in range(epochs):
        rng.shuffle(order)
        for idx in order:
            ex = examples[idx]
            input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=dev)
            labels = torch.tensor([ex["labels"]], dtype=torch.long, device=dev)

            out = peft_model(input_ids=input_ids, labels=labels)
            loss = out.loss
            if not torch.isfinite(loss):
                raise FineTuneError(
                    f"non-finite loss at micro-step {micro} on item "
                    f"{ex['item_id']!r}. Stopping rather than training through it."
                )
            losses.append(float(loss.detach()))
            scaler.scale(loss / accum).backward()
            micro += 1

            if micro % accum == 0 or micro == n_micro:
                lr = _lr_at(step, total_steps, base_lr, warmup_ratio, schedule)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                window = losses[-accum:]
                if step % log_every == 0 or step == total_steps:
                    print(f"    step {step}/{total_steps}  "
                          f"loss={sum(window) / len(window):.4f}  lr={lr:.2e}",
                          flush=True)
                history.append({"step": step, "lr": lr,
                                "loss": sum(window) / len(window)})

    peft_model.eval()
    first = losses[: max(1, len(losses) // 10)]
    last = losses[-max(1, len(losses) // 10):]
    return {
        "epochs": epochs,
        "n_examples": len(examples),
        "n_micro_steps": micro,
        "n_optimizer_steps": step,
        "gradient_accumulation_steps": accum,
        "loss_first": sum(losses[:1]) / 1,
        "loss_mean": sum(losses) / len(losses),
        "loss_final": losses[-1],
        "loss_first_decile_mean": sum(first) / len(first),
        "loss_last_decile_mean": sum(last) / len(last),
        "train_seconds": time.time() - started,
        "lr_schedule": schedule,
        "base_learning_rate": base_lr,
        "history": history,
    }


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #

def _target_weight_snapshot(peft_model) -> dict[str, Any]:
    """CPU copies of the base weights inside every LoRA-wrapped module.

    Keyed by the module's name with PEFT's wrapper segments stripped, so the
    keys line up with the plain checkpoint that `merge_and_unload` returns.
    """
    from peft.tuners.lora import LoraLayer

    out: dict[str, Any] = {}
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer) and hasattr(module, "base_layer"):
            key = (name.replace("base_model.model.", "")
                       .replace(".base_layer", ""))
            out[key] = module.base_layer.weight.detach().to("cpu",
                                                            torch.float32).clone()
    return out


# A merged adapter must move the weights it was trained on. The threshold is
# only there to separate "moved" from "bit-identical": any real LoRA update
# clears it by orders of magnitude, and an adapter whose B matrices are still at
# their zero initialisation produces exactly 0.0.
MIN_MERGE_DELTA = 1e-6


def assert_merge_moved(merged, before: dict[str, Any]) -> dict[str, Any]:
    """Prove `merge_and_unload` actually folded the adapter into the weights.

    This gate exists because of a run that did not have it. A full English P1
    fine-tune, merge and three-precision evaluation completed, and the resulting
    "fine-tuned" logits were bit-identical to the base model's on all 900
    BELEBELE items at all three precisions -- maximum difference 0.000000. That
    number cannot come from a merged adapter: merging a trained LoRA perturbs
    FP16 weights, and even an adapter that learned to reproduce the base model
    would differ in the low bits. It is the signature of the base weights being
    scored instead.

    Nothing in the pipeline noticed, because nothing measured it. Structural
    checks did not help: `assert_adapter_applied` proves LoRA layers EXIST, and
    the dtype check proves the checkpoint is FP16. Neither asks whether the
    numbers changed.
    """
    after = dict(merged.named_parameters())
    checked = 0
    max_delta = 0.0
    worst = None
    for key, prior in before.items():
        param = after.get(f"{key}.weight")
        if param is None:
            continue
        d = float((param.detach().to("cpu", torch.float32) - prior)
                  .abs().max().item())
        checked += 1
        if d > max_delta:
            max_delta, worst = d, key

    if checked == 0:
        raise FineTuneError(
            "no LoRA-targeted module could be matched between the wrapped model "
            "and the merged checkpoint, so the merge cannot be verified. "
            "Refusing to ship a checkpoint that might be the base model."
        )
    if max_delta < MIN_MERGE_DELTA:
        raise FineTuneError(
            f"merging the adapter changed nothing: the largest weight movement "
            f"across {checked} LoRA-targeted modules is {max_delta:.3e}, below "
            f"{MIN_MERGE_DELTA:.0e}. The 'fine-tuned' checkpoint IS the base "
            f"model, and every FT cell built from it would silently reproduce "
            f"the Base arm. Check that gradients reached the LoRA B matrices."
        )
    return {"n_modules_checked": checked,
            "max_abs_weight_delta": max_delta,
            "largest_movement_in": worst}


def merge_and_save(peft_model, tok, out_dir: Path) -> dict[str, Any]:
    """Merge the adapter into FP16 base weights and write one checkpoint.

    `merge_and_unload` folds BA into W, so what lands on disk is an ordinary
    causal-LM checkpoint. That matters: the quantizers in `model._quant_kwargs`
    then act on it exactly as they acted on the base model in P0, with no
    adapter left over to stay in full precision.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the pre-merge weights of the modules LoRA actually targets, so
    # the merge can be PROVEN to have changed something. See assert_merge_moved.
    before = _target_weight_snapshot(peft_model)

    merged = peft_model.merge_and_unload()
    merged = merged.half()
    delta = assert_merge_moved(merged, before)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    dtypes = {str(p.dtype) for p in merged.parameters()}
    if dtypes != {"torch.float16"}:
        raise FineTuneError(
            f"merged checkpoint parameter dtypes are {dtypes}, expected only "
            f"float16. The merged model is the FP16 arm and the source for both "
            f"quantized arms; it must actually be FP16."
        )

    digest = directory_digest(out_dir)
    return {"path": str(out_dir), "parameter_dtypes": sorted(dtypes),
            "weight_delta": delta, **digest}


# --------------------------------------------------------------------------- #
# one language, end to end
# --------------------------------------------------------------------------- #

def finetune_language(
    cfg: dict[str, Any],
    *,
    lang: str,
    hf_id: str,
    model_alias: str,
    revision: str,
    seed: int,
    adapter_dir: Path,
    merged_dir: Path,
    limit: int | None = None,
    device: str = "cuda:0",
    tag: str = "p1",
) -> dict[str, Any]:
    """Train a LoRA for one language, merge it, and record everything."""
    manifest = p1data.load_split_manifest()
    items = p1data.load_partition(cfg, lang, "train", manifest)
    if limit is not None:
        items = items[:limit]
    if not items:
        raise FineTuneError(f"{lang}: no training items")

    adapter_dir = Path(adapter_dir)
    merged_dir = Path(merged_dir)

    print(f"  loading FP16 base {hf_id} @ {revision[:12]}...", flush=True)
    load_t0 = time.perf_counter()
    tok, base, layer_counts = model_mod.load(cfg, hf_id, revision, "fp16", device)
    load_seconds = time.perf_counter() - load_t0

    option_ids = model_mod.option_token_ids(cfg, tok)
    examples = build_examples(cfg, tok, items, option_ids)

    peft_model, param_counts = attach_adapter(cfg, base)
    adapter_counts = assert_adapter_applied(peft_model)
    print(f"  LoRA attached: {param_counts['trainable_parameters']:,} trainable "
          f"of {param_counts['total_parameters']:,} "
          f"({param_counts['trainable_percent']:.3f}%), "
          f"{adapter_counts['lora_layers']} LoRA layers", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    train_stats = train_lora(cfg, peft_model, examples, seed=seed, device=device)

    peak_gb = (torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3
               if torch.cuda.is_available() else None)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(adapter_dir)
    adapter_digest = directory_digest(adapter_dir)
    print(f"  adapter saved: {adapter_digest['total_bytes'] / 1024**2:.1f} MB",
          flush=True)

    merge_info = merge_and_save(peft_model, tok, merged_dir)
    print(f"  merged checkpoint: {merge_info['total_gb']:.2f} GB", flush=True)

    ft = cfg_mod.require(cfg, "finetune")
    meta = {
        "run_id": f"{tag}__{model_alias}__{lang}__seed{seed}",
        "tag": tag,
        "phase": "p1_finetune",
        "lang": lang,
        "base_model": hf_id,
        "base_model_alias": model_alias,
        "base_model_revision": revision,
        "ft_model_alias": ft_alias(model_alias, lang),
        "train_precision": ft["train_precision"],
        "seed": seed,
        "n_train_items_available": manifest["languages"][lang]["n_train_items"],
        "n_train_items_used": len(items),
        # The common size across finetune.final_scope_languages. Both FT arms
        # take the same number of gradient steps, so a difference between them
        # cannot be a difference in how much data each one saw.
        "train_equalise_cap": p1data.train_equalise_cap(cfg, manifest),
        "n_train_items_equalised": manifest["languages"][lang].get(
            "n_train_items_equalised"),
        "split_algorithm_version": manifest["context_window"][
            "algorithm_version"],
        "distractor_source": manifest["distractors"]["source"],
        "limit": limit,
        "train_dataset": ft["train_dataset"],
        "train_dataset_revision": ft["hf_revision"],
        "split_manifest_sha256": manifest["sha256"],
        "split_seed": ft["split_seed"],
        "train_items_sha256": manifest["languages"][lang]["train_items_sha256"],
        "lora": dict(ft["lora"]),
        "training_args": dict(ft["training"]),
        "parameter_counts": param_counts,
        "adapter_layer_counts": adapter_counts,
        "base_layer_counts": layer_counts,
        "training": {k: v for k, v in train_stats.items() if k != "history"},
        "loss_history": train_stats["history"],
        "base_load_seconds": load_seconds,
        "peak_memory_allocated_gb": peak_gb,
        "adapter": {"path": str(adapter_dir), **adapter_digest},
        "merged_checkpoint": merge_info,
        "option_token_ids": option_ids,
        "scoring_method": cfg_mod.require(cfg, "scoring.method"),
        "prompt_template_sha256": hashlib.sha256(
            cfg_mod.require(cfg, "scoring.prompt_template").encode("utf-8")
        ).hexdigest(),
        "git_commit": git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "env": env_metadata(),
    }
    (adapter_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def ft_alias(model_alias: str, lang: str) -> str:
    """Result alias for a fine-tuned model, e.g. qwen2.5-3b-instruct-ft-ben_Beng.

    Distinct from the base alias so P0 and P1 rows can never collide, and free
    of '__' so it stays safe in raw result filenames.
    """
    alias = f"{model_alias}-ft-{lang}"
    if "__" in alias:
        raise FineTuneError(
            f"FT alias {alias!r} contains '__', the field separator in raw "
            f"result filenames.")
    return alias
