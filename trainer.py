"""Real, offline-first, single-device LoRA / QLoRA / Tuna training worker.

stdout is an NDJSON event stream consumed by the local dashboard. Imports of ML
packages are delayed so validation/core tests need only the Python standard lib.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from tuna_core import load_jsonl, pad_examples, policy_manifest, tokenize_conversation
from gguf_import import local_gguf_path


def emit(event_type: str, **values) -> None:
    print(json.dumps({"type": event_type, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def normalized_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Training config must be an object")
    config = {
        "model_path": raw.get("model_path", raw.get("model", raw.get("model_id", ""))),
        "dataset_path": raw.get("dataset_path", ""),
        "method": str(raw.get("method", "qlora")).lower(),
        "system_prompt": raw.get("system_prompt", ""),
        "max_seq_length": raw.get("max_seq_length", raw.get("sequence_length", 1024)),
        "batch_size": raw.get("batch_size", 1),
        "gradient_accumulation_steps": raw.get("gradient_accumulation_steps", raw.get("gradient_accumulation", 8)),
        "learning_rate": raw.get("learning_rate", 2e-4),
        "epochs": raw.get("epochs", 1),
        "lora_rank": raw.get("lora_rank", raw.get("rank", 16)),
        "lora_alpha": raw.get("lora_alpha", 32),
        "lora_dropout": raw.get("lora_dropout", 0.05),
        "gradient_checkpointing": raw.get("gradient_checkpointing", True),
        "seed": raw.get("seed", 42),
        "max_steps": raw.get("max_steps", 0),
        "warmup_ratio": raw.get("warmup_ratio", 0.03),
        "weight_decay": raw.get("weight_decay", 0.01),
        "offline": raw.get("offline", True),
    }
    if config["method"] not in ("lora", "qlora", "tuna"):
        raise ValueError("method must be lora, qlora, or tuna")
    for key in ("model_path", "dataset_path"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} is required")
    if config["method"] == "tuna" and (not isinstance(config["system_prompt"], str) or not config["system_prompt"].strip()):
        raise ValueError("Tuna requires a non-empty system_prompt")
    for key, low, high in (("max_seq_length", 32, 131072), ("batch_size", 1, 128), ("gradient_accumulation_steps", 1, 1024), ("lora_rank", 1, 512), ("lora_alpha", 1, 2048), ("max_steps", 0, 10000000), ("seed", 0, 2147483647)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or not low <= value <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
        config[key] = int(value)
    for key, low, high in (("learning_rate", 1e-8, 0.1), ("epochs", 0.01, 1000), ("lora_dropout", 0, 0.9), ("warmup_ratio", 0, 0.5), ("weight_decay", 0, 1)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        config[key] = float(value)
    if not isinstance(config["offline"], bool):
        raise ValueError("offline must be a boolean")
    if not isinstance(config["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be a boolean")
    if not Path(config["dataset_path"]).is_file():
        raise ValueError("dataset_path is not a readable local file")
    local_gguf_path(config["model_path"])
    return config


def length_grouped_batches(examples: list[dict], batch_size: int, seed: int) -> list[list[int]]:
    """Shuffle length buckets each epoch to reduce padding without fixed ordering."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    batches = []
    bucket_size = batch_size * 32
    for start in range(0, len(order), bucket_size):
        bucket = sorted(order[start:start + bucket_size], key=lambda i: len(examples[i]["input_ids"]))
        batches.extend(bucket[j:j + batch_size] for j in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    return batches


def configure_model_access(offline: bool) -> dict:
    """Only model/tokenizer loading uses the Hub; training data remains local."""
    if not isinstance(offline, bool):
        raise ValueError("offline must be a boolean")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    return {"local_files_only": offline, "trust_remote_code": False}


def run(config: dict, output: Path) -> dict:
    config = normalized_config(config)
    output.mkdir(parents=True, exist_ok=True)
    model_access = configure_model_access(config["offline"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    emit("log", message="Loading training libraries. Datasets and training stay on this machine.")
    emit("log", message="Model access: local cache only." if config["offline"] else "Model downloads enabled: missing tokenizer and model files will be cached from Hugging Face.")
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Training dependencies are unavailable. Install requirements-train.txt in the Python environment running the server. " + str(exc)) from exc

    quantized = config["method"] in ("qlora", "tuna")
    has_cuda = torch.cuda.is_available()
    if quantized and not has_cuda:
        raise RuntimeError("QLoRA and Tuna require a CUDA-capable GPU and bitsandbytes; LoRA can run on CPU.")
    if quantized:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("QLoRA and Tuna require bitsandbytes; install requirements-train.txt.") from exc
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if has_cuda:
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if has_cuda else torch.float32
    wall_start = time.perf_counter()
    source_model_path = config["model_path"]
    imported = None
    if source := local_gguf_path(source_model_path):
        from gguf_import import import_gguf
        base, imported = import_gguf(source, output.parent.parent / "models", output,
            progress=lambda message: emit("log", message=message))
        config["model_path"] = str(base)
        emit("log", message="GGUF import complete. Fine-tuning its restored weights with " + config["method"].upper() + ".")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], **model_access, use_fast=True)
    except OSError as exc:
        if config["offline"]:
            raise RuntimeError("The selected model is not cached or its local files are incomplete. Enable 'Download missing models' and retry, or choose a complete local model directory. " + str(exc)) from exc
        raise
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rows = load_jsonl(config["dataset_path"], config["system_prompt"] if config["method"] == "tuna" else None)
    examples, dropped = [], 0
    for row in rows:
        encoded, ignored = tokenize_conversation(row, tokenizer, config["max_seq_length"])
        examples.extend(encoded)
        dropped += ignored
    if not examples:
        raise ValueError("No assistant target tokens fit the sequence length. Increase max_seq_length or shorten prompts/policy.")
    emit("log", message=f"Prepared {len(examples)} assistant examples from {len(rows)} JSONL rows; {dropped} replies omitted after truncation.")
    if dropped:
        emit("log", level="warning", message="Some prompts consume the whole context. Their replies are excluded; the system policy is never left-truncated.")

    load_kwargs = dict(**model_access, torch_dtype=dtype, attn_implementation="sdpa")
    if quantized:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        load_kwargs["device_map"] = {"": 0}
    emit("log", message=f"Loading {config['method'].upper()} model with SDPA on {device}.")
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], **load_kwargs)
    context = getattr(model.config, "max_position_embeddings", None)
    pad_multiple = 8 if has_cuda and config["max_seq_length"] % 8 == 0 else 1
    if context and config["max_seq_length"] > context:
        raise ValueError(f"Sequence length {config['max_seq_length']} exceeds model context {context}")
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.to(device)
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_class = bnb.optim.PagedAdamW8bit if quantized else torch.optim.AdamW
    optimizer = optimizer_class(trainable, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    batches_per_epoch = math.ceil(len(examples) / config["batch_size"])
    updates_per_epoch = math.ceil(batches_per_epoch / config["gradient_accumulation_steps"])
    total_steps = config["max_steps"] or math.ceil(updates_per_epoch * config["epochs"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * config["warmup_ratio"]), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=has_cuda and dtype == torch.float16)
    versions = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    dataset_hash = hashlib.sha256(Path(config["dataset_path"]).read_bytes()).hexdigest()
    run_info = {"config": config, "source_model_path": source_model_path, "gguf_import": imported, "versions": versions, "dataset_sha256": dataset_hash, "device": str(device), "device_name": torch.cuda.get_device_name(0) if has_cuda else "CPU", "dtype": str(dtype), "examples": len(examples), "dropped_examples": dropped, "training_rows": len(rows), "trainable_parameters": sum(p.numel() for p in trainable), "total_steps": total_steps, "attention": "sdpa", "optimizer": optimizer_class.__name__, "loss": "assistant-only", "packing": False}
    (output / "run_manifest.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    emit("log", message=f"Training {run_info['trainable_parameters']:,} adapter parameters for {total_steps} optimizer steps.", total_steps=total_steps)
    if has_cuda:
        torch.cuda.synchronize()
    train_start = time.perf_counter()
    load_seconds = train_start - wall_start
    step, epoch_index, tokens_seen, targets_seen = 0, 0, 0, 0
    total_loss_sum = 0.0
    metrics_path = output / "training_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < total_steps:
            batches = length_grouped_batches(examples, config["batch_size"], config["seed"] + epoch_index)
            for group_start in range(0, len(batches), config["gradient_accumulation_steps"]):
                group = batches[group_start:group_start + config["gradient_accumulation_steps"]]
                group_targets = sum(sum(label != -100 for label in examples[index]["labels"][1:]) for batch in group for index in batch)
                optimizer.zero_grad(set_to_none=True)
                group_loss_sum = 0.0
                for indices in group:
                    raw_batch = pad_examples([examples[index] for index in indices], tokenizer.pad_token_id, multiple=pad_multiple)
                    batch = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in raw_batch.items()}
                    count = sum(sum(label != -100 for label in examples[index]["labels"][1:]) for index in indices)
                    tokens_seen += sum(len(examples[index]["input_ids"]) for index in indices)
                    autocast = torch.autocast("cuda", dtype=dtype) if has_cuda else contextlib.nullcontext()
                    with autocast:
                        result = model(**batch)
                        loss = result.loss
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError("Non-finite training loss; check data, precision, and learning rate.")
                    raw_loss = float(loss.detach().item())
                    group_loss_sum += raw_loss * count
                    scaler.scale(loss * (count / group_targets)).backward()
                    del result, loss, batch
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                used_learning_rate = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                targets_seen += group_targets
                total_loss_sum += group_loss_sum
                if has_cuda:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - train_start
                metric = {"step": step, "total_steps": total_steps, "loss": group_loss_sum / group_targets, "learning_rate": used_learning_rate, "epoch": epoch_index + min(group_start + len(group), len(batches)) / len(batches), "elapsed_seconds": elapsed, "tokens_per_second": tokens_seen / elapsed, "supervised_tokens_per_second": targets_seen / elapsed, "tokens_seen": tokens_seen, "supervised_tokens_seen": targets_seen, "peak_vram_mb": torch.cuda.max_memory_allocated() / 1048576 if has_cuda else None, "peak_reserved_vram_mb": torch.cuda.max_memory_reserved() / 1048576 if has_cuda else None}
                metrics_file.write(json.dumps(metric, allow_nan=False) + "\n")
                metrics_file.flush()
                emit("metric", **metric)
                if step >= total_steps:
                    break
            epoch_index += 1
    elapsed = time.perf_counter() - train_start
    # B matrices start at zero in this LoRA configuration. A completed run
    # must contain real learned updates, not just serialized initial adapters.
    b_weights = [parameter.detach() for name, parameter in model.named_parameters() if ".lora_B." in name and parameter.requires_grad]
    changed_matrices = sum(bool(torch.count_nonzero(weight).item()) for weight in b_weights)
    if not changed_matrices:
        raise RuntimeError("Training produced no learned adapter changes. Check the dataset, learning rate and precision.")
    adapter_verification = {"lora_b_matrices": len(b_weights), "nonzero_lora_b_matrices": changed_matrices,
                            "learned_updates_present": True}
    emit("log", message=f"Verified learned weight updates in {changed_matrices} / {len(b_weights)} LoRA matrices.")
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    if config["method"] == "tuna":
        manifest = policy_manifest(config["system_prompt"], config["model_path"], dataset_sha256=dataset_hash, adapter_path="adapter", versions=versions)
        (output / "policy.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copyfile(Path(__file__).with_name("tuna_core.py"), output / "tuna_runtime.py")
        (output / "serve.py").write_text(SERVE_SCRIPT, encoding="utf-8")
    summary = {"status": "completed", "method": config["method"], "steps": step, "loss": total_loss_sum / targets_seen, "elapsed_seconds": elapsed, "load_seconds": load_seconds, "wall_seconds": time.perf_counter() - wall_start, "tokens_per_second": tokens_seen / elapsed, "tokens_seen": tokens_seen, "supervised_tokens_seen": targets_seen, "peak_vram_mb": torch.cuda.max_memory_allocated() / 1048576 if has_cuda else None, "peak_reserved_vram_mb": torch.cuda.max_memory_reserved() / 1048576 if has_cuda else None, "adapter_path": str(adapter_dir.resolve()), "model_path": config["model_path"], "source_model_path": source_model_path, "gguf_import": imported, "dataset_sha256": dataset_hash, "versions": versions, "device_name": run_info["device_name"], "benchmark_comparison": "No Unsloth comparison performed"}
    summary["adapter_verification"] = adapter_verification
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit("complete", **summary)
    return summary


SERVE_SCRIPT = '''"""Local Tuna policy-bound inference. This cannot guarantee generated compliance."""
import argparse
import json
from pathlib import Path
from tuna_runtime import PolicyBoundGenerator

parser = argparse.ArgumentParser(description=__doc__)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--prompt")
group.add_argument("--messages", help="Local JSON file containing caller user/assistant messages")
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()
messages = json.loads(Path(args.messages).read_text(encoding="utf-8")) if args.messages else [{"role": "user", "content": args.prompt}]
generator = PolicyBoundGenerator(Path(__file__).parent)
print(generator.generate(messages, max_new_tokens=args.max_new_tokens))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        run(config, args.output.resolve())
        return 0
    except KeyboardInterrupt:
        emit("error", message="Training interrupted; no completed adapter export was produced.")
        return 130
    except Exception as exc:
        emit("error", message=str(exc), error_type=type(exc).__name__)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    from app_runtime import use_app_environment
    use_app_environment(__file__)
    raise SystemExit(main())
