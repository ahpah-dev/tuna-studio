"""Offline, disk-backed LoRA merge and standalone GGUF export.

Only the trained LoRA tensors are merged, in small CPU chunks. Frozen tensors
are copied directly from cached safetensors files without loading the base model.
No GPU allocation, model download, custom repository code or dataset upload.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import struct
import sys
import time
import uuid

FORMATS = {"q8_0", "f16"}
SOURCE_COMMIT = "970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc"


def emit(kind, **fields):
    print(json.dumps({"type": kind, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while data := handle.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def read_header(path):
    path = Path(path)
    with path.open("rb") as handle:
        first = handle.read(8)
        if len(first) != 8:
            raise ValueError("Invalid safetensors file: missing header")
        length = struct.unpack("<Q", first)[0]
        if not 2 <= length <= 100 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    available = path.stat().st_size - length - 8
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError("Invalid safetensors tensor entry")
        offsets = info.get("data_offsets", [])
        if (len(offsets) != 2 or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= available):
            raise ValueError("Invalid safetensors tensor offsets")
    return header, length + 8


def validate_adapter(config):
    if config.get("peft_type") != "LORA":
        raise ValueError("GGUF export currently supports LoRA adapters")
    unsupported = ("use_dora", "use_qalora", "lora_bias", "modules_to_save", "layer_replication",
                   "trainable_token_indices", "target_parameters", "alora_invocation_tokens",
                   "rank_pattern", "alpha_pattern", "kasa_config", "monteclora_config",
                   "arrow_config", "velora_config", "use_bdlora")
    if any(config.get(key) for key in unsupported) or config.get("bias", "none") != "none":
        raise ValueError("This adapter uses a LoRA variant that this streaming exporter cannot merge safely")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if type(rank) is not int or rank <= 0 or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError("Invalid LoRA rank or alpha")
    return alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)


def base_tensor_name(adapter_name):
    name = adapter_name
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    return re.sub(r"\.lora_A(?:\.[^.]+)?\.weight$", ".weight", name)


def adapter_pairs(header):
    names = set(header) - {"__metadata__"}
    pairs = {}
    used = set()
    for name in sorted(names):
        if not re.search(r"\.lora_A(?:\.[^.]+)?\.weight$", name):
            continue
        other = name.replace(".lora_A", ".lora_B")
        if other not in names:
            raise ValueError(f"Adapter is missing a LoRA B tensor for {name}")
        target = base_tensor_name(name)
        if target in pairs:
            raise ValueError("Multiple adapters target the same weight; merge them separately")
        pairs[target] = (name, other)
        used.update((name, other))
    if not pairs or used != names:
        raise ValueError("Adapter contains missing or unsupported trainable tensors")
    return pairs


def merge_safetensors(base_dir, adapter_dir, destination, progress=None, rows_per_chunk=128):
    """Copy base shards and patch W += scale*(B@A), keeping RAM bounded."""
    import torch
    from safetensors import safe_open

    base_dir, adapter_dir, destination = map(Path, (base_dir, adapter_dir, destination))
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = validate_adapter(config)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    adapter_header, _ = read_header(adapter_path)
    pairs = adapter_pairs(adapter_header)
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = sorted(path.name for path in base_dir.glob("*.safetensors"))
    if not shard_names:
        raise ValueError("The base model needs cached safetensors weights for GGUF export")
    root = base_dir.resolve()
    shards = []
    tensor_locations = {}
    for name in shard_names:
        path = (base_dir / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("The cached model has an invalid or missing safetensors shard")
        header, start = read_header(path)
        for tensor_name in header:
            if tensor_name != "__metadata__":
                if tensor_name in tensor_locations:
                    raise ValueError("Duplicate base tensor across safetensors shards")
                tensor_locations[tensor_name] = path.name
        shards.append((path, header, start))
    missing = set(pairs) - set(tensor_locations)
    if missing:
        raise ValueError("Adapter weights do not match this base model: " + next(iter(missing)))
    destination.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(path.stat().st_size for path, _, _ in shards)
    if shutil.disk_usage(destination).free < total_bytes * 2 + 512 * 1024**2:
        raise ValueError("Not enough free disk space for the merged model and GGUF output")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    completed = 0
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        for path, header, data_start in shards:
            output = destination / path.name
            if output.resolve() == path.resolve():
                raise ValueError("The merged model must be written to a separate directory")
            shutil.copyfile(path, output)
            with safe_open(path, framework="pt", device="cpu") as base, output.open("r+b") as handle:
                for name in sorted(set(pairs) & set(header)):
                    info = header[name]
                    if info["dtype"] not in ("F32", "F16", "BF16") or len(info["shape"]) != 2:
                        raise ValueError("Only floating-point linear weights can be merged")
                    a_name, b_name = pairs[name]
                    a = adapter.get_tensor(a_name).float()
                    b = adapter.get_tensor(b_name).float()
                    weight = base.get_tensor(name)
                    transposed = bool(config.get("fan_in_fan_out"))
                    expected = (a.shape[1], b.shape[0]) if transposed else (b.shape[0], a.shape[1])
                    if tuple(weight.shape) != expected or a.shape[0] != b.shape[1] or a.shape[0] != config["r"]:
                        raise ValueError(f"LoRA shape mismatch for {name}")
                    row_bytes = weight.shape[1] * weight.element_size()
                    if row_bytes * weight.shape[0] != info["data_offsets"][1] - info["data_offsets"][0]:
                        raise ValueError("Base tensor size does not match its safetensors metadata")
                    for row in range(0, weight.shape[0], rows_per_chunk):
                        end = min(row + rows_per_chunk, weight.shape[0])
                        delta = (a[:, row:end].T @ b.T) if transposed else (b[row:end] @ a)
                        merged = weight[row:end].float() + delta * scale
                        if not bool(torch.isfinite(merged).all()):
                            raise ValueError(f"Merging the adapter produced non-finite values for {name}")
                        result = merged.to(weight.dtype).contiguous()
                        if not bool(torch.isfinite(result).all()):
                            raise ValueError(f"Merged weights overflowed {weight.dtype} for {name}")
                        raw = result.view(torch.uint8).numpy().tobytes()
                        handle.seek(data_start + info["data_offsets"][0] + row * row_bytes)
                        handle.write(raw)
                        del delta, merged, result, raw
                    del a, b, weight
                    completed += 1
                    if progress:
                        progress(completed, len(pairs))
    # Copy configuration/tokenizer assets, never repository Python or a dataset.
    for source_root in (base_dir, adapter_dir):
        for path in source_root.iterdir():
            if not path.is_file() or path.name.startswith("adapter_"):
                continue
            if path.suffix in (".json", ".jinja", ".model", ".tiktoken") or path.name in ("merges.txt", "vocab.txt"):
                if path.name == "config.json" and source_root == adapter_dir:
                    continue
                shutil.copyfile(path, destination / path.name)
    return {"merged_weights": completed, "base_bytes": total_bytes}


def resolve_base(model_path):
    local = Path(model_path).expanduser()
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(model_path, local_files_only=True,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"])).resolve()
    except Exception as exc:
        raise RuntimeError("The base model is not cached. Enable model downloads for a training run first, then export GGUF.") from exc


def build_modelfile(filename, prompt, context):
    if Path(filename).name != filename or any(char in filename for char in '\\"\r\n'):
        raise ValueError("Invalid model filename")
    value = f"FROM ./{filename}\nPARAMETER num_ctx {int(context)}\n"
    included = not prompt or '"""' not in prompt
    if prompt and included:
        # Ollama strips quote delimiters but does not decode JSON escapes.
        value += 'SYSTEM """' + prompt + '"""\n'
    elif prompt:
        value += "# Set the exact system prompt from system-prompt.txt in your runtime.\n"
    return value, included


def write_companions(run_dir, final_path):
    run_dir, final_path = Path(run_dir), Path(final_path)
    output_dir = final_path.parent
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    policy_path = run_dir / "policy.json"
    prompt = ""
    if policy_path.is_file():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        prompt = policy["system_prompt"]
        shutil.copyfile(policy_path, output_dir / "policy.json")
        (output_dir / "system-prompt.txt").write_text(prompt, encoding="utf-8")
    modelfile, included = build_modelfile(final_path.name, prompt, config.get("max_seq_length", 2048))
    (output_dir / "Modelfile").write_text(modelfile, encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "Standalone GGUF with the trained adapter merged into the base model.\n"
        "Import the .gguf in a compatible local runtime, or use Ollama:\n"
        "  ollama create tuna-model -f Modelfile\n  ollama run tuna-model\n\n"
        "Tuna: the companion Modelfile sets the saved system prompt when the\n"
        "prompt can be represented in Modelfile syntax. Exact text is also in\n"
        "system-prompt.txt. GGUF alone does not enforce that prompt, and output\n"
        "adherence is probabilistic. Keep policy.json for the saved manifest.\n"
        + ("\nThis prompt contains a triple-quote delimiter. Set its exact contents\nfrom system-prompt.txt in your runtime; SYSTEM was omitted from Modelfile.\n" if not included else ""),
        encoding="utf-8")
    return included


def export_run(run_dir, output_type="q8_0"):
    if output_type not in FORMATS:
        raise ValueError("Choose q8_0 or f16 for GGUF export")
    started = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    adapter_dir = run_dir / "adapter"
    output_dir = (run_dir / "gguf").resolve()
    if not output_dir.is_relative_to(run_dir):
        raise ValueError("Invalid GGUF output directory")
    output_dir.mkdir(exist_ok=True)
    scratch = (output_dir / ("merge-work-" + uuid.uuid4().hex)).resolve()
    if not scratch.is_relative_to(output_dir):
        raise ValueError("Invalid merge workspace")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    model_path = config.get("model_path", config.get("model"))
    # A GGUF source is restored to an immutable local training checkpoint.
    # Merge into that exact checkpoint, preserving the source's quantization.
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_path = summary.get("model_path", model_path)
    base_dir = resolve_base(model_path)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(config.get("name", "model"))).strip(".-").lower()[:80] or "model"
    filename = f"{slug}.{output_type.upper()}.gguf"
    final_path = output_dir / filename
    partial = output_dir / (filename + ".partial")
    converter = Path(__file__).resolve().parent / "vendor" / "llama_cpp" / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise RuntimeError("The bundled llama.cpp converter is missing")
    emit("export_progress", stage="merging", message="Merging trained adapters into cached base weights in small CPU chunks.")
    try:
        merged = merge_safetensors(base_dir, adapter_dir, scratch,
            progress=lambda done, total: emit("export_progress", stage="merging", completed=done, total=total,
                message=f"Merged {done} / {total} trained weight matrices."))
        gc.collect()
        emit("export_progress", stage="converting", message=f"Converting the merged model to {output_type.upper()} GGUF with llama.cpp.")
        sys.path.insert(0, str(converter.parent))
        old_argv = sys.argv
        sys.argv = [str(converter), str(scratch), "--outfile", str(partial), "--outtype", output_type,
                    "--use-temp-file", "--model-name", str(config.get("name", "Tuna model"))]
        try:
            try:
                runpy.run_path(str(converter), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise RuntimeError(f"llama.cpp conversion failed with exit code {exc.code}") from exc
        finally:
            sys.argv = old_argv
        emit("export_progress", stage="validating", message="Checking the GGUF tensor table and computing a file checksum.")
        import gguf
        reader = gguf.GGUFReader(partial)
        tensor_count = len(reader.tensors)
        if not tensor_count or "general.architecture" not in reader.fields:
            raise ValueError("The converted GGUF has no valid model tensors")
        architecture = reader.get_field("general.architecture").contents()
        del reader
        gc.collect()
        with partial.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ValueError("Invalid GGUF file header")
        digest = sha256_file(partial)
        partial.replace(final_path)
        prompt_included = write_companions(run_dir, final_path)
        result = {"filename": filename, "format": output_type, "size_bytes": final_path.stat().st_size,
                  "sha256": digest, "tensor_count": tensor_count, "architecture": architecture,
                  "merged_weights": merged["merged_weights"], "converter_commit": SOURCE_COMMIT,
                  "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
                  "source_gguf_sha256": summary.get("gguf_import", {}).get("source_sha256") if summary_path.is_file() and summary.get("gguf_import") else None,
                  "modelfile_has_system_prompt": prompt_included,
                  "elapsed_seconds": time.perf_counter() - started}
        (output_dir / "export_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit("export_complete", **result)
        return result
    finally:
        # Only this worker's verified, generated scratch directory is removed.
        if scratch.is_relative_to(output_dir) and scratch.name.startswith("merge-work-") and scratch.exists():
            try:
                shutil.rmtree(scratch)
            except OSError:
                emit("export_log", message="Temporary weight files will be cleaned after the export process releases its file mappings.")
        if partial.is_file() and partial.parent.resolve() == output_dir:
            partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--format", choices=sorted(FORMATS), default="q8_0")
    args = parser.parse_args()
    try:
        export_run(args.run_dir, args.format)
        return 0
    except Exception as exc:
        emit("export_error", message=str(exc))
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    from app_runtime import use_app_environment
    use_app_environment(__file__)
    raise SystemExit(main())
