"""Local, disk-backed GGUF -> trainable HF checkpoint import.

GGUF blocks are inference weights. Import dequantizes them into F16 safetensors,
then the normal training worker loads that checkpoint with LoRA or NF4 QLoRA.
Only Llama-family GGUF is currently accepted. No model download is performed.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil
import struct
import uuid

from gguf_export import sha256_file

IMPORT_VERSION = 2
SUPPORTED_ARCHITECTURES = {"llama"}


def restore_tokenizer_details(tokenizer, reader, vocab_size):
    """Preserve llama.cpp pre-tokenization rather than silently using GPT-2.

    MiniCPM5 rules follow llama.cpp src/llama-vocab.cpp at the converter's
    recorded commit. Transformers 5.20's generic GGUF tokenizer omits them.
    """
    from tokenizers import Regex, Tokenizer, pre_tokenizers, processors
    from transformers.integrations.gguf.gguf_tokenizer_mapping import GGUF_PRE_TOKENIZER_SPLITS

    def field(name, default=None):
        item = reader.get_field(name)
        return item.contents() if item else default

    kind = field("tokenizer.ggml.model")
    pre = field("tokenizer.ggml.pre", "default")
    if kind == "gpt2":
        if pre == "minicpm5":
            tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
                pre_tokenizers.Split(Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ])
        elif pre not in GGUF_PRE_TOKENIZER_SPLITS and pre not in ("gpt-2", "phi-2"):
            raise ValueError(f"GGUF pre-tokenizer {pre!r} is not supported for training import. Use its original Hugging Face tokenizer/model directory.")
        if pre in ("minicpm5", "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "midm-2.0", "jina-v5-nano", "lfm2"):
            tokenizer.backend_tokenizer.model.ignore_merges = True

    if len(tokenizer) > vocab_size:
        # Generic BPE constructors may append an unused GPT-2 default token.
        if any(token >= vocab_size for token in tokenizer.all_special_ids):
            raise ValueError("GGUF tokenizer special tokens exceed the model vocabulary")
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        backend["added_tokens"] = [token for token in backend.get("added_tokens", []) if token["id"] < vocab_size]
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(backend))
        if len(tokenizer) > vocab_size:
            raise ValueError("GGUF tokenizer vocabulary exceeds the model vocabulary")
    add_bos = bool(field("tokenizer.ggml.add_bos_token", False))
    tokenizer.init_kwargs["add_bos_token"] = add_bos
    tokenizer.clean_up_tokenization_spaces = False
    if add_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("GGUF requests a BOS token but does not define one")
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{tokenizer.bos_token} $A", pair=f"{tokenizer.bos_token} $A {tokenizer.bos_token}:1 $B:1",
            special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id)])


def local_gguf_path(model_path):
    """Detect a file input without importing any training dependencies."""
    path = Path(model_path).expanduser()
    if path.suffix.lower() != ".gguf":
        return None
    if not path.is_file():
        raise ValueError("GGUF input must be an existing local .gguf file. Enter its full file path.")
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError("The selected file does not have a valid GGUF header.")
    return path.resolve()


def shard_plan(entries, max_bytes=256 * 1024**2):
    """Plan tensor shards without holding the tensors themselves in memory."""
    groups, group, size = [], [], 0
    for entry in entries:
        count = 1
        for dim in entry["shape"]:
            count *= dim
        item = {**entry, "bytes": count * (4 if entry["dtype"] == "F32" else 2)}
        if group and size + item["bytes"] > max_bytes:
            groups.append(group)
            group, size = [], 0
        group.append(item)
        size += item["bytes"]
    if group:
        groups.append(group)
    return groups


def write_shard(path, entries, chunks):
    """Stream a valid safetensors shard, checking every tensor's byte count."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for item in entries:
        header[item["name"]] = {"dtype": item["dtype"], "shape": item["shape"],
                                "data_offsets": [offset, offset + item["bytes"]]}
        offset += item["bytes"]
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for item in entries:
            count = 0
            for chunk in chunks(item):
                handle.write(chunk)
                count += len(chunk)
            if count != item["bytes"]:
                raise ValueError(f"GGUF tensor has an invalid size: {item['name']}")


def import_gguf(source, cache_dir, work_dir, progress=None):
    """Return an immutable cached training checkpoint and import provenance.

    Tensor dequantization is row-chunked, including large embeddings. Q/K RoPE
    permutations are reversed before writing, following Transformers' loader.
    The meta model is used only to validate names/shapes; no dense model is built.
    """
    import numpy as np
    import torch
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("GGUF import needs the gguf package. Install requirements-train.txt in the app environment and retry.") from exc
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint, LlamaTensorProcessor, get_gguf_hf_weights_map
    except ImportError as exc:
        raise RuntimeError("GGUF import requires the GGUF loader in Transformers 5.20 or later. Update requirements-train.txt.") from exc

    source = local_gguf_path(str(source))
    if source is None:
        raise ValueError("Select a local .gguf model file")
    cache_dir, work_dir = Path(cache_dir).resolve(), Path(work_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Reading and hashing the local GGUF model.")
    fingerprint = sha256_file(source)
    destination = cache_dir / f"gguf-v{IMPORT_VERSION}-{fingerprint}"
    manifest_path = destination / "gguf_import.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = destination / "model.safetensors.index.json"
        if manifest.get("source_sha256") == fingerprint and index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("weight_map") and all((destination / filename).is_file() for filename in set(index["weight_map"].values())):
                if progress:
                    progress("Using the previously imported GGUF training weights.")
                return destination, manifest
        raise ValueError("The cached GGUF import is incomplete. Remove that model cache entry and retry.")

    reader = gguf.GGUFReader(source)
    architecture = reader.get_field("general.architecture").contents()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"GGUF training import currently supports Llama-family architecture; this file uses {architecture}. Use its Hugging Face weights for training instead.")
    split_count = reader.get_field("split.count")
    if split_count and split_count.contents() != 1:
        raise ValueError("Split GGUF files must be combined into a single GGUF before training import.")
    parsed = load_gguf_checkpoint(str(source), return_tensors=False)
    values = parsed["config"]
    model_type = values.pop("model_type")
    if model_type not in ("llama", "mistral"):
        raise ValueError(f"Unsupported GGUF training model type: {model_type}")
    for key, low, high in (("num_hidden_layers",1,512),("hidden_size",8,32768),
                           ("vocab_size",8,2000000),("num_attention_heads",1,1024)):
        if type(values.get(key)) is not int or not low <= values[key] <= high:
            raise ValueError(f"GGUF has unsupported or missing {key} metadata")
    config = AutoConfig.for_model(model_type, **values)
    config.architectures = ["LlamaForCausalLM" if model_type == "llama" else "MistralForCausalLM"]
    config.dtype = torch.float16
    with init_empty_weights():
        meta = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    processor = LlamaTensorProcessor(config=config.to_dict())
    mapping = get_gguf_hf_weights_map(meta, processor)
    expected = {name: list(tensor.shape) for name, tensor in meta.state_dict().items()}
    source_tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(source_tensors) != len(reader.tensors):
        raise ValueError("GGUF contains duplicate tensor names")
    entries, mapped = [], set()
    for tensor in reader.tensors:
        if tensor.name not in mapping:
            raise ValueError(f"Unrecognized GGUF tensor; refusing partial training import: {tensor.name}")
        name = mapping[tensor.name]
        shape = list(reversed(tensor.shape.tolist()))
        if shape != expected.get(name) or name in mapped:
            raise ValueError(f"GGUF tensor shape or name does not match its model configuration: {name}")
        entries.append({"name": name, "source": tensor.name, "shape": shape,
                        "dtype": "F32" if len(shape) == 1 else "F16"})
        mapped.add(name)
    missing = set(expected) - mapped
    if config.tie_word_embeddings and "model.embed_tokens.weight" in mapped:
        missing.discard("lm_head.weight")
    if missing:
        raise ValueError("GGUF is missing required model weights: " + next(iter(missing)))
    groups = shard_plan(entries)
    total_bytes = sum(item["bytes"] for group in groups for item in group)
    if shutil.disk_usage(cache_dir).free < total_bytes + 512 * 1024**2:
        raise ValueError(f"Not enough disk space to import this GGUF. Need approximately {total_bytes / 1024**3:.1f} GiB plus a reserve for training weights.")
    del meta
    scratch = work_dir / ("gguf-import-work-" + uuid.uuid4().hex)
    scratch.mkdir()
    try:
        if progress:
            progress("Restoring the GGUF tokenizer and chat template locally.")
        tokenizer = AutoTokenizer.from_pretrained(str(source.parent), gguf_file=source.name,
            local_files_only=True, trust_remote_code=False, use_fast=True)
        if not tokenizer.is_fast or not tokenizer.chat_template:
            raise ValueError("The GGUF needs a fast tokenizer and a native chat template for assistant-only training. Use its complete Hugging Face model directory.")
        restore_tokenizer_details(tokenizer, reader, config.vocab_size)
        tokenizer.save_pretrained(scratch)
        config.save_pretrained(scratch)
        index, completed = {}, 0

        def chunks(item):
            tensor = source_tensors[item["source"]]
            dtype = np.float32 if item["dtype"] == "F32" else np.float16
            # Q/K reversal changes the first-dimension row order. Reorder row
            # indices, then dequantize small slices instead of the whole tensor.
            rows = item["shape"][0]
            order = np.arange(rows)
            if ".attn_q." in tensor.name or ".attn_k." in tensor.name:
                heads = config.num_attention_heads if ".attn_q." in tensor.name else config.num_key_value_heads
                order = processor._reverse_permute_weights(order, heads, heads)
            for start in range(0, rows, 128):
                data = tensor.data[order[start:start+128]]
                weights = gguf.dequantize(data, tensor.tensor_type)
                with np.errstate(over="ignore", invalid="ignore"):
                    value = weights.astype(dtype)
                if not np.isfinite(value).all():
                    raise ValueError(f"GGUF tensor cannot be represented as finite training weights: {tensor.name}")
                yield value.tobytes(order="C")

        for number, group in enumerate(groups, 1):
            filename = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
            write_shard(scratch / filename, group, chunks)
            index.update({item["name"]: filename for item in group})
            completed += len(group)
            if progress:
                progress(f"Imported {completed} / {len(entries)} GGUF tensors into training weights.")
        (scratch / "model.safetensors.index.json").write_text(json.dumps({"metadata":{"total_size":total_bytes},"weight_map":index},indent=2), encoding="utf-8")
        manifest = {"schema_version":IMPORT_VERSION,"source_path":str(source),
                    "source_sha256":fingerprint,"architecture":architecture,
                    "tensor_count":len(entries),"training_weight_bytes":total_bytes,
                    "training_dtype":"F16 matrices / F32 vectors",
                    "note":"Imported quantized GGUF; original pre-quantization precision cannot be recovered."}
        (scratch / "gguf_import.json").write_text(json.dumps(manifest,indent=2), encoding="utf-8")
        scratch.replace(destination)
        return destination, manifest
    finally:
        del source_tensors, reader
        gc.collect()
        if scratch.exists() and scratch.parent == work_dir and scratch.name.startswith("gguf-import-work-"):
            shutil.rmtree(scratch)


def clean_import_work(run_dir):
    """Clean a stopped worker's generated scratch files after process exit."""
    root = Path(run_dir).resolve()
    for path in root.glob("gguf-import-work-*"):
        if path.is_dir() and not path.is_symlink() and path.resolve().is_relative_to(root):
            shutil.rmtree(path)
