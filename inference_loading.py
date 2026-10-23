"""Conservative, offline inference precision selection without importing torch."""
from __future__ import annotations

import json
from pathlib import Path

GIB = 1024 ** 3


def checkpoint_size(base_path):
    """Estimate dense weights from local metadata; never fetch missing files.

    Unknown or already-quantized checkpoints stay on the lower-memory path.
    The stored byte size is conservative for F32 checkpoints loaded as BF16.
    """
    directory = Path(base_path)
    if not directory.is_dir():
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(str(base_path), 'config.json')
            if not isinstance(cached, str):
                return None
            directory = Path(cached).parent
        except (ImportError, OSError, ValueError):
            return None
    try:
        config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        if config.get('quantization_config'):
            return None
        index = directory / 'model.safetensors.index.json'
        if index.is_file():
            value = json.loads(index.read_text(encoding='utf-8')).get('metadata', {}).get('total_size')
            return value if type(value) is int and value > 0 else None
        weights = directory / 'model.safetensors'
        return weights.stat().st_size if weights.is_file() else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def choose_precision(weight_bytes, free_bytes, total_bytes, *, cuda, bf16, preference='auto'):
    if preference not in ('auto', 'nf4'):
        raise ValueError('TUNA_CHAT_PRECISION must be auto or nf4.')
    if not cuda:
        return 'fp32'
    # Reserve space for the KV/recurrent caches, activations, adapter merge and
    # other desktop applications. An OOM still falls back to NF4 in the worker.
    reserve = max(1.5 * GIB, .15 * total_bytes)
    fits = weight_bytes is not None and weight_bytes > 0 and free_bytes >= 1.25 * weight_bytes + reserve
    return ('bf16' if bf16 else 'fp16') if preference == 'auto' and fits else 'nf4'
