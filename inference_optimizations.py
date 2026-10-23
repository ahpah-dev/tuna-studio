"""Small, model-local inference optimizations with reference fallbacks.

These do not change checkpoint weights or training. Qwen3.5's reference cached
decode performs a grouped convolution over four values per channel. An explicit
dot product avoids dispatching a general convolution for that narrow case.
"""
from __future__ import annotations

import functools
import inspect
from types import FunctionType, MethodType


def _make_single_token_conv_update(original):
    import torch
    import torch.nn.functional as F

    @functools.wraps(original)
    def update(hidden_states, conv_state, weight, bias=None, activation=None, **kwargs):
        # Keep the reference implementation for prefill, training, other model
        # layouts, mixed dtypes, and future kernel arguments. No tensor values
        # are read on the CPU, so these guards cannot synchronize CUDA.
        supported = (
            not kwargs
            and not torch.is_grad_enabled()
            and activation in (None, "silu", "swish")
            and hidden_states.ndim == 3
            and conv_state.ndim == 3
            and weight.ndim == 2
            and hidden_states.shape[-1] == 1
            and conv_state.shape[-1] == weight.shape[-1] == 4
            and hidden_states.shape[:2] == conv_state.shape[:2]
            and hidden_states.shape[1] == weight.shape[0]
            and hidden_states.shape[0] > 0
            and hidden_states.shape[1] > 0
            and hidden_states.device == conv_state.device == weight.device
            and hidden_states.dtype == conv_state.dtype == weight.dtype
            and hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and conv_state.is_contiguous()
            and weight.is_contiguous()
            and (bias is None or (
                bias.ndim == 1
                and bias.shape[0] == weight.shape[0]
                and bias.device == weight.device
                and bias.dtype == weight.dtype
            ))
        )
        if not supported:
            return original(hidden_states, conv_state, weight, bias, activation, **kwargs)

        # This is the same last convolution window as the reference's
        # cat([state, token]) followed by conv1d(...)[..., -1:]. Copy into the
        # existing cache allocation; replacing it would break static addresses.
        window = torch.cat((conv_state[..., 1:], hidden_states), dim=-1)
        conv_state.copy_(window)
        # cuDNN accumulates FP16/BF16 convolution products in FP32. Round once
        # before applying the activation, just as conv1d does.
        result = (window.float() * weight.float().unsqueeze(0)).sum(dim=-1, keepdim=True)
        if bias is not None:
            result = result + bias.float()[None, :, None]
        result = result.to(hidden_states.dtype)
        return F.silu(result) if activation is not None else result

    return update


def _uses_reference_conv(function):
    """Avoid replacing an available upstream fused implementation."""
    current = function
    while inspect.isfunction(current):
        closed = inspect.getclosurevars(current).nonlocals
        if closed.get("is_new_implementation", False):
            return False
        current = getattr(current, "__wrapped__", None)
    raw = inspect.unwrap(function)
    return (
        inspect.isfunction(raw)
        and raw.__module__ == "transformers.models.qwen3_5.modeling_qwen3_5"
        and raw.__name__ == "causal_conv1d_update"
        and "conv1d" in raw.__code__.co_names
    )


def _uses_fused_implementation(function):
    current = function
    while inspect.isfunction(current):
        if inspect.getclosurevars(current).nonlocals.get("is_new_implementation", False):
            return True
        current = getattr(current, "__wrapped__", None)
    return False


def install_optimizations(model, *, allow_fused: bool = True) -> list[str]:
    """Optimize supported mixers on this model; return human-readable labels.

    No Transformers globals/classes are changed. Unsupported versions and
    existing fused kernels keep their original forward methods. This function
    is idempotent for the same mode and safe on other architectures. Calling
    again with allow_fused=False replaces optional linear-attention kernels
    with references on this model, retaining the convolution optimization.
    """
    conv_count, cpu_count, fused_count, fallback_count = 0, 0, 0, 0
    for mixer in model.modules():
        cls = type(mixer)
        if (
            cls.__name__ != "Qwen3_5GatedDeltaNet"
            or cls.__module__ != "transformers.models.qwen3_5.modeling_qwen3_5"
            or (getattr(mixer, "_tuna_inference_optimizations", False)
                and getattr(mixer, "_tuna_allow_fused", True) == allow_fused)
        ):
            continue
        # Accelerate may own the outer forward. Preserve it and replace only
        # the bound original in that case, retaining its device/offload hooks.
        target = "_old_forward" if hasattr(mixer, "_hf_hook") and hasattr(mixer, "_old_forward") else "forward"
        bound = getattr(mixer, target)
        if not inspect.ismethod(bound):
            continue
        raw = inspect.unwrap(bound.__func__)
        if (
            not inspect.isfunction(raw)
            or raw.__module__ != cls.__module__
            or "causal_conv1d_update" not in raw.__code__.co_names
        ):
            continue
        original = raw.__globals__.get("causal_conv1d_update")
        if not callable(original):
            continue
        from transformers.integrations.accelerate import force_accelerate_hooks

        namespace = dict(raw.__globals__)
        cuda = mixer.conv1d.weight.device.type == "cuda"
        rules = ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule")
        fused = allow_fused and cuda and all(_uses_fused_implementation(namespace.get(name)) for name in rules)
        if fused:
            fused_count += 1
        if not cuda or not allow_fused:
            # Optional FLA kernels are selected by Transformers at import time,
            # even for a model on CPU. Keep this model's CPU forwards on the
            # upstream Torch references without changing CUDA models/globals.
            reference_names = (*rules, "causal_conv1d_fn", "causal_conv1d_update") if not cuda else rules
            for name in reference_names:
                function = namespace.get(name)
                reference = inspect.unwrap(function) if callable(function) else None
                if inspect.isfunction(reference) and reference.__module__ == cls.__module__:
                    namespace[name] = reference
            if not cuda:
                cpu_count += 1
            else:
                fallback_count += 1
            original = namespace["causal_conv1d_update"]
        conv_applied = getattr(mixer, "conv_kernel_size", None) == 4 and _uses_reference_conv(original)
        if conv_applied:
            namespace["causal_conv1d_update"] = _make_single_token_conv_update(original)
            conv_count += 1
        if cuda and not conv_applied and allow_fused:
            mixer._tuna_inference_optimizations = True
            mixer._tuna_allow_fused = allow_fused
            continue
        local_forward = FunctionType(raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__)
        local_forward.__kwdefaults__ = raw.__kwdefaults__
        functools.update_wrapper(local_forward, raw)
        # This is the upstream mixer decorator, needed because it accesses
        # conv1d.weight directly instead of calling the child module's forward.
        local_forward = force_accelerate_hooks("conv1d")(local_forward)
        setattr(mixer, target, MethodType(local_forward, mixer))
        mixer._tuna_inference_optimizations = True
        mixer._tuna_allow_fused = allow_fused
        mixer._tuna_single_token_conv = conv_applied
    labels = []
    if conv_count:
        labels.append(f"Qwen3.5 single-token convolution ({conv_count} layers)")
    if fused_count:
        labels.append("Qwen3.5 fused linear attention")
    if cpu_count:
        labels.append(f"Qwen3.5 CPU reference linear attention ({cpu_count} layers)")
    if fallback_count:
        labels.append("Qwen3.5 reference linear attention")
    return labels
