# Chat inference performance

Measured locally on 23 October 2026 with a saved QLoRA adapter for
`Qwen/Qwen3.5-2B`, an RTX 5070, and the same greedy code-generation prompt.
Each of three trials per mode produced 128 tokens from a 33-token input.

| Inference path | Median tokens/s | Median generation time | Median API round trip |
| --- | ---: | ---: | ---: |
| Previous NF4 base + separate PEFT adapter | 8.56 | 14.95 s | 15.31 s |
| Auto BF16 + merged adapter + narrow convolution optimization | 17.44 | 7.34 s | 7.79 s |
| Above + optional fused linear-attention kernels | 19.01 | 6.73 s | 7.20 s |
| Above + compiled static-cache decoding | 56.49 | 2.27 s | 2.66 s |

The measured median generation rate improved **6.60×**. The final path ranged
from 54.96 to 57.64 tokens/s. First-token generation latency was 0.07–0.15 s
after model/kernel preparation. Its peak PyTorch allocated memory was about
3,931 MiB; this is process-local tensor memory, not whole-device VRAM.

## Changes

- Small models use BF16/FP16 when free GPU memory and local checkpoint metadata
  indicate enough headroom. Dense adapters are merged in memory once. Larger or
  unknown checkpoints retain NF4, with an OOM fallback during dense loading.
- Qwen3.5's supported four-value, single-token convolution uses a direct dot
  product and preserves its cache allocation. Other shapes/architectures keep
  the original implementation.
- Optional local FLA/Triton kernels replace recurrent linear-attention fallback
  operations. CPU models retain CPU references. A failed fused-kernel warmup
  restores reference linear attention for that loaded model.
- Supported dense Qwen3.5 models use compiled decoding and a static cache with
  the fixed chat context capacity. This reduces repeated GPU dispatch overhead.
  A failed compile falls back to ordinary decoding. Initial preparation may take
  several minutes and is excluded from the generation measurements above.
- Loading includes forced three-token warmups so compilation and cached decode
  run before chat is ready. Active UI polling is 100 ms. Replies show speed and
  first-token timing.

## Verification and scope

**77 Python tests passed**, including policy/context validation, cancellation,
worker lifecycle, offline precision selection and CPU execution with optional
GPU packages installed. Eight actual CUDA numerical checks compared the fused
recurrent/prefill kernels and optimized convolution against upstream reference
calculations. Cache updates and allocation identity were checked. All passed
the recorded numerical tolerances.

A real Tuna adapter also passed compiled-runtime checks for saved policy binding,
caller-role/control-token rejection, context limits without truncation, fresh
cache state across an A/B/A prompt sequence, and cancellation. Unloading removed
the entire compiled worker process tree. See
[fast-chat-runtime-verification.json](fast-chat-runtime-verification.json).

The three compiled outputs exactly matched the tested uncompiled BF16 outputs.
Repeated calls used fresh attention state with the same static capacity. The
benchmark reuses the same saved adapter and leaves checkpoint files alone.
BF16 and NF4 have different base-weight precision and can generate different
answers. This validates software execution and this workload's timing, not
held-out quality or exact output equivalence across precisions. Loading and
first-time compilation are excluded from generation timing; GPU clock state,
prompt length, model architecture and other applications affect results.

This is **not an Unsloth comparison** or a training-speed measurement. No claim
of lower VRAM or faster fine-tuning than Unsloth follows from these results.

Machine-readable measurements: [chat-speed-verification.json](chat-speed-verification.json)
and [fast-kernel-verification.json](fast-kernel-verification.json).

Setup and memory controls: [CHAT.md](CHAT.md).

To repeat the workload, load a completed model in Chat, leave it ready and run:

```powershell
python tools/benchmark_chat.py --output data/benchmarks/chat-speed.json
```

This uses the local API with a fixed public prompt, greedy decoding, 128 output
tokens and three trials. It does not change browser conversation history. Other
chat requests should wait until the benchmark completes.

The optional implementations come from [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
and [Triton Windows](https://github.com/triton-lang/triton-windows). Adapter merging
uses [PEFT's supported merge operation](https://huggingface.co/docs/peft/package_reference/lora).
Static-cache compilation uses [Transformers' inference optimization API](https://huggingface.co/docs/transformers/main/en/llm_optims).
