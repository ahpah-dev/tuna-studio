# Training with Tuna Studio

Tuna Studio runs a real local Hugging Face / PEFT training worker. The dashboard
does not simulate losses or fill charts with synthetic benchmarks. A valid local
model, dataset and compatible runtime are necessary before a run can start.

## Supported methods

| Method | Base weights | Trainable weights | Runtime |
| --- | --- | --- | --- |
| LoRA | BF16/FP16 on CUDA, FP32 on CPU | LoRA adapters on all linear layers | PyTorch + Transformers + PEFT |
| QLoRA | 4-bit NF4, nested/double quantization | LoRA adapters on all linear layers | CUDA + bitsandbytes |
| Tuna | Same 4-bit recipe as QLoRA | Adapters trained on policy-conditioned examples | CUDA + bitsandbytes |

Tuna replaces existing dataset system messages with one canonical system prompt
before tokenization. This teaches the model to respond in that policy's context.
It exports the prompt and its SHA-256 in `policy.json`, plus a local serving
wrapper that rejects caller `system` and `developer` messages and inserts the
saved policy for each request. The wrapper also rejects reserved model control
tokens supplied in caller content, and refuses requests that exceed the model
context rather than truncating the policy.

**No fine-tuning method can promise an unoverrideable system prompt in model
weights.** Training adherence is probabilistic. The wrapper enforces prompt
assembly, not the meaning of generated output. Prompt injection or adversarial
inputs can still induce noncompliant answers. Direct weight access, changing
the export or calling the model outside this wrapper bypasses its controls.
The manifest hash detects accidental changes; it is not a cryptographic trust
boundary against an operator who can rewrite the manifest.

## Environment setup

Use an isolated Python environment. The local dashboard server and trainer must
run with the same interpreter; `sys.executable` launches the training worker.
Python 3.11 or 3.12 is a practical starting point when GPU wheel compatibility is
uncertain. Choose the current CUDA-enabled PyTorch build for your OS and driver
using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

On Windows with a CUDA 13.0-compatible NVIDIA driver, `./setup-cuda.ps1` installs the app-local CUDA environment. The launcher and direct server/trainer commands automatically use that `.venv`. A separate CUDA Toolkit installation is not required for the prebuilt PyTorch wheels.

Manual PowerShell setup, from the app directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install the appropriate CUDA-enabled torch command from pytorch.org first.
python -m pip install -r requirements-train.txt
python -c "import torch; print(torch.__version__); print('CUDA ready:', torch.cuda.is_available())"
```

A GPU listed by `nvidia-smi` does not imply that this Python environment has a
CUDA build of PyTorch. If the command reports a `+cpu` build or `False`, LoRA can
run on CPU, but QLoRA and Tuna cannot run until the runtime is configured.
There is no automatic model download or heavyweight package installation.

Use a local model directory or a Hugging Face model ID. When **Download missing models** is enabled (`offline: false`), the worker downloads missing model and tokenizer files into the local Hugging Face cache. Disabling downloads (`offline: true`) enforces `local_files_only=True` and offline environment flags. Both modes keep datasets, system prompts, training, and adapter exports local. Model licenses and access requirements still apply. `trust_remote_code=False` remains enforced; models requiring custom repository Python code are unsupported. Choose a text causal language model with SDPA support,
a fast tokenizer and a native chat template.

You can also select a full local `.gguf` file path. Llama-family GGUF models are
restored into disk-backed training checkpoints before LoRA/QLoRA/Tuna loading.
See [GGUF.md](GGUF.md) for input support, disk requirements, continued training
from an existing export, and saving the tuned result as a new GGUF.

## Dataset formats

Upload UTF-8 `.jsonl`, one object per line. Blank lines and a leading UTF-8 BOM
are supported. The dashboard accepts these row formats:

```jsonl
{"messages":[{"role":"user","content":"What is Tuna?"},{"role":"assistant","content":"A local fine-tuning workspace."}]}
{"prompt":"What is Tuna?","completion":"A local fine-tuning workspace."}
```

The direct worker CLI additionally accepts `instruction`/`input`/`output` rows;
convert them to either format above before uploading through the dashboard:

```jsonl
{"instruction":"Explain this term.","input":"QLoRA","output":"Quantized low-rank adaptation."}
```

Message conversations must alternate user and assistant and end in an assistant
reply. One leading system message is allowed for standard methods. Tuna removes
dataset system messages and inserts the canonical policy. Text-only content is
supported; tool calls, images and multipart message content are not supported.

Each assistant turn becomes one example with the preceding conversation as
context. Only tokens in that assistant reply receive loss; system, user,
previous assistant and padding tokens are masked. A native tokenizer's character
offsets prevent a token spanning the prompt/reply boundary from accidentally
supervising prompt text. A tokenizer whose generation prefix does not match its
training prefix is rejected with an actionable message; some reasoning-specific
templates need a compatible non-reasoning template before use.

Tokenization truncates from the right, preserving the policy and prompt. An
assistant reply can be partially truncated. Replies with no remaining supervised
tokens are dropped and counted in the log and manifest. A run fails when none
remain. Increase sequence length or shorten prompts when this happens.

## Direct worker invocation

The dashboard writes a run-specific configuration and launches this same command:

```powershell
python trainer.py --config C:\path\to\config.json --output C:\path\to\run
```

Example configuration:

```json
{
  "model_path": "C:/models/my-cached-chat-model",
  "dataset_path": "C:/datasets/support.jsonl",
  "method": "tuna",
  "system_prompt": "You are a concise support assistant. Use the provided product facts and say when information is missing.",
  "max_seq_length": 1024,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 0.0002,
  "epochs": 1,
  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "gradient_checkpointing": true,
  "seed": 42,
  "max_steps": 0,
  "offline": true
}
```

`max_steps: 0` uses epochs; a positive number overrides epochs. Fractional epochs
are rounded up to whole optimizer updates. Runs are single-device. Distributed
training, checkpoint resume, automatic evaluation and distributed training are not implemented. Completed runs can merge adapter weights into a standalone Q8_0 or F16 GGUF through the run dialog; see [GGUF.md](GGUF.md). Cancellation may leave diagnostic logs but does
not produce a completed export. Keep a separate evaluation dataset and compare
task quality before deploying an adapter.

## Memory and speed choices

The worker uses SDPA attention, dynamic padding, length-grouped batches, disabled
KV caching and optional non-reentrant gradient checkpointing. QLoRA and Tuna
also use NF4 double quantization and paged 8-bit AdamW. LoRA uses PyTorch AdamW.
Gradients accumulate over microbatches with assistant-target-token weighting.
BF16 is selected when CUDA supports it; other CUDA devices use FP16 with gradient
scaling. CPU uses FP32. Gradient clipping rejects non-finite gradients.

Checkpointing lowers activation memory by recomputing during backpropagation; it
can cost time. Quantization and paging also have workload-specific overhead.
These techniques **do not establish lower VRAM or faster training than Unsloth**.
No custom GPU kernels are included and no superiority benchmark has been run.
Sequence packing is deliberately absent: sequences remain independent, preserving
the attention boundaries between separate conversations.

## Measured metrics and exports

The worker emits NDJSON events with `type` equal to `log`, `metric`, `complete`
or `error`. Each update records real loss, learning rate, epoch progress, elapsed
training time, processed non-padding tokens/second, supervised tokens/second and
CUDA peak allocated/reserved memory in MiB. Throughput includes forward/backward,
optimization and synchronization, and excludes model loading/tokenization/export.
Allocated/reserved CUDA peaks begin before model loading and are process-local;
they are not whole-device memory measurements. Both are `null` on CPU.

Each run saves:

- `adapter/`: PEFT adapter weights/config and the tokenizer. The original base
  model is still required; it is not bundled.
- `run_manifest.json`: configuration, dataset SHA-256, library versions, device,
  precision, optimizer, example counts and training parameter counts.
- `training_metrics.jsonl`: per-update measured metrics.
- `training_summary.json`: final measured metrics and output paths.
- Tuna only: `policy.json`, `tuna_runtime.py` and `serve.py`.

Run a Tuna export locally:

```powershell
python C:\path\to\run\serve.py --prompt "What can you help me with?"
```

For multi-turn requests, `--messages conversation.json` accepts an array of
alternating user/assistant objects ending in a user message. The base model path
or cached ID from the manifest must still be accessible. The serving wrapper uses
4-bit CUDA loading with local/cached model files only. Training downloads cache the base model for this offline serving path.

## Comparing against Unsloth

Measure both trainers on the same GPU/driver, exact base-model revision,
tokenizer/template, data split and ordering, precision/quantization, LoRA targets,
rank, context length, effective batch size, optimizer, update count and quality
target. Record cold-start and warmed training separately. Repeat trials after
warmup and report distributions, not a single best run. Compare peak allocated,
peak reserved and whole-device VRAM separately. Evaluate held-out quality and
Tuna policy adherence/adversarial robustness alongside speed. A smaller or less
accurate workload does not demonstrate a better trainer.

## Verification

The core tests require only Python's standard library:

```powershell
python -m unittest discover -s tests -p test_training_core.py -v
python -m py_compile trainer.py tuna_core.py
```

The test suite checks dataset normalization, policy binding, rejection of caller
system messages, line-specific parse failures, assistant-only masks including BPE
boundary cases, preservation of the system prefix during truncation, padding,
offline validation and batch coverage. End-to-end CUDA training requires the
actual CUDA runtime and model; passing core tests does not validate GPU kernels.

The current suite contains 16 core tests and 19 server tests. It includes checks that the model download permission reaches worker environment flags, and strict offline loading remains available. The original 15 core tests passed during implementation. An independent smoke run created
a tiny random Llama model and fast tokenizer entirely locally, ran three real
CPU LoRA updates with gradient checkpointing and accumulation, then verified
nonzero saved adapter weights and changed logits after adapter reload. That run
used PyTorch 2.14.1 (CPU), Transformers 5.20.2, PEFT 0.21.2 and Accelerate 1.15.0.
It validates the training/export path, not useful model quality or GPU performance.
A subsequent CUDA repair installed PyTorch 2.14.1+cu130 into the app-local environment. Real CUDA matrix multiplication/backpropagation and bitsandbytes NF4 quantization passed on the RTX 5070. QLoRA and Tuna each completed three GPU optimizer updates through the local API, with changed adapter weights, measured CUDA memory, and verified ZIP exports. Tuna GPU inference and rejection of caller system messages also passed. These small random-model tests verify execution; Unsloth comparisons and useful-model-quality evaluations remain unverified.

## Technical references

- [Hugging Face PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Hugging Face chat templates](https://huggingface.co/docs/transformers/main/chat_templating)
- [Hugging Face training](https://huggingface.co/docs/transformers/training)
- [bitsandbytes AdamW optimizers](https://huggingface.co/docs/bitsandbytes/v0.48.0/en/reference/optim/adamw)
- [PyTorch peak CUDA allocation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)
