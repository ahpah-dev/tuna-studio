# Verification record

Verified locally on 22 October 2026.

The user flow is: browser dataset import → local JSON API → persisted JSONL →
run configuration → real Python training process → adapter export and measured
metrics in the browser.

## Checks completed

- 44 Python tests: 22 server/API tests, 16 training-core tests, and six GGUF merge/format tests.
- JavaScript syntax checked with Node.
- All local HTML, CSS, JavaScript and logo assets served successfully.
- Browser invalid-JSONL validation returned a line-specific error.
- Browser sample import saved four conversations and displayed their stored preview.
- Browser-created LoRA run completed two actual optimizer updates on a tiny,
  locally initialized Llama model. The dashboard showed measured loss, throughput,
  elapsed time, logs and an available adapter ZIP.
- Independent CPU worker smoke completed three optimizer updates with gradient
  accumulation and checkpointing, verified nonzero LoRA adapter weights and
  changed logits after adapter reload.
- Policy tests covered canonical prompt replacement, rejected caller system and
  developer messages, manifest hash validation and preservation of policy during
  right truncation.
- Desktop and narrow browser layouts inspected. No JavaScript console errors
  occurred during the dataset and training flow.

Test fixtures and run state were isolated under the task's `work/` directory.
They are not presented as useful trained models or competitive benchmarks.

## Environment and limits

NVIDIA GeForce RTX 5070 detected, approximately 12 GB VRAM. The installed PyTorch
2.14.1 runtime was a CPU build and reported CUDA unavailable. CPU training used
Transformers 5.20.2, PEFT 0.21.2 and Accelerate 1.15.0.

The initial CPU-only setup was subsequently repaired using an isolated app-local environment with PyTorch 2.14.1+cu130 and CUDA 13.0. The app now detects the RTX 5070 and reports LoRA, QLoRA, and Tuna ready.

The CUDA follow-up verified matrix multiplication/backpropagation, NF4 quantization, and three real GPU training updates each for QLoRA and Tuna through the local API. Both adapters had nonzero learned weights and passed ZIP integrity checks. Tuna GPU inference and caller-system-message rejection passed. Checkpointing, gradient accumulation and paged AdamW8bit were enabled.

There is no measured Unsloth comparison, validation-quality evaluation or adversarial policy benchmark.
The small random-model tests validate software execution, not model quality.

See [TRAINING.md](TRAINING.md) for supported models, metric definitions,
environment setup, method behavior and remaining limitations.

## CUDA runtime launch repair

`setup-cuda.ps1` recreates the CUDA environment from official package sources.
`app_runtime.py` makes direct `python server.py` and `python trainer.py` launches
use that environment. The status probe synchronizes a real CUDA kernel before
marking GPU training ready. Its import timeout accommodates first-load Windows
DLL checks, and readiness is cached for ten minutes.

## Model download repair

The app previously forced every worker offline, causing uncached Hugging Face
model IDs to fail with a misleading connection error. Model/tokenizer downloads
can now be enabled per run or through Settings, and are enabled by default in
the app. Offline mode remains available. The server propagates the selected
permission to both worker environment flags and Transformers loading options.
Training data remains local, and remote repository Python code remains disabled.

Regression checks cover persisted preferences, both access modes, and permission
propagation into an actual worker subprocess.

## GGUF export verification

GGUF merge tests check exact LoRA addition for F32/F16/BF16 tensors and both
linear weight layouts, unchanged base files, incompatible adapters, corrupt
headers and Modelfile prompt representation. API tests cover real subprocess
transport, progress, cancellation, idempotency, download gating and companion
files. GGUF artifacts are excluded from ordinary adapter ZIPs.

The actual completed 2B Tuna model was exported to a standalone Q8_0 GGUF:
381 tensors, 294 merged weight matrices, 2,679,710,624 bytes. GGUF metadata
and checksum checks passed. An official Windows CPU llama.cpp b11669 CLI loaded
the file and completed a single-token inference run with exit code zero.
The real download headers, companion ZIP, exact saved policy representation,
and removal of temporary merged shards were also verified. See
`gguf-verification.json`. These checks do not evaluate model quality.

## GGUF input and Tuna follow-up

The full browser → GGUF import → CUDA Tuna training → standalone GGUF download
flow passed on the actual Dad model. All 294 adapted matrices were verified in
the resulting Q8_0 file, and native llama.cpp inference passed. The source
tokenizer was preserved, including MiniCPM5 numeric splitting and BOS. Training
examples scored with/without the learned adapter showed an 18.77% target-loss
reduction; no held-out quality claim is made. The current suite has 50 passing
tests. See [the detailed report](GGUF-TRAINING-VERIFICATION.md).

## Local model chat

The Chat page now runs completed fine-tunes locally, streams replies, keeps
conversation history and provides stop/unload controls. Real Qwen3.5 adapter
and Dad GGUF conversations passed, including multi-turn recall and policy
validation. Model unloading released the app's Windows interpreter process tree.
The suite now has 61 passing tests. See [the chat report](CHAT-VERIFICATION.md).

## Faster chat inference

The suite now has 77 passing tests. Memory-aware dense inference, in-memory
adapter merging, optional fused Qwen3.5 kernels and compiled static-cache decode
passed real RTX 5070 checks. Median Qwen3.5-2B generation improved from 8.56 to
56.49 tokens/s on the recorded 128-token workload. Real Tuna policy/context
checks, cache isolation, cancellation and process-tree unloading also passed.
See [the performance report](CHAT-PERFORMANCE.md) for measurements, setup costs,
memory tradeoffs and limitations. No Unsloth comparison was performed.
