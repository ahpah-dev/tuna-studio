# Tuna Studio

<img src="web/assets/tuna-mark.png" alt="Friendly teal Tuna Studio logo" width="112">

A friendly, local-first workspace for LoRA, QLoRA and policy-conditioned **Tuna**
fine-tuning. Bring a local model and JSONL examples, configure a run, follow real
training metrics, and export the adapter. Your datasets and runs stay in the
app's local `data/` folder.

## Start the app

Clone the source with Git, then launch with Python 3.10 or later:

```powershell
git clone https://github.com/ahpah-dev/tuna-studio.git
cd tuna-studio
python server.py
```

Open **http://127.0.0.1:8765**. Press **Ctrl+C** in the terminal to stop the
server and any active training run. The dashboard uses Python's standard library;
there is no npm build or package installation needed to explore it. When an app-local `.venv` exists, direct server and trainer launches automatically use it.

On Windows, this optional launcher starts the server in the background:

```powershell
.\start-tuna.ps1 -OpenBrowser
```

Omit `-OpenBrowser` to print the URL without opening a browser. The launcher uses
`.venv\Scripts\python.exe` when present, otherwise `python` on your PATH. It logs
to `data/logs/` and reports the server process ID. It reuses only a verified Tuna
Studio endpoint for this app's data directory. An occupied port belonging to
something else is left alone; use `-Port 8766` for a separate port. For convenient
terminal-based shutdown, use the foreground `python server.py` command above.

The server binds to **127.0.0.1** by default and supports loopback only. The UI
loads local assets without remote scripts, fonts or telemetry. Training stays local. The **Download missing models** option fetches model/tokenizer files from Hugging Face and caches them for reuse. Switch it off for strict offline loading. Your datasets and system prompts are never uploaded.

## Train a model

Read **[docs/TRAINING.md](docs/TRAINING.md)** for the CUDA environment setup,
supported model/tokenizer requirements, dataset examples and direct worker CLI.
GPU verification used an isolated environment with PyTorch 2.14.1+cu130 and CUDA
13.0 on an RTX 5070. Both QLoRA and Tuna passed GPU training and adapter-export
checks. Install the runtime on your own machine before starting GPU training.

To recreate the CUDA environment on a compatible Windows machine, run:

```powershell
.\setup-cuda.ps1
.\start-tuna.ps1
```

The installer uses official CUDA 13.0 PyTorch wheels and needs a compatible NVIDIA driver. It creates `.venv` inside the app and leaves global Python installations unchanged.

1. Add a UTF-8 `.jsonl` dataset in the Datasets page.
2. Create a run and select a local Llama-family `.gguf` file, model directory, or Hugging Face model ID. Allow downloads when a Hub model is not yet cached.
3. Choose LoRA, QLoRA or Tuna; set Tuna's canonical system policy when applicable.
4. Review the configuration, start training and follow measured loss, time,
   throughput and CUDA memory. Export the adapter when the run completes.

| Method | What it does | Runtime |
| --- | --- | --- |
| LoRA | Trains adapters on a frozen full-precision base | CPU or CUDA |
| QLoRA | Trains adapters on a frozen 4-bit NF4 base | CUDA + bitsandbytes |
| Tuna | QLoRA with a canonical policy in every example and an exported policy-bound serving wrapper | CUDA + bitsandbytes |

The app includes dataset validation and previews, local hardware/runtime checks,
assistant-only training loss, gradient accumulation and checkpointing, live run
logs, cancellation, persistent run history, adapter exports and a clean local UI.
Models, data, logs and generated artifacts are ordinary local files. Interface preferences and external baselines use local browser storage.

## What is verified—and what remains open

This is a working foundation for local fine-tuning. **No lower-VRAM or faster
training result than Unsloth has been established.** The worker uses established
memory-saving techniques; it does not include custom GPU kernels. Compare both
tools on matching hardware, workloads and model quality before making a claim.

**Tuna cannot make a system prompt unoverrideable in model weights.** Its wrapper
rejects caller system/developer messages and inserts the saved policy, but output
adherence remains probabilistic. Adversarial inputs may cause violations, and
direct model or filesystem access can bypass or alter the wrapper.

Validation during implementation:

- **85 unit tests passed**, covering the local server, training core, GGUF import/export, chat policy binding, UTF-8 transport, worker lifecycle, inference optimizations and reasoning/response budgets.
- A tiny random Llama model and fast tokenizer were created entirely locally.
  **Three real CPU LoRA updates** exercised gradient accumulation and
  checkpointing; saved nonzero adapter weights were reloaded and changed logits.
  Assistant-only labels, policy replacement/hash checks and right truncation
  also passed. This smoke run used PyTorch 2.14.1 CPU, Transformers 5.20.2,
  PEFT 0.21.2 and Accelerate 1.15.0; it is not a useful-model-quality benchmark.
- **CUDA verified on the RTX 5070:** QLoRA and Tuna each completed three actual GPU optimizer updates with checkpointing, gradient accumulation, NF4 double quantization and paged AdamW8bit. Adapter weights changed and ZIP exports passed integrity checks. Tuna GPU inference and caller-system-message rejection passed. The model was a tiny local random fixture; no useful-model-quality or Unsloth performance claim follows from it.

Re-run the standard-library suite with:

```powershell
python -m unittest discover -s tests -v
```

Runs are single-device and text-only. Resume, distributed training, automated
evaluation, and sequence packing are not implemented. Adapter exports require the original base model. See the training
guide for detailed behavior and benchmark methodology.

## Visual identity

The friendly teal tuna logo was generated with built-in imagegen. The final
[logo](web/assets/tuna-mark.png) and the exact [generation prompts](web/assets/brand-prompt.txt)
are included. See [verification notes](docs/VERIFICATION.md) for the full checked flow.

Model access is controlled per run and through Settings. Failed or stopped runs
include a Retry run action that restores their configuration for review. Downloads
do not enable execution of custom repository Python code.

## Chat with your fine-tune

Choose **Chat** in the sidebar, select a completed run and **Load model**. Or
open a run and choose **Chat with model**. Chat has streaming replies, saved
conversation history, code formatting, stop/unload controls and idle unloading.
Choose **Unlimited** to use the remaining chat context without a fixed reply
cap. Supported models also have a **Reasoning** switch with a separate,
collapsible thinking section. Both settings are saved in this browser.
Tuna is described as **Policy-conditioned QLoRA**; its saved policy is included
on every chat turn. See [docs/CHAT.md](docs/CHAT.md) for local model requirements
and memory behavior.

Chat selects faster BF16/FP16 inference when weights fit in GPU memory, merges
adapters in memory once, and keeps NF4 for larger models. Optional Qwen3.5 fast
kernels can be installed with `./setup-fast-chat.ps1` after CUDA setup. See
[chat performance](docs/CHAT-PERFORMANCE.md) for actual measurements and tradeoffs.
On the measured RTX 5070 / Qwen3.5-2B workload, median generation throughput
increased from 8.56 to 56.49 tokens/s after preparation. This is a local chat
before/after result; no Unsloth comparison or training-speed claim is implied.

## GGUF export

Completed runs include **Save as GGUF** with Q8_0 and F16 choices, progress,
cancellation and direct downloads. The standalone file merges your adapter into
the cached base model. Optional Ollama companion files carry the Tuna system
prompt. See [docs/GGUF.md](docs/GGUF.md) for usage, resources and limitations.

You can also **start from a local Llama-family GGUF**. Enter its full file path
or choose **Fine-tune this GGUF** in a completed export. The app restores training
weights locally, trains the selected method and merges the learned changes into
a new GGUF. The actual Dad model passed this complete flow on the RTX 5070,
including exact checks of all 294 adapted matrices and native llama.cpp inference.
See [the verification report](docs/GGUF-TRAINING-VERIFICATION.md).

## Repository contents

This repository contains the application source, logo, tests, setup scripts and
verification reports. Model weights, personal datasets, training runs, system
policies, chat history and Python environments are not included. The app creates
its own local `data/` directory on first launch; browser chat history stays in
that browser's local storage.

The dashboard and standard-library tests can run without downloading a model.
The tokenizer restoration test requires the optional ML dependencies. Training,
chat and GGUF conversion additionally require compatible local dependencies and
model files; GPU execution is verified separately from the core test suite.

The bundled llama.cpp conversion source retains its upstream MIT license and
recorded revision. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
