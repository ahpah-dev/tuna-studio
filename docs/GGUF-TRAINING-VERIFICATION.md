# Verification report: Tuna training from GGUF

Story: open a saved GGUF → Fine-tune this GGUF → select Tuna and the saved
JSONL/policy → local GGUF import → CUDA training → export → download and run a
new standalone GGUF.

## Flow status

| Boundary | Status | Evidence |
| --- | --- | --- |
| UI entry | Passed | Fine-tune this GGUF populated the source path, existing nine-row dataset, policy and shown training settings. |
| Client → API | Passed | Starting the wizard created real run `901593c2a7924f399dcf9d5c947c3f77`, named Dad GGUF Tuna. |
| GGUF → training checkpoint | Passed | All 381 source tensors mapped to F16/F32 safetensors; original source SHA-256 matched the prior export. The small fixture's 12 tensors matched the official Transformers GGUF loader. |
| Tokenizer | Passed | Fourteen encoding comparisons matched the original MiniCPM tokenizer, including BOS, numeric, multilingual and chat/control-token input. |
| CUDA training | Passed | Two optimizer updates on RTX 5070; 294/294 LoRA B matrices contain learned updates; 5,911.93 MiB peak allocated VRAM. |
| Learning check | Passed on training data | Same nine examples scored with adapters disabled/enabled: assistant-target loss 0.704356 → 0.572131, an 18.77% reduction. This is not held-out validation. |
| Adapter → GGUF | Passed | All 294 exported adapted matrices exactly match restored source weights plus LoRA deltas, followed by the correct Q/K permutation and Q8_0 quantization. All differ from this run's frozen training base. |
| Native inference | Passed | Official Windows llama.cpp b11669 loaded the new 381-tensor model and generated with exit code 0. |
| Response → UI | Passed | Completed state, learned-update confirmation, Download GGUF, Ollama files and Fine-tune this GGUF visible; browser console error list empty. |
| Download/files | Passed | Correct download length, exact companion policy, ZIP integrity, copied model checksum and scratch cleanup verified. |

## Issues fixed

- Existing GGUF paths previously reached the Hugging Face loader as model IDs.
  Local file validation and a bounded GGUF import now precede training.
- Transformers' generic tokenizer reconstruction omitted MiniCPM5 numeric split
  rules and its whole-vocabulary BPE matching behavior. The importer restores
  the llama.cpp rules, removes an unused synthetic token and preserves BOS.
  Unknown pre-tokenizers are rejected rather than silently substituted.
- Export previously resolved the run's source as a Hugging Face directory/ID.
  It now uses the exact restored checkpoint recorded by completed training.
- A run now checks for learned LoRA updates before reporting completion.

## Artifact and scope

`dad-gguf-tuna.Q8_0.gguf`: 2,679,710,368 bytes.

SHA-256: `36acb21b98817ebdeff54c57c25bf0116aa57c696c3a023df480b0c5640ab64c`.

The current input importer supports single-file Llama-family architecture with
a compatible tokenizer and native chat template. It dequantizes to a disk-backed
checkpoint, trains NF4 adapters, and re-quantizes merged weights. It does not
train directly on packed GGUF inference blocks or recover lost precision.

Fifty unit tests passed in the app environment; `uv pip check` reported all
46 installed packages compatible. An isolated CUDA GGUF round trip also passed
20 updates, a training-loss improvement, exact seven-matrix merge checks and
native inference. The earlier Dad GGUF passed an independent 21-matrix audit.

These checks establish correct software execution and learning on training
examples. They do not establish held-out task quality, an Unsloth performance
advantage or an unoverrideable system prompt. Tuna's policy must also be set in
the inference runtime, using the supplied companion files.

Machine-readable evidence: `gguf-training-verification.json`,
`tuna-gguf-weight-audit.json`, `tuna-gguf-retraining-audit.json`,
`dad-gguf-tuna-learning.json`, and `dad-gguf-tuna-export.json`.
