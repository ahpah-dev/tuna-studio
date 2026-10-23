# Save a trained model as GGUF

## Start from an existing GGUF

In a new run, enter the **full path to a local `.gguf` file** in Base model,
choose **Tuna**, select your JSONL dataset and enter your system policy. You can
also open a completed GGUF export and click **Fine-tune this GGUF**; this carries
its dataset, policy and training settings into a new run.

The input importer currently supports **Llama-family GGUF architecture**,
including this machine's MiniCPM5 model using Llama weights. A single complete
file, a compatible tokenizer and a chat template are required. Unsupported
architectures, unknown pre-tokenizers, split files, missing tensors and shape
mismatches are rejected instead of producing a partially imported model.

GGUF is restored locally into F16 safetensors (F32 normalization vectors), then
Tuna trains NF4 QLoRA adapters on that checkpoint. GGUF inference blocks are not
updated directly. Dequantization is row-chunked to limit RAM usage, and the
restored checkpoint is kept under `data/models/gguf-v2-<source-sha256>/` for
subsequent inference, adapter merging and reuse. Keep that directory while using
its adapters. Import does not download a second base model or alter the source
GGUF. It needs roughly two bytes per model parameter in additional disk space;
export needs further temporary space. Quantization already lost from the source
cannot be recovered.

For Tuna, dataset assistant replies must demonstrate the desired behavior. The
system policy supplies context; inserting a policy alone cannot create correct
target answers. Completed runs verify that learned LoRA updates are present.

## Save the tuned model

Open a **completed training run**. In **Save as GGUF**, choose a format and click
**Export GGUF**. The app merges your trained adapter into the cached base weights
and converts the result locally using the bundled llama.cpp converter.

- **Q8_0** saves 8-bit quantized weights and is the default smaller-file option.
- **F16** saves 16-bit floating-point weights and uses more disk space.

The resulting `.gguf` is a standalone model, including the base weights and your
learned adapter changes. You do not need to load a separate LoRA adapter when
using that file. Quantization can change numerical outputs.

The base model must already be cached or available as a local Hugging Face
directory with safetensors weights, including a restored GGUF checkpoint.
For a GGUF input, export merges into that exact restored source checkpoint.
The exporter does not download models or
upload data. It supports the standard LoRA/QLoRA/Tuna adapters created by this
app, provided the base model architecture/tokenizer is supported by llama.cpp.
DoRA, saved extra modules, and other unsupported adapter variants are rejected.

## Use the exported file

After export, click **Download GGUF**. Import the file in a GGUF-compatible local
runtime such as llama.cpp or LM Studio. Configure the context length to fit
your available memory.

For Ollama, also download **Ollama files**, extract them beside the GGUF, then run:

```powershell
ollama create my-tuna-model -f .\Modelfile
ollama run my-tuna-model
```

These commands require an existing Ollama installation. Tuna Studio does not
install it or change models in another application automatically.

For Tuna, companion files include the saved system prompt, policy manifest and
Modelfile. GGUF itself does not enforce the system prompt. Other runtimes should
be configured with the text from `system-prompt.txt`. Output adherence remains
probabilistic. Prompts containing a literal triple-quote delimiter cannot be
represented losslessly by the generated Modelfile; their exact text is preserved
in `system-prompt.txt` and a note explains manual configuration.

## Local files and resources

Files are saved under `data/runs/<run-id>/gguf/`, including a checksum and export
manifest. Downloading a GGUF streams the file from disk. The adapter ZIP remains
separate and does not include large GGUF or temporary merge files.

The merge runs on CPU, copying base shards to a temporary directory and patching
trained matrices in small chunks. It does not load the full base model on the
GPU. Conversion uses llama.cpp's lazy loading and a temporary output stream.
RAM and disk are still needed for conversion. At least twice the cached base
weight size plus a reserve must be free before merging. Large embedding tensors
can require additional RAM during conversion.

Only one export or training operation runs at a time through Tuna Studio. You can
cancel an export; generated scratch files are cleaned after the worker releases
them. Restarting the app marks interrupted exports as failed so they can be
retried. The original model and adapter weights remain available.

## Conversion provenance

The bundled, unmodified Python converter and GGUF library come from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), commit
`970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`. Its MIT license and source record are
included under `vendor/llama_cpp/`. Python export dependencies are declared in
`requirements-export.txt` and included by `requirements-train.txt`.

The app validates GGUF metadata and a nonempty tensor table, then computes a
SHA-256 checksum before exposing the completed download. Merge tests check
mathematics, transposed layouts, F32/F16/BF16 preservation, and unchanged base
files. Server tests cover export lifecycle, cancellation, concurrency, downloads,
companion ZIPs and isolation from the adapter ZIP.

A real CUDA round trip also imported a Q8_0 GGUF, ran 20 Tuna updates, measured
lower loss on its training examples, verified the seven exported adapted
matrices byte-for-byte, and loaded/generated with llama.cpp. MiniCPM tokenizer
encoding was checked against the original tokenizer for chat, numeric,
multilingual and control-token examples. See `gguf-training-verification.json`
and `tuna-gguf-weight-audit.json`. These are software checks, not a held-out
quality or adversarial policy evaluation.
