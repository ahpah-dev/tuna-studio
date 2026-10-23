# Chat with a fine-tuned model

Open **Chat** in the sidebar, choose a completed fine-tune, then click **Load
model**. Or open a completed run and click **Chat with model**. Type a message
and press **Enter** to send; use **Shift+Enter** for a new line. Replies stream
into the conversation. **Stop reply** interrupts your current generation.

Choose **Trained adapter** to use the saved adapter and its local base weights.
If the original base was deleted, select **Exported GGUF** when the run has a
completed export. GGUF chat uses the local Llama-family importer described in
[GGUF.md](GGUF.md). It restores a checkpoint on disk when one is not cached,
then loads it for inference. It does not run llama.cpp or modify the GGUF.
Imports can need substantial extra disk space; a matching cached import is reused.

Chat never downloads models or uploads messages. Models needing custom repository
code remain disabled. CUDA inference automatically chooses BF16/FP16 when the
local weights fit with reserved memory for generation; otherwise it uses NF4
double quantization. The saved adapter is applied in both cases. Dense inference
merges it into the in-memory model once to avoid extra adapter projections on
every token. Saved model and adapter files are unchanged. Standard LoRA runs can also use
CPU inference; Tuna and QLoRA chat require CUDA. A missing base is reported with
instructions to select a GGUF export or restore the base files.

## Tuna policy

Tuna is **policy-conditioned QLoRA**: 4-bit adapter training on examples
conditioned by the saved system policy. Chat validates the policy manifest and
inserts that system message on every turn. Caller system/developer messages and
reserved model control tokens are rejected, including controls recorded only in
the tokenizer backend. The saved policy is read-only in chat and can be inspected
under **View saved policy**. Generated adherence remains probabilistic.

History is stored separately for each run and format in this browser's local
storage. **New chat** clears the current conversation. Generated content is
escaped text with fenced-code formatting; it cannot execute HTML or JavaScript.
Clearing browser data removes stored chat history.

## Memory and controls

- **Unload** stops the app's inference process tree and releases model memory.
  It can also cancel a model that is loading.
- Ready models unload after five minutes without a generation request. Page
  polling does not reset the timer. Browser conversation history remains.
- One model and one reply are active at a time. Completed replies are retained
  by request ID so another chat cannot replace a result before its page receives it.
- Training/export unloads an idle chat model. Active loading or replies must
  first be stopped, preventing overlapping heavy operations.
- Response length and creativity can be changed between turns. Focused mode
  uses greedy generation; other modes sample at the selected temperature.
- **Unlimited** removes the fixed reply cap and uses all remaining loaded chat
  context. Generation ends when the model finishes, you press Stop, or the
  context fills. It is not infinite context. Fixed choices up to 2,048 tokens
  remain available. A context-limited reply is labeled in the conversation.
- The context budget is at most 8,192 tokens, including the requested reply,
  and can be lower for small-context models. Oversized conversations are rejected
  instead of truncating the policy. Start a new chat when the budget is reached.

API routes: `/api/chat`, `/api/chat/load`, `/api/chat/message`, `/api/chat/stop`,
`/api/chat/unload` and `/api/chat/replies/<request-id>`. They use the same loopback
and origin checks as training. A separate resident `chat_worker.py` handles
inference; unloading it leaves unrelated GPU apps alone.

## Reasoning

Choose **Reasoning → On · think before answering** to enable the tokenizer's
native `enable_thinking` mode. The worker checks that the switch actually changes
the model's prompt. Models without that switch show the control as unavailable;
requesting it through the API returns an actionable error.

Supported `<think>…</think>` output streams into a separate **Reasoning**
disclosure. It opens while the model thinks and collapses after completion; the
final answer appears underneath. The app routes the model's actual output and
does not manufacture reasoning. Reasoning tokens and answer tokens share the
selected response budget, so Unlimited is useful for longer reasoning tasks.

Reasoning is stored locally alongside the answer but is not sent back as chat
history. If you stop before any final answer, the partial reasoning remains
visible; that unfinished user/assistant pair is omitted from the next model
request to keep roles valid. Completed answers remain in the normal history.
The saved Tuna policy is applied in both modes.

If a model repeats itself while thinking, press **Stop reply** and try Balanced
creativity. Reasoning quality and termination depend on the model; the
[Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B) documents its
susceptibility to thinking loops, including with recommended sampling settings.

API example after a supported model is loaded:

```json
{
  "messages": [{"role": "user", "content": "What is 17 times 23?"}],
  "max_new_tokens": null,
  "reasoning": true,
  "temperature": 0.7
}
```

`max_new_tokens: null` means Unlimited. Omitting the field retains the 256-token
default; omitting `reasoning` retains direct-answer mode.

Verified locally on 2026-10-23 with a Qwen3.5-2B QLoRA run on an RTX 5070:
Unlimited generated 3,403 tokens before an explicit Stop, exceeding the old
2,048-token cap. A separate sampled reasoning request finished normally with
173 generated tokens, a separate reasoning section and final answer `4`.
The available reply budget was 8,168 tokens after its 24-token prompt. All 85
Python tests and the Node chat UI checks passed, including split delimiters,
context limits, cancellation, retained replies and unfinished-turn history.
The browser also completed a sampled reasoning turn after that stopped turn:
1,059 generated tokens, final answer `4`, a collapsible reasoning section and
both settings preserved across reload. No browser warnings or errors were logged.

## Faster inference

Small models often decode faster in BF16/FP16 than in 4-bit NF4. Auto selection
uses local checkpoint byte counts and free GPU memory, with 25% weight headroom
plus at least 1.5 GiB reserved for caches and activations. Unknown/prequantized
checkpoints stay on NF4. A dense model that runs out of memory during loading
falls back to NF4. Dense inference uses more VRAM and may produce different
answers from NF4 because base-weight precision changes; this is not retraining.

Qwen3.5 also uses a model-local optimization for its single-token convolution.
Optional Flash Linear Attention kernels accelerate its recurrent attention:

```powershell
# After setup-cuda.ps1; targets that script's PyTorch 2.14 environment.
.\setup-fast-chat.ps1
```

The installer adds `fla-core`, `einops` and the matching Windows Triton wheel in
the app `.venv`, then verifies a real GPU kernel. Unload/reload an existing chat
model afterward. It leaves the installed PyTorch build in place. No models or
messages are downloaded/uploaded by this setup. Supported dense Qwen3.5 models
also compile their decoder with a fixed-capacity static cache. The initial model
load warms kernels and compiled decoding before enabling chat; first-time
preparation can take several minutes. Repeated messages use fresh attention
state, so one conversation cannot leak its cache into another. Set
`TUNA_CHAT_COMPILE=0` in the server environment to skip decoder compilation.
If compilation fails or needs too much memory, ordinary decoding remains available.
Optional kernels that fail during warmup fall back to reference attention on
the loaded model. CPU inference always retains CPU-compatible attention.

For the previous lower-memory decoding path, launch the server with
`TUNA_CHAT_PRECISION=nf4` in its environment. In PowerShell:

```powershell
$env:TUNA_CHAT_PRECISION = 'nf4'
python server.py
```

Completed replies show measured tokens/s and first-token latency. These are
inference timings: first-token latency begins immediately before generation,
excluding model loading and HTTP transport. Text appears at word boundaries.
Active chat polls at 100 ms and idle chat at 1.5 seconds. See
[CHAT-PERFORMANCE.md](CHAT-PERFORMANCE.md) for the measured workload and limits.
