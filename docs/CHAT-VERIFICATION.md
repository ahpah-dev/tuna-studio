# Verification report: local fine-tune chat

Story: choose Chat or a completed run → load its local fine-tune → send a
conversation → receive streamed model output → stop/unload without affecting
other GPU apps.

| Boundary | Result | Evidence |
| --- | --- | --- |
| Method description | Passed | Overview, method cards and the run wizard describe Tuna as Policy-conditioned QLoRA. |
| UI → local API | Passed | Chat with model opened the selected completed run; model/source, response length and creativity controls worked. |
| API → inference | Passed | Actual saved Qwen3.5-2B adapter and Dad GGUF were loaded on RTX 5070. No hosted inference service was used. |
| Conversation → reply | Passed | The fine-tuned Qwen replied to a greeting and correctly recalled the user's first message on a second turn. |
| Streaming → UI | Passed | Multiple increasing partial-text snapshots were observed while generation was active, followed by the final reply and measured token/time usage. |
| Stop | Passed | UI cancellation produced a stopped reply. An independent real GGUF request stopped after one generated token with finish reason cancelled. |
| Tuna policy | Passed | Saved policy bound on each turn. Caller system/developer roles, backend-only reserved control tokens and oversized context were rejected. Request errors left the worker ready. |
| Unload | Passed | Zero chat worker processes remained, GPU usage returned near baseline (1,182 MiB observed), and the app continued serving requests. |
| GGUF/base deletion | Passed | Dad GGUF inference worked after its original Hugging Face base cache was removed. The existing restored GGUF checkpoint was reused; model-cache directory count remained one. |
| Browser state | Passed | Selected run and GGUF format survived reload. Histories are separate per run/format. Console error list was empty during the flow. |
| Concurrent requests | Passed in lifecycle tests | Results remain retrievable by request ID after another request starts. Stale selected-model requests are rejected instead of being sent to another loaded model. |

The app environment passed **61 unit tests**, including worker subprocess
transport, UTF-8 text, input validation, context/policy primitives, cancellation,
idle unloading, process exit errors, origin gates and retained results.

Issues corrected during verification included Windows pipe encoding, stale
poll responses, run/source selection being reset during initial data loading,
Windows venv launcher child cleanup, backend-only special-token detection, and
completed replies being replaced by a subsequent request before retrieval.

This verifies the software flow, not general model quality or an unoverrideable
policy. GGUF chat restores weights through the local importer and runs them in
Transformers with NF4; it does not use native llama.cpp inference.

See `chat-runtime-verification.json` for the real GGUF policy/cancellation checks
and [CHAT.md](CHAT.md) for usage and memory behavior.
