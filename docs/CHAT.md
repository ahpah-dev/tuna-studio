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
code remain disabled. CUDA inference uses NF4 double quantization and applies
the saved adapter when that format is selected. Standard LoRA runs can also use
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
- The context budget is at most 8,192 tokens, including the requested reply,
  and can be lower for small-context models. Oversized conversations are rejected
  instead of truncating the policy. Start a new chat when the budget is reached.

API routes: `/api/chat`, `/api/chat/load`, `/api/chat/message`, `/api/chat/stop`,
`/api/chat/unload` and `/api/chat/replies/<request-id>`. They use the same loopback
and origin checks as training. A separate resident `chat_worker.py` handles
inference; unloading it leaves unrelated GPU apps alone.
