"""Dependency-free dataset/policy primitives and an optional local serving wrapper.

Tuna binds an application-controlled system message to every training example.
The model's adherence remains probabilistic. Only this wrapper enforces which
system message is submitted; it does not guarantee compliant generated text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100
POLICY_LIMITATION = (
    "Fine-tuning cannot make a prompt unoverrideable in model weights. "
    "The serving wrapper rejects caller system messages and always inserts the "
    "saved policy, but generated compliance remains probabilistic. Anyone with "
    "direct model or filesystem access can bypass or change the wrapper."
)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def normalize_record(record: dict, system_prompt: str | None = None) -> list[dict[str, str]]:
    """Accept messages, prompt/completion, and instruction/input/output JSONL.

    A Tuna policy intentionally replaces dataset system messages. All other
    methods retain one leading dataset system message. No source object mutates.
    """
    if not isinstance(record, dict):
        raise ValueError("Each JSONL row must be an object")
    if "messages" in record:
        raw = record["messages"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("messages must be a non-empty list")
        messages = []
        for index, msg in enumerate(raw):
            if not isinstance(msg, dict):
                raise ValueError(f"messages[{index}] must be an object")
            role = msg.get("role")
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"Unsupported role at messages[{index}]: {role!r}")
            messages.append({"role": role, "content": _text(msg.get("content"), f"messages[{index}].content")})
    elif "prompt" in record and ("completion" in record or "response" in record):
        messages = [
            {"role": "user", "content": _text(record["prompt"], "prompt")},
            {"role": "assistant", "content": _text(record.get("completion", record.get("response")), "completion")},
        ]
    elif "instruction" in record and "output" in record:
        prompt = _text(record["instruction"], "instruction")
        extra = record.get("input", "")
        if not isinstance(extra, str):
            raise ValueError("input must be a string")
        if extra.strip():
            prompt += "\n\n" + extra
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": _text(record["output"], "output")}]
    else:
        raise ValueError("Expected messages, prompt/completion, or instruction/output fields")
    if system_prompt is not None:
        policy = _text(system_prompt, "system_prompt")
        messages = [{"role": "system", "content": policy}] + [m for m in messages if m["role"] != "system"]
    elif any(m["role"] == "system" for m in messages[1:]):
        raise ValueError("Only one leading system message is supported")
    conversation = messages[1:] if messages and messages[0]["role"] == "system" else messages
    if len(conversation) < 2 or conversation[-1]["role"] != "assistant":
        raise ValueError("Each training conversation must contain a user and end with an assistant reply")
    for index, message in enumerate(conversation):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError("Conversation roles must alternate user, assistant")
    return messages


def load_jsonl(path: str | Path, system_prompt: str | None = None) -> list[list[dict[str, str]]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(normalize_record(json.loads(line), system_prompt))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("Dataset has no non-empty training rows")
    return rows


def policy_manifest(system_prompt: str, model_path: str, **metadata: Any) -> dict:
    policy = _text(system_prompt, "system_prompt")
    return {
        "schema_version": 1,
        "method": "tuna",
        "base_model": model_path,
        "system_prompt": policy,
        "policy_sha256": hashlib.sha256(policy.encode("utf-8")).hexdigest(),
        "enforcement": "serving-wrapper-only",
        "limitation": POLICY_LIMITATION,
        "metadata": metadata,
    }


def prepare_inference_messages(messages: list[dict], manifest: dict) -> list[dict[str, str]]:
    """Bind the saved policy; callers cannot supply system/developer messages."""
    policy = _text(manifest.get("system_prompt"), "manifest.system_prompt")
    digest = hashlib.sha256(policy.encode("utf-8")).hexdigest()
    if manifest.get("policy_sha256") != digest:
        raise ValueError("Policy manifest hash mismatch")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Provide at least one user message")
    prepared = [{"role": "system", "content": policy}]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("Every message must be an object")
        role = message.get("role")
        if role in ("system", "developer"):
            raise ValueError("Tuna policy is application-controlled; caller system/developer messages are rejected")
        expected = "user" if index % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError("Caller messages must alternate user, assistant, and end with user")
        prepared.append({"role": role, "content": _text(message.get("content"), "message.content")})
    if prepared[-1]["role"] != "user":
        raise ValueError("The final caller message must be user")
    return prepared


def tokenize_conversation(messages: list[dict], tokenizer: Any, max_length: int) -> tuple[list[dict], int]:
    """One example per assistant turn, supervising only that assistant reply.

    The native chat template is required. Fast-tokenizer offsets identify the
    boundary without accidentally supervising a BPE token spanning the prompt.
    Right truncation preserves the system policy. Replies truncated entirely are
    omitted and counted, rather than producing all-masked / NaN training losses.
    """
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Model tokenizer needs a native chat_template; configure a template before training")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Assistant-only loss requires a fast tokenizer with character offsets")
    tokenizer.truncation_side = "right"
    examples, dropped = [], 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages[:index + 1], tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise ValueError("This chat template's generation prefix differs from its training prefix; use a compatible non-reasoning chat template")
        encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=max_length)
        ids = list(encoded["input_ids"])
        labels = [token if start >= len(prompt) and end > start else IGNORE_INDEX
                  for token, (start, end) in zip(ids, encoded["offset_mapping"])]
        if not any(label != IGNORE_INDEX for label in labels[1:]):
            dropped += 1
            continue
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return examples, dropped


def pad_examples(examples: list[dict], pad_token_id: int, multiple: int = 8) -> dict[str, list[list[int]]]:
    """Dynamic padding; labels never include prompt or padding positions."""
    if not examples:
        raise ValueError("Cannot pad an empty batch")
    longest = max(len(item["input_ids"]) for item in examples)
    target = ((longest + multiple - 1) // multiple) * multiple if multiple > 1 else longest
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in examples:
        size = target - len(example["input_ids"])
        result["input_ids"].append(example["input_ids"] + [pad_token_id] * size)
        result["attention_mask"].append(example["attention_mask"] + [0] * size)
        result["labels"].append(example["labels"] + [IGNORE_INDEX] * size)
    return result


def reserved_control_tokens(tokenizer):
    """Include backend controls missing from a generic HF special-token map."""
    tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    for token in (getattr(tokenizer, "added_tokens_decoder", {}) or {}).values():
        if getattr(token, "special", False):
            tokens.add(getattr(token, "content", str(token)))
    return {token for token in tokens if isinstance(token, str) and token}


class PolicyBoundGenerator:
    """Optional offline HF/PEFT inference, using a Tuna export directory."""

    def __init__(self, export_dir: str | Path):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        directory = Path(export_dir).resolve()
        self.manifest = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
        prepare_inference_messages([{"role": "user", "content": "validate"}], self.manifest)
        if not torch.cuda.is_available():
            raise RuntimeError("Tuna 4-bit inference requires a CUDA-capable GPU")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(directory / "adapter", local_files_only=True, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(
            self.manifest["base_model"], local_files_only=True, trust_remote_code=False,
            device_map={"": 0}, attn_implementation="sdpa", torch_dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        )
        self.model = PeftModel.from_pretrained(base, directory / "adapter", local_files_only=True).eval()

    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        import torch

        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 2048:
            raise ValueError("max_new_tokens must be between 1 and 2048")
        prepared = prepare_inference_messages(messages, self.manifest)
        # Keep literal model control tokens from being introduced by callers.
        controls = reserved_control_tokens(self.tokenizer)
        for message in prepared[1:]:
            if any(token in message["content"] for token in controls):
                raise ValueError("Caller content contains reserved model control tokens")
        prompt = self.tokenizer.apply_chat_template(prepared, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context and encoded["input_ids"].shape[1] + max_new_tokens > context:
            raise ValueError("Request exceeds model context length; policy will not be truncated")
        with torch.inference_mode():
            output = self.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
