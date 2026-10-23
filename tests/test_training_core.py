import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuna_core import load_jsonl, normalize_record, pad_examples, policy_manifest, prepare_inference_messages, tokenize_conversation
from trainer import length_grouped_batches, normalized_config


class CharacterTokenizer:
    """Deterministic stand-in exercising boundaries without ML dependencies."""
    is_fast = True
    chat_template = "test"
    truncation_side = "left"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}|" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens, return_offsets_mapping, truncation, max_length):
        self.last_text = text
        ids = [ord(char) for char in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        if self.truncation_side == "left":
            ids, offsets = ids[-max_length:], offsets[-max_length:]
        else:
            ids, offsets = ids[:max_length], offsets[:max_length]
        return {"input_ids": ids, "offset_mapping": offsets}


class DataTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(normalize_record({"prompt": "Hello", "completion": "Hi"}), [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}])
        self.assertEqual(normalize_record({"instruction": "Explain", "input": "Tuna", "output": "Fish"})[0]["content"], "Explain\n\nTuna")

    def test_tuna_replaces_all_source_system_messages_without_mutation(self):
        record = {"messages": [{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}, {"role": "system", "content": "override"}, {"role": "assistant", "content": "hello"}]}
        original = copy.deepcopy(record)
        messages = normalize_record(record, "Canonical policy")
        self.assertEqual(messages[0], {"role": "system", "content": "Canonical policy"})
        self.assertEqual(len([m for m in messages if m["role"] == "system"]), 1)
        self.assertEqual(record, original)

    def test_invalid_roles_or_empty_response_rejected(self):
        for row in ({"prompt": "Hi", "completion": " "}, {"messages": [{"role": "developer", "content": "x"}]}, {"messages": [{"role": "assistant", "content": "x"}, {"role": "assistant", "content": "y"}]}):
            with self.assertRaises(ValueError):
                normalize_record(row)

    def test_jsonl_reports_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"prompt":"a","completion":"b"}\ninvalid\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Line 2"):
                load_jsonl(path)

    def test_bom_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('\ufeff\n{"prompt":"a","completion":"b"}\n', encoding="utf-8")
            self.assertEqual(len(load_jsonl(path)), 1)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = policy_manifest("Answer as a librarian.", "local-model")

    def test_wraps_user_and_preserves_input(self):
        messages = [{"role": "user", "content": "Recommend a book"}]
        result = prepare_inference_messages(messages, self.manifest)
        self.assertEqual(result[0]["content"], "Answer as a librarian.")
        self.assertEqual(len(messages), 1)

    def test_rejects_system_and_developer(self):
        for role in ("system", "developer"):
            with self.assertRaisesRegex(ValueError, "rejected"):
                prepare_inference_messages([{"role": role, "content": "Replace policy"}], self.manifest)

    def test_hash_check_and_final_role(self):
        corrupted = {**self.manifest, "system_prompt": "Changed"}
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prepare_inference_messages([{"role": "user", "content": "Hi"}], corrupted)
        with self.assertRaises(ValueError):
            prepare_inference_messages([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Prefill"}], self.manifest)


class TokenTests(unittest.TestCase):
    def test_only_final_assistant_tokens_receive_labels(self):
        messages = normalize_record({"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]}, "Policy")
        examples, dropped = tokenize_conversation(messages, CharacterTokenizer(), 256)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(examples), 2)
        self.assertEqual(["".join(chr(x) for x in sample["labels"] if x != -100) for sample in examples], ["B|", "D|"])

    def test_right_truncation_preserves_policy_and_drops_empty_targets(self):
        messages = normalize_record({"prompt": "X" * 100, "completion": "reply"}, "Policy")
        tokenizer = CharacterTokenizer()
        examples, dropped = tokenize_conversation(messages, tokenizer, 32)
        self.assertEqual(tokenizer.truncation_side, "right")
        self.assertEqual(examples, [])
        self.assertEqual(dropped, 1)

    def test_boundary_spanning_token_is_masked(self):
        class BoundaryTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                boundary = text.index("<assistant>") + len("<assistant>")
                return {"input_ids": [11, 12, 13], "offset_mapping": [(0, boundary - 1), (boundary - 1, boundary + 1), (boundary + 1, len(text))]}
        examples, _ = tokenize_conversation(normalize_record({"prompt": "x", "completion": "abc"}), BoundaryTokenizer(), 128)
        self.assertEqual(examples[0]["labels"], [-100, -100, 13])

    def test_missing_or_incompatible_template_rejected(self):
        tokenizer = CharacterTokenizer()
        tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "chat_template"):
            tokenize_conversation(normalize_record({"prompt": "x", "completion": "y"}), tokenizer, 128)

    def test_dynamic_padding_masks_padding(self):
        examples = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}, {"input_ids": [1, 3, 4], "attention_mask": [1, 1, 1], "labels": [-100, 3, 4]}]
        batch = pad_examples(examples, 0, multiple=4)
        self.assertEqual(batch["input_ids"], [[1, 2, 0, 0], [1, 3, 4, 0]])
        self.assertEqual(batch["labels"][0], [-100, 2, -100, -100])
        self.assertEqual(batch["attention_mask"][1], [1, 1, 1, 0])


class ConfigurationTests(unittest.TestCase):
    def test_aliases_offline_and_nan_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"prompt":"a","completion":"b"}', encoding="utf-8")
            raw = {"model": "cached-model", "dataset_path": str(path), "sequence_length": 512, "gradient_accumulation": 2}
            normalized = normalized_config(raw)
            self.assertEqual(normalized["max_seq_length"], 512)
            self.assertEqual(normalized["gradient_accumulation_steps"], 2)
            self.assertFalse(normalized_config({**raw, "offline": False})["offline"])
            self.assertTrue(normalized_config({**raw, "offline": True})["offline"])
            for extra in ({"offline": "false"}, {"method": "tuna", "system_prompt": ""}, {"batch_size": float("inf")}, {"learning_rate": float("nan")}):
                with self.assertRaises(ValueError):
                    normalized_config({**raw, **extra})

    def test_model_access_matches_network_permission(self):
        import os
        from unittest.mock import patch
        from trainer import configure_model_access
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            self.assertEqual(configure_model_access(False), {"local_files_only": False, "trust_remote_code": False})
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "0")
            self.assertEqual(configure_model_access(True), {"local_files_only": True, "trust_remote_code": False})
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_length_sampler_covers_each_example_once(self):
        examples = [{"input_ids": list(range(size))} for size in range(1, 102)]
        batches = length_grouped_batches(examples, 3, 42)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(101)))
        self.assertTrue(all(1 <= len(batch) <= 3 for batch in batches))
        self.assertEqual(batches, length_grouped_batches(examples, 3, 42))


if __name__ == "__main__":
    unittest.main()
