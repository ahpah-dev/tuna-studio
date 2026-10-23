"""Offline loader lifecycle checks using stdlib fakes, without GPU libraries."""
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chat_worker
from inference_loading import GIB
from tuna_core import policy_manifest


class FakeOOM(RuntimeError):
    pass


class Encoded(dict):
    def to(self, device):
        return self


class LoaderTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, preference="auto", compile_enabled=True, load_oom=False,
                generation_failures=(), stable_cache=True):
        events, prepared = [], []
        failures = list(generation_failures)
        device = SimpleNamespace(type="cuda")

        class Tokenizer:
            chat_template = "fixture-template"
            pad_token_id, eos_token_id = 0, 2
            def get_vocab(self):
                return {"<|im_end|>": 2}
            def apply_chat_template(self, messages, **kwargs):
                prepared.append((messages, kwargs))
                return "fixture-rendered-prompt"
            def __call__(self, prompt, **kwargs):
                return Encoded(input_ids=SimpleNamespace(shape=(1, 8)))

        class Model:
            def __init__(self):
                self.config = SimpleNamespace(max_position_embeddings=32768)
                self.generation_config = SimpleNamespace(eos_token_id=2,
                    cache_implementation=None, compile_config=None, disable_compile=False)
                self.device = device
            def _prepare_static_cache(self):
                return self._previous_max_cache_length
            def eval(self):
                events.append(("eval",))
                return self
            def merge_and_unload(self, **kwargs):
                events.append(("merge", kwargs))
                return self
            def generate(self, **kwargs):
                events.append(("generate", self.generation_config.cache_implementation,
                               dict(kwargs), self.generation_config.disable_compile))
                if self.generation_config.cache_implementation == "static":
                    self._compiled_call = object()
                if failures:
                    failure = failures.pop(0)
                    if failure is not None:
                        raise failure
                return object()

        model, tokenizer = Model(), Tokenizer()
        if not stable_cache:
            model._prepare_static_cache = lambda: None

        def cleanup(kind):
            # A live exception traceback can retain failed CUDA graph/cache
            # allocations. The retry must happen after leaving its handler.
            events.append((kind, sys.exception() is None))

        torch = ModuleType("torch")
        torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
        torch.inference_mode = nullcontext
        torch.cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
            mem_get_info=lambda: (10*GIB, 12*GIB), OutOfMemoryError=FakeOOM,
            empty_cache=lambda: cleanup("empty_cache"), synchronize=lambda: events.append(("sync",)),
            get_device_name=lambda index: "Fixture CUDA", memory_allocated=lambda: 4*GIB)
        torch._dynamo = SimpleNamespace(reset=lambda: cleanup("reset"))
        transformers, peft, optimizations = [ModuleType(name) for name in (
            "transformers", "peft", "inference_optimizations")]

        def load_weights(base, **kwargs):
            events.append(("load", kwargs))
            if load_oom and sum(row[0] == "load" for row in events) == 1:
                raise FakeOOM("fixture dense weights exceed VRAM")
            return model

        def load_adapter(loaded, directory, **kwargs):
            self.assertIs(loaded, model)
            events.append(("adapter", kwargs))
            return loaded

        def install(loaded, *, allow_fused=True):
            self.assertIs(loaded, model)
            events.append(("optimize", allow_fused))
            return ["Qwen3.5 fused linear attention"] if allow_fused else ["Qwen3.5 reference linear attention"]

        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_weights)
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
        transformers.BitsAndBytesConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        transformers.CompileConfig = lambda **kwargs: SimpleNamespace(**kwargs)
        peft.PeftModel = SimpleNamespace(from_pretrained=load_adapter)
        optimizations.install_optimizations = install

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run = directory / "run"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({"model_path":"fixture/local", "method":"tuna"}), encoding="utf-8")
            (run / "training_summary.json").write_text(json.dumps({"model_path":"fixture/local"}), encoding="utf-8")
            (run / "policy.json").write_text(json.dumps(policy_manifest("Always be friendly.", "fixture/local")), encoding="utf-8")
            args = SimpleNamespace(run_dir=run, source="adapter", cache_dir=directory/"cache", work_dir=directory/"work")
            environment = {"TUNA_CHAT_PRECISION":preference, "TUNA_CHAT_COMPILE":"1" if compile_enabled else "0"}
            with patch.dict(sys.modules, {"torch":torch, "transformers":transformers,
                    "peft":peft, "inference_optimizations":optimizations}), \
                    patch.dict(chat_worker.os.environ, environment), \
                    patch.object(chat_worker, "checkpoint_size", return_value=4*GIB), \
                    patch.object(chat_worker.gc, "collect", side_effect=lambda: cleanup("collect")), \
                    patch.object(chat_worker, "emit", side_effect=lambda kind, **fields: events.append((kind, fields))):
                yield SimpleNamespace(args=args, model=model, events=events, prepared=prepared)

    def ready(self, fixture):
        rows = [row[1] for row in fixture.events if row[0] == "ready"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(fixture.events[-1][0], "ready")
        self.assertEqual(fixture.events[-2][0], "sync")
        return rows[0]

    def test_dense_adapter_merges_and_compiles_only_after_both_complete_warmups(self):
        with self.fixture() as fixture:
            model, tokenizer, manifest, context = chat_worker.load_model(fixture.args)
            self.assertIs(model, fixture.model)
            ready = self.ready(fixture)
            self.assertEqual(ready["precision"], "bf16")
            self.assertTrue(ready["adapter_merged"])
            self.assertTrue(ready["compiled"])
            self.assertFalse(ready["quantized"])
            load = next(row[1] for row in fixture.events if row[0] == "load")
            self.assertNotIn("quantization_config", load)
            self.assertTrue(load["local_files_only"])
            self.assertEqual(next(row[1] for row in fixture.events if row[0] == "merge"), {"safe_merge":True})
            self.assertEqual(fixture.model._previous_max_cache_length, 8192)
            calls = [row for row in fixture.events if row[0] == "generate"]
            self.assertEqual(len(calls), 2)
            for row in calls:
                self.assertEqual(row[1], "static")
                self.assertEqual(row[2]["max_new_tokens"], 3)
                self.assertEqual(row[2]["min_new_tokens"], 3)
                self.assertFalse(row[2]["do_sample"])
            self.assertTrue(all(messages[0] == {"role":"system", "content":"Always be friendly."}
                                for messages, kwargs in fixture.prepared))

    def test_forced_nf4_retains_adapter_and_avoids_dense_compile(self):
        with self.fixture(preference="nf4") as fixture:
            chat_worker.load_model(fixture.args)
            ready = self.ready(fixture)
            self.assertEqual(ready["precision"], "nf4")
            self.assertTrue(ready["quantized"])
            self.assertFalse(ready["adapter_merged"])
            self.assertFalse(ready["compiled"])
            self.assertFalse(any(row[0] == "merge" for row in fixture.events))
            load = next(row[1] for row in fixture.events if row[0] == "load")
            self.assertTrue(load["quantization_config"].load_in_4bit)
            self.assertEqual(load["quantization_config"].bnb_4bit_quant_type, "nf4")
            self.assertEqual(len([row for row in fixture.events if row[0] == "generate"]), 1)

    def test_dense_load_oom_collects_after_exception_before_nf4_retry(self):
        with self.fixture(load_oom=True) as fixture:
            chat_worker.load_model(fixture.args)
            loads = [index for index, row in enumerate(fixture.events) if row[0] == "load"]
            self.assertEqual(len(loads), 2)
            middle = fixture.events[loads[0]+1:loads[1]]
            self.assertIn(("collect", True), middle)
            self.assertIn(("empty_cache", True), middle)
            self.assertEqual(self.ready(fixture)["precision"], "nf4")

    def test_compile_error_and_oom_clean_up_outside_handler_then_use_dynamic_cache(self):
        for failure in (RuntimeError("fixture compiler unavailable"), FakeOOM("fixture graph pool OOM")):
            with self.subTest(failure=type(failure).__name__), self.fixture(generation_failures=(failure,)) as fixture:
                chat_worker.load_model(fixture.args)
                ready = self.ready(fixture)
                self.assertFalse(ready["compiled"])
                self.assertFalse(hasattr(fixture.model, "_compiled_call"))
                calls = [row for row in fixture.events if row[0] == "generate"]
                self.assertEqual([row[1] for row in calls], ["static", None])
                self.assertTrue(calls[-1][3])
                self.assertIn(("reset", True), fixture.events)
                self.assertIn(("collect", True), fixture.events)
                self.assertIn(("empty_cache", True), fixture.events)
                cleanup_index = next(index for index, row in enumerate(fixture.events) if row[0] == "empty_cache")
                retry_index = next(index for index, row in enumerate(fixture.events)
                                   if row[0] == "generate" and row[1] is None)
                self.assertLess(cleanup_index, retry_index)

    def test_optional_kernel_failure_retries_model_local_reference_before_ready(self):
        with self.fixture(compile_enabled=False, generation_failures=(RuntimeError("fixture unsupported FLA"),)) as fixture:
            chat_worker.load_model(fixture.args)
            ready = self.ready(fixture)
            self.assertEqual([row[1] for row in fixture.events if row[0] == "optimize"], [True, False])
            self.assertNotIn("Qwen3.5 fused linear attention", ready["optimizations"])
            self.assertEqual(len([row for row in fixture.events if row[0] == "generate"]), 2)
            self.assertIn(("collect", True), fixture.events)

    def test_changed_static_cache_internal_api_keeps_compatible_decoder(self):
        with self.fixture(stable_cache=False) as fixture:
            chat_worker.load_model(fixture.args)
            self.assertFalse(self.ready(fixture)["compiled"])
            calls = [row for row in fixture.events if row[0] == "generate"]
            self.assertEqual(len(calls), 1)
            self.assertIsNone(calls[0][1])


if __name__ == "__main__":
    unittest.main()
