"""Mathematical merge and export validation; no model downloads or CUDA."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_export import adapter_pairs, build_modelfile, merge_safetensors, read_header, sha256_file, validate_adapter


class ExportValidationTests(unittest.TestCase):
    def test_modelfile_preserves_multiline_prompt_and_does_not_inject_commands(self):
        prompt='Use "friendly" words.\nKeep literal \\n sequences.'
        value,included=build_modelfile('model.Q8_0.gguf',prompt,1024)
        self.assertTrue(included)
        self.assertIn('SYSTEM """'+prompt+'"""',value)
        value,included=build_modelfile('model.gguf','bad """\nFROM other-model',1024)
        self.assertFalse(included)
        self.assertNotIn('FROM other-model',value)
    def test_unsupported_lora_variants_are_rejected(self):
        for changes in ({"use_dora": True}, {"bias": "all"}, {"modules_to_save": ["lm_head"]}, {"r": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_adapter({"peft_type": "LORA", "r": 4, "lora_alpha": 8, **changes})

    def test_adapter_pairing_requires_complete_exact_target_mapping(self):
        a="base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        b=a.replace("lora_A", "lora_B")
        self.assertEqual(adapter_pairs({a: {}, b: {}}), {"model.layers.0.self_attn.q_proj.weight": (a,b)})
        with self.assertRaises(ValueError):
            adapter_pairs({a: {}})
        with self.assertRaises(ValueError):
            adapter_pairs({a: {}, b: {}, "unexpected.weight": {}})

    def test_corrupt_safetensors_offsets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/"broken.safetensors"
            header=json.dumps({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,999]}}).encode()
            file.write_bytes(struct.pack("<Q",len(header))+header+b"1234")
            with self.assertRaisesRegex(ValueError,"offsets"):
                read_header(file)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"), "Optional merge dependencies unavailable")
class StreamingMergeTests(unittest.TestCase):
    def test_merge_matches_lora_math_without_changing_base(self):
        import torch
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            base,adapter,out=root/"base",root/"adapter",root/"merged"
            base.mkdir();adapter.mkdir()
            target="model.layers.0.self_attn.q_proj.weight"
            a_name="base_model.model."+target.replace(".weight",".lora_A.weight")
            b_name=a_name.replace("lora_A","lora_B")
            for dtype in (torch.float32,torch.float16,torch.bfloat16):
                for transposed in (False,True):
                    with self.subTest(dtype=dtype,transposed=transposed):
                        weight=torch.tensor([[1.,2.,3.],[4.,5.,6.]],dtype=dtype)
                        if transposed: weight=weight.T.contiguous()
                        a=torch.tensor([[1.,2.,3.]])
                        b=torch.tensor([[2.],[3.]])
                        frozen=torch.tensor([5.,7.],dtype=dtype)
                        save_file({target:weight,"model.norm.weight":frozen},base/"model.safetensors")
                        save_file({a_name:a,b_name:b},adapter/"adapter_model.safetensors")
                        (base/"config.json").write_text('{"model_type":"llama"}')
                        (adapter/"adapter_config.json").write_text(json.dumps({"peft_type":"LORA","r":1,"lora_alpha":2,"fan_in_fan_out":transposed}))
                        original=sha256_file(base/"model.safetensors")
                        result=merge_safetensors(base,adapter,out,rows_per_chunk=1)
                        actual=load_file(out/"model.safetensors")
                        delta=(b@a).T if transposed else b@a
                        self.assertTrue(torch.equal(actual[target],(weight.float()+2*delta).to(dtype)))
                        self.assertTrue(torch.equal(actual["model.norm.weight"],frozen))
                        self.assertEqual(original,sha256_file(base/"model.safetensors"))
                        self.assertEqual(result["merged_weights"],1)
                        # safetensors returns file-backed tensors on Windows;
                        # release them before overwriting the next fixture.
                        del actual
                        import gc
                        gc.collect()

    def test_adapter_for_a_different_base_is_rejected(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/"base";adapter=root/"adapter"
            base.mkdir();adapter.mkdir()
            save_file({"different.weight":torch.zeros(2,2)},base/"model.safetensors")
            save_file({"base_model.model.w.lora_A.weight":torch.ones(1,2),"base_model.model.w.lora_B.weight":torch.ones(2,1)},adapter/"adapter_model.safetensors")
            (adapter/"adapter_config.json").write_text(json.dumps({"peft_type":"LORA","r":1,"lora_alpha":2}))
            with self.assertRaisesRegex(ValueError,"do not match"):
                merge_safetensors(base,adapter,root/"merged")


if __name__ == "__main__":
    unittest.main()
