"""Bounded checkpoint writing and GGUF input validation without CUDA."""
import json
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gguf_import import clean_import_work, local_gguf_path, shard_plan, write_shard
from gguf_export import read_header


class ImportTests(unittest.TestCase):
    def test_gguf_file_validation_does_not_confuse_hub_ids_with_files(self):
        self.assertIsNone(local_gguf_path('owner/model'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.GGUF'
            with self.assertRaisesRegex(ValueError,'existing local'):
                local_gguf_path(str(path))
            path.write_bytes(b'NOPE')
            with self.assertRaisesRegex(ValueError,'header'):
                local_gguf_path(str(path))
            path.write_bytes(b'GGUF')
            self.assertEqual(local_gguf_path(str(path)),path.resolve())

    def test_shard_budget_keeps_large_tensors_separate(self):
        entries=[{'name':'w'+str(i),'shape':[size],'dtype':'F16'} for i,size in enumerate((2,3,100,2))]
        groups=shard_plan(entries,max_bytes=10)
        self.assertEqual([[item['name'] for item in group] for group in groups],[['w0','w1'],['w2'],['w3']])

    def test_streaming_safetensors_offsets_and_bytes_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.safetensors'
            entries=shard_plan([{'name':'w','shape':[2,2],'dtype':'F16'}, {'name':'norm','shape':[2],'dtype':'F32'}])[0]
            data={'w':struct.pack('<4e',1,2,3,4),'norm':struct.pack('<2f',5,6)}
            write_shard(path,entries,lambda item: (data[item['name']][:3],data[item['name']][3:]))
            header,start=read_header(path)
            self.assertEqual(header['w']['data_offsets'],[0,8])
            self.assertEqual(header['norm']['data_offsets'],[8,16])
            self.assertEqual(path.read_bytes()[start:],data['w']+data['norm'])
            with self.assertRaisesRegex(ValueError,'invalid size'):
                write_shard(path,entries,lambda item:[b'x'])

    def test_cleanup_keeps_completed_model_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'adapter').mkdir()
            (root/'adapter'/'weights').write_text('keep')
            scratch=root/'gguf-import-work-abc'
            scratch.mkdir()
            (scratch/'weights').write_text('partial')
            clean_import_work(root)
            self.assertFalse(scratch.exists())
            self.assertTrue((root/'adapter'/'weights').is_file())


@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('tokenizers'), 'Optional tokenizer dependencies unavailable')
class TokenizerRestorationTests(unittest.TestCase):
    def test_minicpm_numeric_splits_and_bos_survive_import(self):
        from tokenizers import Tokenizer,models,pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from gguf_import import restore_tokenizer_details
        vocab={'<s>':0,'</s>':1,'1':2,'2':3,'3':4,'4':5,'12':6,'123':7,'1234':8}
        backend=Tokenizer(models.BPE(vocab=vocab,merges=[('1','2'),('12','3'),('123','4')]))
        backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<s>',eos_token='</s>')
        self.assertEqual(tokenizer.encode('1234',add_special_tokens=False),[8])
        class Field:
            def __init__(self,value): self.value=value
            def contents(self): return self.value
        class Reader:
            def get_field(self,name):
                return Field({'tokenizer.ggml.model':'gpt2','tokenizer.ggml.pre':'minicpm5','tokenizer.ggml.add_bos_token':True}[name])
        restore_tokenizer_details(tokenizer,Reader(),len(vocab))
        self.assertEqual(tokenizer.encode('1234',add_special_tokens=False),[7,5])
        self.assertEqual(tokenizer.encode('1234',add_special_tokens=True),[0,7,5])
        self.assertTrue(tokenizer.backend_tokenizer.model.ignore_merges)

    def test_unknown_pre_tokenizer_is_rejected_instead_of_silently_changing_encoding(self):
        from gguf_import import restore_tokenizer_details
        class Field:
            def __init__(self,value): self.value=value
            def contents(self): return self.value
        class Reader:
            def get_field(self,name):
                return Field({'tokenizer.ggml.model':'gpt2','tokenizer.ggml.pre':'unknown-new-format'}[name])
        with self.assertRaisesRegex(ValueError,'pre-tokenizer'):
            restore_tokenizer_details(None,Reader(),100)


if __name__=='__main__':
    unittest.main()
