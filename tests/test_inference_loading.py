"""Memory-aware inference selection stays offline and leaves cache headroom."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inference_loading import GIB, checkpoint_size, choose_precision


class PrecisionTests(unittest.TestCase):
    def test_small_model_uses_dense_and_large_or_unknown_model_uses_nf4(self):
        def choose(size,free=10*GIB,**extra):
            return choose_precision(size,free,12*GIB,cuda=True,bf16=True,**extra)
        self.assertEqual(choose(4*GIB),'bf16')
        for size in (None,0,8*GIB,24*GIB):
            self.assertEqual(choose(size),'nf4')
        self.assertEqual(choose(4*GIB,free=5*GIB),'nf4')
        self.assertEqual(choose(4*GIB,preference='nf4'),'nf4')
        self.assertEqual(choose_precision(4*GIB,10*GIB,12*GIB,cuda=True,bf16=False),'fp16')
        self.assertEqual(choose_precision(None,0,0,cuda=False,bf16=False),'fp32')
        with self.assertRaises(ValueError):
            choose(4*GIB,preference='unknown')

    def test_metadata_is_local_and_quantized_or_bad_checkpoints_are_conservative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=root/'config.json'
            config.write_text('{}')
            weights=root/'model.safetensors'
            weights.write_bytes(b'local-weights')
            self.assertEqual(checkpoint_size(root),13)
            index=root/'model.safetensors.index.json'
            index.write_text(json.dumps({'metadata':{'total_size':4*GIB}}))
            self.assertEqual(checkpoint_size(root),4*GIB)
            for value in (True,-1,0,'4GB',None):
                index.write_text(json.dumps({'metadata':{'total_size':value}}))
                self.assertIsNone(checkpoint_size(root))
            config.write_text('{"quantization_config":{"bits":4}}')
            self.assertIsNone(checkpoint_size(root))
            config.write_text('broken')
            self.assertIsNone(checkpoint_size(root))


if __name__=='__main__':
    unittest.main()
