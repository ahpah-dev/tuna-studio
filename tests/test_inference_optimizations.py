"""CPU numerical checks for the narrow Qwen3.5 cached convolution fast path."""
from pathlib import Path
import inspect
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_optimizations import _make_single_token_conv_update, install_optimizations

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional for the source-only test suite")
class SingleTokenConvolutionTests(unittest.TestCase):
    def setUp(self):
        self.calls = 0

        def reference(hidden_states, conv_state, weight, bias=None, activation=None, **kwargs):
            self.calls += 1
            length = conv_state.shape[-1]
            extended = torch.cat((conv_state, hidden_states), dim=-1).to(weight.dtype)
            conv_state.copy_(extended[..., -length:])
            result = F.conv1d(extended, weight.unsqueeze(1), bias, groups=hidden_states.shape[1])
            result = result[..., -hidden_states.shape[-1]:]
            if activation is not None:
                result = F.silu(result)
            return result.to(hidden_states.dtype)

        self.reference = reference
        self.optimized = _make_single_token_conv_update(reference)

    def reference_qwen(self):
        """Keep CPU fixtures on the upstream Torch path if FLA is installed.

        Transformers selects optional GPU kernels at import time. These tests
        exercise the same reference convolution branch the optimizer replaces,
        while GPU kernel validation is a separate hardware-dependent check.
        """
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
        except ImportError:
            self.skipTest("Transformers Qwen3.5 is not installed")
        for name in ("causal_conv1d_update", "causal_conv1d_fn", "torch_chunk_gated_delta_rule",
                     "torch_recurrent_gated_delta_rule"):
            original = getattr(upstream, name)
            reference = inspect.unwrap(original)
            context = patch.object(upstream, name, reference)
            context.start()
            self.addCleanup(context.stop)
        return upstream

    def test_matches_grouped_convolution_and_preserves_cache_allocation(self):
        generator = torch.Generator().manual_seed(402)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for batch, channels in ((1, 1), (1, 96), (3, 13)):
                for activation in (None, "silu", "swish"):
                    for with_bias in (False, True):
                        with self.subTest(dtype=dtype, batch=batch, channels=channels, activation=activation, bias=with_bias):
                            hidden = torch.randn(batch, channels, 1, generator=generator).to(dtype)
                            original = torch.randn(batch, channels, 4, generator=generator).to(dtype)
                            state, expected_state = original.clone(), original.clone()
                            weight = torch.randn(channels, 4, generator=generator).to(dtype)
                            bias = torch.randn(channels, generator=generator).to(dtype) if with_bias else None
                            address = state.data_ptr()
                            with torch.inference_mode():
                                expected = self.reference(hidden, expected_state, weight, bias, activation)
                                calls = self.calls
                                actual = self.optimized(hidden, state, weight, bias, activation)
                            self.assertEqual(self.calls, calls)
                            self.assertEqual(state.data_ptr(), address)
                            torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
                            torch.testing.assert_close(actual, expected, atol=2e-6 if dtype == torch.float32 else .016,
                                                       rtol=2e-6 if dtype == torch.float32 else .016)

    def test_multiple_decode_steps_preserve_window_order(self):
        state = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        expected_state = state.clone()
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11
        with torch.inference_mode():
            for step in range(7):
                hidden = torch.full((2, 3, 1), step + .25)
                actual = self.optimized(hidden, state, weight, activation="silu")
                expected = self.reference(hidden, expected_state, weight, activation="silu")
                torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
                torch.testing.assert_close(actual, expected)

    def test_unsupported_layout_dtype_training_and_extra_arguments_use_reference(self):
        for case in ("prefill", "kernel", "noncontiguous", "float64", "training", "extra"):
            with self.subTest(case=case):
                hidden = torch.randn(1, 3, 2 if case == "prefill" else 1)
                kernel = 3 if case == "kernel" else 4
                state = torch.randn(1, 3, kernel)
                weight = torch.randn(3, kernel)
                if case == "noncontiguous":
                    state = torch.randn(1, 3, 8)[..., ::2]
                if case == "float64":
                    hidden, state, weight = hidden.double(), state.double(), weight.double()
                kwargs = {"future_argument": True} if case == "extra" else {}
                calls = self.calls
                with torch.set_grad_enabled(case == "training"):
                    self.optimized(hidden, state, weight, **kwargs)
                self.assertEqual(self.calls, calls + 1)

    def test_other_architectures_are_unchanged(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4))
        original = model[0].forward
        self.assertEqual(install_optimizations(model), [])
        self.assertEqual(model[0].forward, original)

    def test_installer_changes_only_selected_instance_and_keeps_child_hooks(self):
        upstream = self.reference_qwen()
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        except ImportError:
            self.skipTest("Transformers Qwen3.5 is not installed")
        config = Qwen3_5TextConfig(hidden_size=16, linear_num_value_heads=2,
            linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
            linear_conv_kernel_dim=4, num_hidden_layers=1, layer_types=["linear_attention"])
        selected = upstream.Qwen3_5GatedDeltaNet(config, 0).eval()
        untouched = upstream.Qwen3_5GatedDeltaNet(config, 0).eval()
        original_global = upstream.causal_conv1d_update
        original_forward = untouched.forward.__func__
        labels = install_optimizations(selected)
        self.assertIn("Qwen3.5 single-token convolution (1 layers)", labels)
        self.assertEqual(install_optimizations(selected), [])
        self.assertIs(upstream.causal_conv1d_update, original_global)
        self.assertIs(untouched.forward.__func__, original_forward)

        events = []
        class Hook:
            def pre_forward(self, module): events.append("pre")
            def post_forward(self, module, output): events.append("post")
        selected.conv1d._hf_hook = Hook()
        # Equal checkpoints through both prefill and cached decode, including
        # static cache addresses and the actual model-local rebound callable.
        from transformers.cache_utils import DynamicCache
        untouched.load_state_dict(selected.state_dict())
        selected_cache, reference_cache = DynamicCache(config=config), DynamicCache(config=config)
        with torch.inference_mode():
            hidden = torch.randn(1, 3, 16)
            torch.testing.assert_close(selected(hidden, cache_params=selected_cache),
                                       untouched(hidden, cache_params=reference_cache))
            address = selected_cache.layers[0].conv_states[0].data_ptr()
            for _ in range(3):
                token = torch.randn(1, 1, 16)
                torch.testing.assert_close(selected(token, cache_params=selected_cache),
                                           untouched(token, cache_params=reference_cache), atol=2e-6, rtol=2e-5)
                self.assertEqual(selected_cache.layers[0].conv_states[0].data_ptr(), address)
                torch.testing.assert_close(selected_cache.layers[0].conv_states[0],
                                           reference_cache.layers[0].conv_states[0], atol=0, rtol=0)
        self.assertEqual(events, ["pre", "post"] * 4)

    def test_installer_preserves_outer_accelerate_forward(self):
        upstream = self.reference_qwen()
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        except ImportError:
            self.skipTest("Transformers Qwen3.5 is not installed")
        config = Qwen3_5TextConfig(hidden_size=16, linear_num_value_heads=2,
            linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
            linear_conv_kernel_dim=4, num_hidden_layers=1, layer_types=["linear_attention"])
        mixer = upstream.Qwen3_5GatedDeltaNet(config, 0).eval()
        original = mixer.forward
        mixer._old_forward = original
        mixer._hf_hook = object()
        def outer(*args, **kwargs):
            return mixer._old_forward(*args, **kwargs)
        mixer.forward = outer
        self.assertTrue(install_optimizations(mixer))
        self.assertIs(mixer.forward, outer)
        self.assertIsNot(mixer._old_forward.__func__, original.__func__)

    def test_cpu_model_uses_reference_even_when_optional_gpu_kernels_are_installed(self):
        # Deliberately do not patch upstream dispatch here. With FLA installed,
        # the original functions select GPU kernels even on a CPU checkpoint.
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            from transformers.cache_utils import DynamicCache
        except ImportError:
            self.skipTest("Transformers Qwen3.5 is not installed")
        config = Qwen3_5TextConfig(hidden_size=16, linear_num_value_heads=2,
            linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
            linear_conv_kernel_dim=4, num_hidden_layers=1, layer_types=["linear_attention"])
        mixer = upstream.Qwen3_5GatedDeltaNet(config, 0).eval()
        originals = {name: getattr(upstream, name) for name in (
            "torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule")}
        labels = install_optimizations(mixer)
        self.assertNotIn("Qwen3.5 fused linear attention", labels)
        self.assertIn("Qwen3.5 CPU reference linear attention (1 layers)", labels)
        cache = DynamicCache(config=config)
        with torch.inference_mode():
            prefill = mixer(torch.randn(1, 5, 16), cache_params=cache)
            decode = mixer(torch.randn(1, 1, 16), cache_params=cache)
        self.assertEqual(prefill.device.type, "cpu")
        self.assertEqual(decode.shape, (1, 1, 16))
        self.assertTrue(torch.isfinite(decode).all())
        for name, function in originals.items():
            self.assertIs(getattr(upstream, name), function)

    def test_reinstall_can_disable_fused_kernels_without_losing_conv_or_hooks(self):
        upstream = self.reference_qwen()
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.cache_utils import DynamicCache
        config = Qwen3_5TextConfig(hidden_size=16, linear_num_value_heads=2,
            linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
            linear_conv_kernel_dim=4, num_hidden_layers=1, layer_types=["linear_attention"])
        mixer = upstream.Qwen3_5GatedDeltaNet(config, 0).eval()
        install_optimizations(mixer)
        before = mixer.forward.__func__
        labels = install_optimizations(mixer, allow_fused=False)
        self.assertNotIn("Qwen3.5 fused linear attention", labels)
        self.assertIn("Qwen3.5 single-token convolution (1 layers)", labels)
        self.assertIsNot(mixer.forward.__func__, before)
        self.assertFalse(mixer._tuna_allow_fused)
        self.assertEqual(install_optimizations(mixer, allow_fused=False), [])
        cache = DynamicCache(config=config)
        with torch.inference_mode():
            mixer(torch.randn(1, 3, 16), cache_params=cache)
            self.assertTrue(torch.isfinite(mixer(torch.randn(1, 1, 16), cache_params=cache)).all())


if __name__ == "__main__":
    unittest.main()
