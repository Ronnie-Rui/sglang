"""Unit tests for srt/configs/linear_attn_model_registry.py"""

import unittest

from sglang.srt.configs.linear_attn_model_registry import (
    _LINEAR_ATTN_MODEL_REGISTRY,
    LinearAttnModelSpec,
    get_linear_attn_config,
    get_linear_attn_spec_by_arch,
    import_backend_class,
    register_linear_attn_model,
    resolve_hybrid_gdn_config,
    resolve_hybrid_lightning_config,
    resolve_kimi_linear_config,
    resolve_mamba2_config,
    resolve_mambaish_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


# Dummy config classes for testing
class FakeLinearAttnConfig:
    full_attention_layer_ids = [0, 2, 4]


class FakeVLMWrapperConfig:
    """Simulates a VLM wrapper that has get_text_config()."""

    def __init__(self):
        self._text_config = FakeLinearAttnConfig()

    def get_text_config(self):
        return self._text_config


class AnotherConfig:
    pass


class TestLinearAttnModelRegistry(CustomTestCase):
    def setUp(self):
        # Save and clear the global registry between tests
        self._saved_registry = list(_LINEAR_ATTN_MODEL_REGISTRY)
        _LINEAR_ATTN_MODEL_REGISTRY.clear()

    def tearDown(self):
        _LINEAR_ATTN_MODEL_REGISTRY.clear()
        _LINEAR_ATTN_MODEL_REGISTRY.extend(self._saved_registry)

    def _make_spec(self, **overrides):
        defaults = dict(
            config_class=FakeLinearAttnConfig,
            backend_class_name="sglang.srt.layers.attention.triton_backend.TritonAttnBackend",
            arch_names=["FakeModelForCausalLM"],
        )
        defaults.update(overrides)
        return LinearAttnModelSpec(**defaults)

    def test_register_and_lookup_by_config(self):
        spec = self._make_spec()
        register_linear_attn_model(spec)

        hf_config = FakeLinearAttnConfig()
        result = get_linear_attn_config(hf_config)
        self.assertIsNotNone(result)
        self.assertIs(result[0], spec)
        self.assertIs(result[1], hf_config)

    def test_lookup_no_match(self):
        spec = self._make_spec()
        register_linear_attn_model(spec)

        result = get_linear_attn_config(AnotherConfig())
        self.assertIsNone(result)

    def test_lookup_empty_registry(self):
        result = get_linear_attn_config(FakeLinearAttnConfig())
        self.assertIsNone(result)

    def test_unwrap_text_config(self):
        spec = self._make_spec(unwrap_text_config=True)
        register_linear_attn_model(spec)

        vlm_config = FakeVLMWrapperConfig()
        result = get_linear_attn_config(vlm_config)
        self.assertIsNotNone(result)
        self.assertIs(result[0], spec)
        # The resolved config should be the inner text config
        self.assertIsInstance(result[1], FakeLinearAttnConfig)
        self.assertIs(result[1], vlm_config._text_config)

    def test_unwrap_text_config_no_match(self):
        """unwrap_text_config=False should not call get_text_config()."""
        spec = self._make_spec(unwrap_text_config=False)
        register_linear_attn_model(spec)

        vlm_config = FakeVLMWrapperConfig()
        # VLM wrapper itself is not a FakeLinearAttnConfig, so no match
        result = get_linear_attn_config(vlm_config)
        self.assertIsNone(result)

    def test_resolve_builtin_linear_attention_categories(self):
        from sglang.srt.configs import (
            BailingHybridConfig,
            FalconH1Config,
            KimiLinearConfig,
            Qwen3NextConfig,
        )

        gdn = Qwen3NextConfig()
        mamba2 = FalconH1Config()
        kimi = KimiLinearConfig(
            linear_attn_config={"kda_layers": [1], "full_attn_layers": [2]}
        )
        lightning = BailingHybridConfig()

        self.assertIs(resolve_hybrid_gdn_config(gdn), gdn)
        self.assertIs(resolve_mamba2_config(mamba2), mamba2)
        self.assertIs(resolve_kimi_linear_config(kimi), kimi)
        self.assertIs(resolve_hybrid_lightning_config(lightning), lightning)
        for config in (gdn, mamba2, kimi, lightning):
            self.assertIs(resolve_mambaish_config(config), config)

    def test_resolve_mamba2_preserves_model_specific_exceptions(self):
        from sglang.srt.configs import GraniteMoeHybridConfig, NemotronHConfig

        attention_only = GraniteMoeHybridConfig(
            num_hidden_layers=2, layer_types=["attention", "attention"]
        )
        mixed = GraniteMoeHybridConfig(
            num_hidden_layers=2, layer_types=["mamba", "attention"]
        )
        self.assertIsNone(resolve_mamba2_config(attention_only))
        self.assertIs(resolve_mamba2_config(mixed), mixed)

        nemotron = NemotronHConfig()
        nemotron.mtp_hybrid_override_pattern = "*E"
        self.assertIs(resolve_mamba2_config(nemotron), nemotron)
        self.assertIsNone(resolve_mamba2_config(nemotron, is_draft_worker=True))

    def test_resolve_mambaish_falls_back_to_external_registry(self):
        spec = self._make_spec()
        register_linear_attn_model(spec)
        config = FakeLinearAttnConfig()

        self.assertIs(resolve_mambaish_config(config), config)
        self.assertIsNone(resolve_mambaish_config(AnotherConfig()))

    def test_lookup_by_arch(self):
        spec = self._make_spec(arch_names=["AlphaForCausalLM", "BetaForCausalLM"])
        register_linear_attn_model(spec)

        self.assertIs(get_linear_attn_spec_by_arch("AlphaForCausalLM"), spec)
        self.assertIs(get_linear_attn_spec_by_arch("BetaForCausalLM"), spec)
        self.assertIsNone(get_linear_attn_spec_by_arch("GammaForCausalLM"))

    def test_lookup_by_arch_empty_registry(self):
        self.assertIsNone(get_linear_attn_spec_by_arch("AnyArch"))

    def test_multiple_registrations(self):
        spec_a = self._make_spec(
            config_class=FakeLinearAttnConfig,
            arch_names=["AlphaForCausalLM"],
        )
        spec_b = self._make_spec(
            config_class=AnotherConfig,
            arch_names=["BetaForCausalLM"],
        )
        register_linear_attn_model(spec_a)
        register_linear_attn_model(spec_b)

        # Config-based lookup
        self.assertIs(get_linear_attn_config(FakeLinearAttnConfig())[0], spec_a)
        self.assertIs(get_linear_attn_config(AnotherConfig())[0], spec_b)

        # Arch-based lookup
        self.assertIs(get_linear_attn_spec_by_arch("AlphaForCausalLM"), spec_a)
        self.assertIs(get_linear_attn_spec_by_arch("BetaForCausalLM"), spec_b)

    def test_first_match_wins(self):
        """When two specs match the same config class, the first registered wins."""
        spec1 = self._make_spec(backend_class_name="pkg.Backend1")
        spec2 = self._make_spec(backend_class_name="pkg.Backend2")
        register_linear_attn_model(spec1)
        register_linear_attn_model(spec2)

        result = get_linear_attn_config(FakeLinearAttnConfig())
        self.assertIs(result[0], spec1)

    def test_import_backend_class(self):
        # Import a real stdlib class to verify the mechanism
        cls = import_backend_class("collections.OrderedDict")
        from collections import OrderedDict

        self.assertIs(cls, OrderedDict)

    def test_spec_defaults(self):
        spec = LinearAttnModelSpec(
            config_class=FakeLinearAttnConfig,
            backend_class_name="pkg.mod.Cls",
        )
        self.assertEqual(spec.arch_names, [])
        self.assertTrue(spec.uses_mamba_radix_cache)
        self.assertTrue(spec.support_mamba_cache)
        self.assertFalse(spec.support_mamba_cache_extra_buffer)
        self.assertFalse(spec.unwrap_text_config)


if __name__ == "__main__":
    unittest.main()
