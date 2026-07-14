import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as configurator_module
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKVCacheAllocatorSelection(unittest.TestCase):
    @staticmethod
    def _configurator(*, algorithm: str, page_size: int):
        configurator = object.__new__(KVCacheConfigurator)
        configurator.server_args = SimpleNamespace(
            disaggregation_mode="null",
            enable_hisparse=True,
            hisparse_config=json.dumps({"algorithm": algorithm}),
            page_size=page_size,
            dcp_size=1,
        )
        configurator.is_hybrid_swa = False
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.device = "cpu"
        return configurator

    @staticmethod
    def _sizes():
        return SimpleNamespace(
            max_total_num_tokens=64,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )

    def test_runtime_quest_uses_regular_paged_allocator(self):
        configurator = self._configurator(algorithm="quest", page_size=16)
        expected_allocator = Mock()

        with (
            patch.object(
                configurator_module.current_platform,
                "is_out_of_tree",
                return_value=False,
            ),
            patch.object(configurator_module, "_is_npu", False),
            patch.object(
                configurator_module,
                "PagedTokenToKVPoolAllocator",
                return_value=expected_allocator,
            ) as paged_allocator,
            patch.object(
                configurator_module,
                "HiSparseTokenToKVPoolAllocator",
            ) as hisparse_allocator,
        ):
            actual_allocator = configurator._build_token_to_kv_pool_allocator(
                sizes=self._sizes(),
                token_to_kv_pool=Mock(),
                is_dsv4_model=False,
                req_to_token_pool=SimpleNamespace(),
                token_to_kv_pool_allocator=None,
            )

        self.assertIs(actual_allocator, expected_allocator)
        paged_allocator.assert_called_once()
        hisparse_allocator.assert_not_called()

    def test_legacy_dsa_keeps_hisparse_allocator(self):
        configurator = self._configurator(algorithm="dsa", page_size=1)
        expected_allocator = Mock()

        with (
            patch.object(
                configurator_module.current_platform,
                "is_out_of_tree",
                return_value=False,
            ),
            patch.object(configurator_module, "_is_npu", False),
            patch(
                "sglang.srt.mem_cache.sparsity.parse_hisparse_config",
                return_value=SimpleNamespace(host_to_device_ratio=4),
            ),
            patch.object(
                configurator_module,
                "HiSparseTokenToKVPoolAllocator",
                return_value=expected_allocator,
            ) as hisparse_allocator,
            patch.object(
                configurator_module,
                "PagedTokenToKVPoolAllocator",
            ) as paged_allocator,
        ):
            actual_allocator = configurator._build_token_to_kv_pool_allocator(
                sizes=self._sizes(),
                token_to_kv_pool=Mock(),
                is_dsv4_model=False,
                req_to_token_pool=SimpleNamespace(),
                token_to_kv_pool_allocator=None,
            )

        self.assertIs(actual_allocator, expected_allocator)
        hisparse_allocator.assert_called_once()
        paged_allocator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
