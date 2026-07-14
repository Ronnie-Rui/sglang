import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.hisparse_hook import (
    apply_runtime_sparse_cuda_graph_defaults,
)
from sglang.srt.mem_cache.sparsity import parse_runtime_sparse_config
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _args(config, *, page_size=16):
    return SimpleNamespace(
        hisparse_config=json.dumps(config),
        page_size=page_size,
    )


class TestRuntimeSparseConfig(unittest.TestCase):
    def _parse(self, config, *, page_size=16):
        with patch(
            "sglang.srt.arg_groups.overrides.attention_backends_of",
            return_value=("fa3", "fa3"),
        ):
            return parse_runtime_sparse_config(_args(config, page_size=page_size))

    def test_parses_quest_graph_metadata_options(self):
        config = self._parse(
            {
                "algorithm": "quest",
                "backend": "fa3",
                "page_size": 16,
                "sparsity_ratio": 0.125,
                "num_recent_pages": 4,
                "enable_cuda_graph_retrieval": True,
                "use_direct_fa_metadata_kernel": True,
                "cuda_graph_context_buckets": [8192, 32768],
            }
        )

        self.assertEqual(config.algorithm, "quest")
        self.assertEqual(config.backend, "fa3")
        self.assertEqual(config.page_size, 16)
        self.assertEqual(config.sparse_extra_config["sparsity_ratio"], 0.125)
        self.assertTrue(config.sparse_extra_config["enable_cuda_graph_retrieval"])

    def test_rejects_runtime_page_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "page_size.*match"):
            self._parse(
                {"algorithm": "quest", "backend": "fa3", "page_size": 8},
                page_size=16,
            )

    def test_rejects_invalid_ratio_and_non_boolean_kernel_flag(self):
        invalid = (
            ({"sparsity_ratio": 0}, "sparsity_ratio"),
            ({"sparsity_ratio": 1.1}, "sparsity_ratio"),
            (
                {"use_direct_fa_metadata_kernel": 1},
                "use_direct_fa_metadata_kernel",
            ),
            ({"enable_cuda_graph_retrieval": "yes"}, "enable_cuda_graph_retrieval"),
        )
        for extra, message in invalid:
            config = {
                "algorithm": "quest",
                "backend": "fa3",
                "page_size": 16,
                **extra,
            }
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self._parse(config)


class TestRuntimeSparseCudaGraphDefaults(unittest.TestCase):
    def test_unlocked_defaults_use_breakable_decode_and_disable_prefill(self):
        args = SimpleNamespace(
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.FULL),
                prefill=PhaseConfig(backend=Backend.BREAKABLE),
            ),
            _cuda_graph_config_locked=set(),
        )
        with (
            patch(
                "sglang.srt.arg_groups.hisparse_hook.use_runtime_sparse_attention",
                return_value=True,
            ),
            patch(
                "sglang.srt.arg_groups.hisparse_hook.get_hisparse_algorithm",
                return_value="quest",
            ),
        ):
            apply_runtime_sparse_cuda_graph_defaults(args)

        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.BREAKABLE)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)

    def test_explicit_graph_backends_are_preserved(self):
        args = SimpleNamespace(
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.FULL),
                prefill=PhaseConfig(backend=Backend.BREAKABLE),
            ),
            _cuda_graph_config_locked={
                (Phase.DECODE, "backend"),
                (Phase.PREFILL, "backend"),
            },
        )
        with (
            patch(
                "sglang.srt.arg_groups.hisparse_hook.use_runtime_sparse_attention",
                return_value=True,
            ),
            patch(
                "sglang.srt.arg_groups.hisparse_hook.get_hisparse_algorithm",
                return_value="quest",
            ),
        ):
            apply_runtime_sparse_cuda_graph_defaults(args)

        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)


if __name__ == "__main__":
    unittest.main()
