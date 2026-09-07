# Copyright © 2023-2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestMemory(mlx_tests.MLXTestCase):
    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_memory_plan_uses_native_graph_schedule(self):
        graphs = {
            "unary": (
                lambda: mx.sin(mx.arange(16, dtype=mx.float32).reshape(4, 4)) + 1,
                68,
            ),
            "reduction": (
                lambda: mx.sum(mx.arange(16, dtype=mx.float32).reshape(4, 4), axis=1),
                80,
            ),
            "matmul": (
                lambda: (
                    mx.arange(16, dtype=mx.float32).reshape(4, 4)
                    @ mx.arange(16, dtype=mx.float32).reshape(4, 4).T
                ),
                192,
            ),
            "radix_fft": (lambda: mx.fft.fft(mx.arange(8, dtype=mx.float32)), 160),
            "rader_fft": (lambda: mx.fft.fft(mx.arange(7, dtype=mx.float32)), 140),
        }
        for name, (graph, expected_peak) in graphs.items():
            with self.subTest(name=name):
                peak, retained = mx.plan_memory(graph())
                self.assertEqual(peak, expected_peak)
                self.assertLessEqual(retained, peak)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_callable_memory_plan_includes_internal_evaluation_barriers(self):
        def graph():
            value = mx.zeros((4, 4), dtype=mx.float32)
            mx.eval(value)
            return mx.sin(value)

        peak, retained = mx.plan_memory(graph)

        self.assertEqual(peak, 68)
        self.assertEqual(retained, 0)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_callable_memory_plan_preserves_external_root_ownership(self):
        value = mx.zeros((4, 4), dtype=mx.float32)

        peak, retained = mx.plan_memory(
            lambda: mx.fft.fft2(mx.sin(value)), roots=(value,)
        )

        self.assertEqual(peak, 644)
        self.assertEqual(retained, 68)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_callable_memory_plan_composes_backend_lifecycle_effects(self):
        def graph():
            with mx.stream(mx.gpu):
                key = mx.random.key(7)
                value = mx.random.normal((2, 2), key=key)
            mx.eval(value)
            mx.clear_cache()
            return mx.sin(value)

        peak, retained = mx.plan_memory(graph)

        self.assertEqual(peak, 48)
        self.assertEqual(retained, 0)

    def test_memory_info(self):
        old_limit = mx.set_cache_limit(0)

        a = mx.zeros((4096,))
        mx.eval(a)
        del a
        self.assertEqual(mx.get_cache_memory(), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), old_limit)

        old_limit = mx.set_memory_limit(10)
        self.assertEqual(mx.set_memory_limit(old_limit), 10)
        self.assertEqual(mx.set_memory_limit(old_limit), old_limit)

        # Query active and peak memory
        a = mx.zeros((4096,))
        mx.eval(a)
        mx.synchronize()
        active_mem = mx.get_active_memory()
        self.assertTrue(active_mem >= 4096 * 4)

        b = mx.zeros((4096,))
        mx.eval(b)
        del b
        mx.synchronize()

        new_active_mem = mx.get_active_memory()
        self.assertEqual(new_active_mem, active_mem)
        peak_mem = mx.get_peak_memory()
        self.assertTrue(peak_mem >= 4096 * 8)

        if mx.metal.is_available():
            cache_mem = mx.get_cache_memory()
            self.assertTrue(cache_mem >= 4096 * 4)

        mx.clear_cache()
        self.assertEqual(mx.get_cache_memory(), 0)

        mx.reset_peak_memory()
        self.assertEqual(mx.get_peak_memory(), 0)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_wired_memory(self):
        old_limit = mx.set_wired_limit(1000)
        old_limit = mx.set_wired_limit(0)
        self.assertEqual(old_limit, 1000)

        max_size = mx.device_info(mx.gpu)["max_recommended_working_set_size"]
        with self.assertRaises(ValueError):
            mx.set_wired_limit(max_size + 10)

    def test_active_memory_count(self):
        mx.synchronize()
        mx.clear_cache()
        init_mem = mx.get_active_memory()
        a = mx.zeros((128, 128))
        mx.eval(a)
        mx.synchronize()
        del a
        a = mx.zeros((90, 128))
        mx.eval(a)
        mx.synchronize()
        del a
        self.assertEqual(init_mem, mx.get_active_memory())


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
