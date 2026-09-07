# Copyright © 2023-2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


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
    def test_memory_plan_does_not_change_a_direct_graph(self):
        value = mx.arange(4, dtype=mx.float32)
        result = value + 1

        first = mx.plan_memory(result)
        second = mx.plan_memory(result)

        self.assertEqual(first, second)
        self.assertGreaterEqual(*first)
        self.assertTrue(
            np.array_equal(np.asarray(result), np.arange(1, 5, dtype=np.float32))
        )

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_memory_plan_does_not_change_a_captured_graph_root(self):
        value = mx.arange(4, dtype=mx.float32)

        peak, retained = mx.plan_memory(lambda: mx.sin(value))
        result = value + 1

        self.assertGreaterEqual(peak, retained)
        self.assertTrue(
            np.array_equal(np.asarray(result), np.arange(1, 5, dtype=np.float32))
        )

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_memory_plan_multi_root_covers_real_execution(self):
        value = mx.zeros((4, 4), dtype=mx.float32)
        peak, retained = mx.plan_memory(lambda: (mx.fft.fft2(mx.sin(value)), value))
        old_limit = mx.set_cache_limit(0)
        try:
            mx.clear_cache()
            mx.reset_peak_memory()
            actual = mx.fft.fft2(mx.sin(value))
            mx.eval(actual)
            self.assertGreaterEqual(peak, mx.get_peak_memory())
            self.assertLessEqual(retained, peak)
        finally:
            mx.clear_cache()
            mx.set_cache_limit(old_limit)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_memory_plan_counts_an_evaluated_root_without_changing_it(self):
        value = mx.arange(4, dtype=mx.float32)
        mx.eval(value)
        result = mx.sin(value)

        peak, retained = mx.plan_memory(result)

        self.assertGreaterEqual(peak, retained)
        self.assertGreaterEqual(retained, value.nbytes + result.nbytes)
        self.assertTrue(
            np.allclose(np.asarray(result), np.sin(np.arange(4, dtype=np.float32)))
        )

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_memory_plan_preserves_multi_output_graphs(self):
        value = mx.arange(4, dtype=mx.float32)
        left, right = mx.split(value, [2])

        peak, retained = mx.plan_memory(left, right)

        self.assertGreaterEqual(peak, retained)
        self.assertTrue(
            np.array_equal(np.asarray(left), np.array([0, 1], dtype=np.float32))
        )
        self.assertTrue(
            np.array_equal(np.asarray(right), np.array([2, 3], dtype=np.float32))
        )

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_memory_plan_rejects_evaluation_inside_a_callable(self):
        value = mx.zeros((4,), dtype=mx.float32)

        with self.assertRaisesRegex(ValueError, "Evaluation inside a planned function"):
            mx.plan_memory(lambda: (mx.eval(value), value)[1])

        with self.assertRaisesRegex(
            ValueError, "Nested planning inside a planned function"
        ):
            mx.plan_memory(lambda: (mx.plan_memory(value), value)[1])

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_memory_plan_rejects_a_cpu_graph(self):
        q, _ = mx.linalg.qr(mx.eye(2, dtype=mx.float32), stream=mx.cpu)

        with self.assertRaisesRegex(ValueError, "requires a Metal graph"):
            mx.plan_memory(q)

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
