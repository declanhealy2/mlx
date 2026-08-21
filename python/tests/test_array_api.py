import unittest

import mlx.core as mx


class TestArrayApi(unittest.TestCase):
    def test_namespace_info_capabilities(self):
        info = mx.__array_namespace_info__()
        self.assertEqual(
            info.capabilities(),
            {
                "boolean indexing": False,
                "data-dependent shapes": False,
                "max dimensions": None,
            },
        )

    def test_namespace_info_devices(self):
        info = mx.__array_namespace_info__()
        devices = info.devices()
        self.assertIsInstance(devices, tuple)
        self.assertIn(info.default_device(), devices)
        self.assertIn(mx.cpu, devices)

    def test_namespace_info_dtypes(self):
        info = mx.__array_namespace_info__()
        self.assertEqual(
            info.default_dtypes(),
            {
                "real floating": mx.float32,
                "complex floating": mx.complex64,
                "integral": mx.int32,
                "indexing": mx.int32,
            },
        )
        self.assertEqual(info.dtypes(kind="bool"), {"bool": mx.bool_})
        self.assertNotIn("complex128", info.dtypes())
        if mx.gpu in info.devices():
            self.assertNotIn("float64", info.dtypes(device=mx.gpu))

    def test_namespace_info_invalid_kind(self):
        with self.assertRaises(ValueError):
            mx.__array_namespace_info__().dtypes(kind="not-a-kind")

    def test_moveaxis_sequence_axes(self):
        x = mx.arange(24).reshape((2, 3, 4))
        moved = mx.moveaxis(x, (0, 2), (2, 0))
        expected = mx.transpose(x, (2, 1, 0))
        self.assertTrue(mx.array_equal(moved, expected).item())
        self.assertTrue(mx.array_equal(mx.moveaxis(x, (), ()), x).item())

        with self.assertRaises(ValueError):
            mx.moveaxis(x, (0,), (0, 1))
        with self.assertRaises(ValueError):
            mx.moveaxis(x, (0, 0), (1, 2))


if __name__ == "__main__":
    unittest.main()
