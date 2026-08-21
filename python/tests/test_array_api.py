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


if __name__ == "__main__":
    unittest.main()
