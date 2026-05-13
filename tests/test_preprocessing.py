import unittest

import numpy as np

from ibamlkit.training import ConstantFactorTransform


class PreprocessingTests(unittest.TestCase):
    def test_constant_factor_transform_supports_out_buffer(self) -> None:
        transform = ConstantFactorTransform(0.25)
        x = np.asarray([[4.0, 8.0], [12.0, 16.0]], dtype=np.float32)
        transform.fit(x)

        out = np.empty_like(x)
        result = transform.inverse_transform(x, out=out)

        self.assertIs(result, out)
        self.assertTrue(np.allclose(out, [[16.0, 32.0], [48.0, 64.0]]))


if __name__ == "__main__":
    unittest.main()
