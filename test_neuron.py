import unittest

from neuron import gradients, loss

class GradientTests(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        cases = [
            (0.5, 0.5, 2.0, 1.0),
            (-0.3, 0.7, -1.5, 0.2),
            (0.8, -0.2, 0.0, 1.0),
        ]
        h = 1e-6
        for weight, bias, x, target in cases:
            with self.subTest(weight=weight, bias=bias, x=x, target=target):
                d_weight, d_bias = gradients(weight, bias, x, target)
                numeric_weight = (
                    loss(weight + h, bias, x, target)
                    - loss(weight - h, bias, x, target)
                ) / (2 * h)
                numeric_bias = (
                    loss(weight, bias + h, x, target)
                    - loss(weight, bias - h, x, target)
                ) / (2 * h)
                self.assertAlmostEqual(d_weight, numeric_weight, places=7)
                self.assertAlmostEqual(d_bias, numeric_bias, places=7)


if __name__ == "__main__":
    unittest.main()
