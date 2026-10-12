import unittest

from train_line import mean_gradients, mean_loss


class MeanGradientTests(unittest.TestCase):
    def test_gradients_match_changes_in_mean_loss(self):
        data = [(-2.0, 1.0), (0.5, -1.0), (3.0, 2.0)]
        weight, bias = 0.3, -0.2
        h = 1e-6
        d_weight, d_bias = mean_gradients(weight, bias, data)
        numeric_weight = (
            mean_loss(weight + h, bias, data)
            - mean_loss(weight - h, bias, data)
        ) / (2 * h)
        numeric_bias = (
            mean_loss(weight, bias + h, data)
            - mean_loss(weight, bias - h, data)
        ) / (2 * h)
        self.assertAlmostEqual(d_weight, numeric_weight, places=7)
        self.assertAlmostEqual(d_bias, numeric_bias, places=7)

    def test_repeating_the_dataset_does_not_change_the_average(self):
        data = [(-1.0, 2.0), (2.0, -0.5)]
        self.assertAlmostEqual(
            mean_loss(0.2, 0.1, data), mean_loss(0.2, 0.1, data * 2)
        )
        original = mean_gradients(0.2, 0.1, data)
        repeated = mean_gradients(0.2, 0.1, data * 2)
        for first, second in zip(original, repeated):
            self.assertAlmostEqual(first, second)


if __name__ == "__main__":
    unittest.main()
