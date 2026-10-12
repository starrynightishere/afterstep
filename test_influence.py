import math
from random import Random
import unittest

from influence import one_step_influence, training_influence
from network import gradients, hessian_vector_product, loss, predict
from network import prediction_gradients
from train_xor import train


DATA = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0),
        ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]


class InfluenceTests(unittest.TestCase):
    def assert_close(self, actual, expected, atol=1e-7, rtol=1e-5):
        self.assertTrue(math.isfinite(actual) and math.isfinite(expected))
        self.assertTrue(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
                        f"{actual} != {expected}")

    def test_network_derivatives(self):
        rng = Random(23)
        for case in range(10):
            params = [rng.uniform(-1, 1) for _ in range(9)]
            x = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
            target = rng.uniform(-1, 1)
            vector = [rng.uniform(-1, 1) for _ in params]
            prediction_gradient = prediction_gradients(params, x)
            loss_gradient = gradients(params, x, target)
            product = hessian_vector_product(params, x, target, vector)
            for i in range(len(params)):
                with self.subTest(case=case, parameter=i):
                    h = 1e-6
                    plus, minus = params.copy(), params.copy()
                    plus[i] += h
                    minus[i] -= h
                    self.assert_close(prediction_gradient[i],
                                      (predict(plus, x) - predict(minus, x)) / (2 * h))
                    self.assert_close(loss_gradient[i],
                                      (loss(plus, x, target) - loss(minus, x, target)) / (2 * h))

                    h = 1e-4
                    values = {}
                    for a in (-1, 1):
                        for b in (-1, 1):
                            shifted = [p + a * h * v for p, v in zip(params, vector)]
                            shifted[i] += b * h
                            values[a, b] = loss(shifted, x, target)
                    numeric = (values[1, 1] - values[1, -1]
                               - values[-1, 1] + values[-1, -1]) / (4 * h * h)
                    self.assert_close(product[i], numeric, atol=3e-6, rtol=3e-4)

    def test_influence_matches_retraining(self):
        h = 1e-5
        for seed in (0, 7, 19):
            for steps in (0, 1, 10, 100, 1000):
                for weights in ([1.0] * 4, [0.5, 1.3, 0.8, 2.0]):
                    for index in range(len(DATA)):
                        with self.subTest(seed=seed, steps=steps, weights=weights, index=index):
                            query = (0.25, 0.75)
                            actual = training_influence(DATA, index, query, steps=steps,
                                                        seed=seed, sample_weights=weights)
                            plus, minus = weights.copy(), weights.copy()
                            plus[index] += h
                            minus[index] -= h
                            high = train(DATA, steps=steps, seed=seed, sample_weights=plus)
                            low = train(DATA, steps=steps, seed=seed, sample_weights=minus)
                            numeric = (predict(high, query) - predict(low, query)) / (2 * h)
                            self.assert_close(actual, numeric, atol=2e-7, rtol=2e-4)
                            if steps == 1:
                                direct = one_step_influence(DATA, index, query, seed=seed,
                                                            sample_weights=weights)
                                self.assert_close(actual, direct, atol=1e-12)

    def test_no_update(self):
        initial = train(DATA, steps=0)
        self.assertEqual(train(DATA, steps=10, learning_rate=0), initial)
        self.assertEqual(train(DATA, steps=10, sample_weights=[0.0] * 4), initial)
        self.assertEqual(training_influence(DATA, 1, (0, 1), steps=10, learning_rate=0), 0)

    def test_invalid_settings(self):
        for function, args in ((train, (DATA,)),
                               (training_influence, (DATA, 1, (0, 1)))):
            for steps in (-1, 1.5, True):
                with self.subTest(function=function.__name__, steps=steps):
                    with self.assertRaises(ValueError):
                        function(*args, steps=steps)
        for function, args in ((train, (DATA,)),
                               (training_influence, (DATA, 1, (0, 1))),
                               (one_step_influence, (DATA, 1, (0, 1)))):
            for rate in (-0.1, float("nan"), float("inf")):
                with self.subTest(function=function.__name__, rate=rate):
                    with self.assertRaises(ValueError):
                        function(*args, learning_rate=rate)

    def test_invalid_index(self):
        for function in (one_step_influence, training_influence):
            for index, error in ((-1, IndexError), (4, IndexError),
                                 (0.5, TypeError), (True, TypeError)):
                with self.subTest(function=function.__name__, index=index):
                    with self.assertRaises(error):
                        function(DATA, index, (0, 1))


if __name__ == "__main__":
    unittest.main()
