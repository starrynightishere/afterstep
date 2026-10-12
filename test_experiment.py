import unittest

from datasets import regression, xor
from experiment import compare_queries
from influence import training_sensitivity
from network import predict
from train_xor import train


class ExperimentTests(unittest.TestCase):
    def test_queries_are_unseen(self):
        for dataset in (xor, regression):
            data, queries = dataset()
            self.assertEqual(len(set(queries)), 9)
            self.assertTrue(set(queries).isdisjoint(x for x, _ in data))

    def test_parameter_sensitivity_matches_retraining(self):
        data, _ = regression()
        weights = [0.8 + i / 20 for i in range(len(data))]
        h = 1e-5
        for index in (0, 5, 11):
            with self.subTest(index=index):
                params, sensitivity = training_sensitivity(
                    data, index, steps=30, seed=19, sample_weights=weights
                )
                self.assertEqual(params, train(data, steps=30, seed=19, sample_weights=weights))
                plus, minus = weights.copy(), weights.copy()
                plus[index] += h
                minus[index] -= h
                high = train(data, steps=30, seed=19, sample_weights=plus)
                low = train(data, steps=30, seed=19, sample_weights=minus)
                for actual, a, b in zip(sensitivity, high, low):
                    self.assertAlmostEqual(actual, (a - b) / (2 * h), places=7)

    def test_batched_queries_match_independent_retraining(self):
        for dataset in (xor, regression):
            data, queries = dataset()
            baseline = train(data, steps=15, seed=19)
            for index in range(len(data)):
                rows = compare_queries(data, index, queries, [-0.1, 0.0, 0.1],
                                       steps=15, seed=19)
                self.assertEqual(len(rows), 27)
                for row in rows:
                    with self.subTest(dataset=dataset.__name__, index=index,
                                      query=row["query_index"], change=row["weight_change"]):
                        query = queries[row["query_index"]]
                        weights = [1.0] * len(data)
                        weights[index] += row["weight_change"]
                        retrained = train(data, steps=15, seed=19, sample_weights=weights)
                        expected = predict(retrained, query) - predict(baseline, query)
                        self.assertEqual(row["baseline_prediction"], predict(baseline, query))
                        self.assertAlmostEqual(row["actual_change"], expected, places=14)
                        self.assertEqual(row["predicted_change"],
                                         row["sensitivity"] * row["weight_change"])


if __name__ == "__main__":
    unittest.main()
