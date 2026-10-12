from math import isfinite
from random import Random

from network import gradients


def train(data, steps=3000, learning_rate=0.1, seed=7, sample_weights=None):
    if not data:
        raise ValueError("data must not be empty")
    if sample_weights is None:
        sample_weights = [1.0] * len(data)
    if len(sample_weights) != len(data):
        raise ValueError("provide one weight per example")
    if any(not isfinite(w) or w < 0 for w in sample_weights):
        raise ValueError("weights must be finite and nonnegative")

    rng = Random(seed)
    params = [rng.uniform(-1.0, 1.0) for _ in range(9)]

    for _ in range(steps):
        total = [0.0] * len(params)
        for (x, target), weight in zip(data, sample_weights):
            for i, gradient in enumerate(gradients(params, x, target)):
                total[i] += weight * gradient

        params = [
            p - learning_rate * g / len(data)
            for p, g in zip(params, total)
        ]

    return params


def main():
    data = [
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 1.0),
        ((1.0, 0.0), 1.0),
        ((1.0, 1.0), 0.0),
    ]
    return train(data)


if __name__ == "__main__":
    main()
