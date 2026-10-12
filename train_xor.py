from random import Random

from network import gradients


def train(data, steps=3000, learning_rate=0.1, seed=7):
    rng = Random(seed)
    params = [rng.uniform(-1.0, 1.0) for _ in range(9)]

    for _ in range(steps):
        total = [0.0] * len(params)
        for x, target in data:
            for i, gradient in enumerate(gradients(params, x, target)):
                total[i] += gradient

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
