from network import gradients, hessian_vector_product, prediction_gradients
from train_xor import train


def one_step_influence(data, example_index, query, learning_rate=0.1,
                       seed=7, sample_weights=None):
    initial = train(data, steps=0, seed=seed, sample_weights=sample_weights)
    if not 0 <= example_index < len(data):
        raise IndexError("example index is out of range")

    updated = train(data, steps=1, learning_rate=learning_rate,
                    seed=seed, sample_weights=sample_weights)
    x, target = data[example_index]
    example_gradient = gradients(initial, x, target)
    query_gradient = prediction_gradients(updated, query)

    return -learning_rate * sum(
        a * b for a, b in zip(example_gradient, query_gradient)
    ) / len(data)


def training_influence(data, example_index, query, steps=3000,
                       learning_rate=0.1, seed=7, sample_weights=None):
    params = train(data, steps=0, seed=seed, sample_weights=sample_weights)
    if not 0 <= example_index < len(data):
        raise IndexError("example index is out of range")
    if sample_weights is None:
        sample_weights = [1.0] * len(data)
    sensitivity = [0.0] * len(params)

    for _ in range(steps):
        total_gradient = [0.0] * len(params)
        total_change = [0.0] * len(params)

        for index, ((x, target), weight) in enumerate(zip(data, sample_weights)):
            gradient = gradients(params, x, target)
            change = hessian_vector_product(params, x, target, sensitivity)
            for i in range(len(params)):
                total_gradient[i] += weight * gradient[i]
                total_change[i] += weight * change[i]
                if index == example_index:
                    total_change[i] += gradient[i]

        params = [
            p - learning_rate * g / len(data)
            for p, g in zip(params, total_gradient)
        ]
        sensitivity = [
            s - learning_rate * change / len(data)
            for s, change in zip(sensitivity, total_change)
        ]

    return sum(
        g * s for g, s in zip(prediction_gradients(params, query), sensitivity)
    )
