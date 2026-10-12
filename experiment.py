import csv

from influence import training_influence
from network import predict
from train_xor import train


def compare(data, example_index, query, changes, steps=1000,
            learning_rate=0.1, seed=7):
    sensitivity = training_influence(
        data, example_index, query, steps, learning_rate, seed
    )
    params = train(data, steps=steps, learning_rate=learning_rate, seed=seed)
    baseline = predict(params, query)
    rows = []

    for change in changes:
        weights = [1.0] * len(data)
        weights[example_index] += change
        updated = train(data, steps=steps, learning_rate=learning_rate,
                        seed=seed, sample_weights=weights)
        predicted_change = sensitivity * change
        actual_change = predict(updated, query) - baseline
        rows.append({
            "seed": seed,
            "steps": steps,
            "learning_rate": learning_rate,
            "example_index": example_index,
            "query_x0": query[0],
            "query_x1": query[1],
            "weight_change": change,
            "predicted_change": predicted_change,
            "actual_change": actual_change,
            "absolute_error": abs(predicted_change - actual_change),
        })

    return rows


def main(output="influence.csv"):
    data = [
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 1.0),
        ((1.0, 0.0), 1.0),
        ((1.0, 1.0), 0.0),
    ]
    changes = [-1.0, -0.5, -0.1, -0.01, -0.001, 0.0,
               0.001, 0.01, 0.1, 0.5, 1.0]
    rows = []
    for seed in (0, 7, 19):
        for steps in (10, 100, 1000):
            rows.extend(compare(data, 1, (0.0, 1.0), changes,
                                steps=steps, seed=seed))

    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    main()
