import csv

from datasets import regression, xor
from influence import training_sensitivity
from network import predict, prediction_gradients
from train_xor import train


def compare(data, example_index, query, changes, steps=1000,
            learning_rate=0.1, seed=7):
    return compare_queries(data, example_index, [query], changes,
                           steps, learning_rate, seed)


def compare_queries(data, example_index, queries, changes, steps=1000,
                    learning_rate=0.1, seed=7):
    params, sensitivity = training_sensitivity(
        data, example_index, steps, learning_rate, seed
    )
    baselines = [predict(params, query) for query in queries]
    influences = [sum(g * s for g, s in zip(prediction_gradients(params, query), sensitivity))
                  for query in queries]
    rows = []

    for change in changes:
        weights = [1.0] * len(data)
        weights[example_index] += change
        updated = params if change == 0 else train(
            data, steps=steps, learning_rate=learning_rate,
            seed=seed, sample_weights=weights
        )
        for index, (query, baseline, influence) in enumerate(zip(queries, baselines, influences)):
            predicted_change = influence * change
            actual_change = predict(updated, query) - baseline
            rows.append({
                "seed": seed,
                "steps": steps,
                "learning_rate": learning_rate,
                "example_index": example_index,
                "query_index": index,
                "query_x0": query[0],
                "query_x1": query[1],
                "weight_change": change,
                "baseline_prediction": baseline,
                "sensitivity": influence,
                "predicted_change": predicted_change,
                "actual_change": actual_change,
                "absolute_error": abs(predicted_change - actual_change),
            })

    return rows


def main(output="influence.csv"):
    changes = [-1.0, -0.5, -0.1, -0.01, -0.001, 0.0,
               0.001, 0.01, 0.1, 0.5, 1.0]
    rows = []
    for name, dataset in (("xor", xor), ("regression", regression)):
        data, queries = dataset()
        for seed in (0, 7, 19):
            for steps in (10, 100, 1000):
                for example_index in range(len(data)):
                    comparisons = compare_queries(data, example_index, queries, changes,
                                                  steps=steps, seed=seed)
                    rows.extend({"dataset": name, **row} for row in comparisons)

    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    main()
