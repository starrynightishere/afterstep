import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with open(path, newline="") as file:
        return [{key: value if key == "dataset" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(file)]


def summarize(rows):
    summary = []
    for dataset in ("xor", "regression"):
        for steps in (10, 100, 1000):
            for change in (0.001, 0.01, 0.1, 0.5, 1.0):
                group = [r for r in rows if r["dataset"] == dataset
                         and r["steps"] == steps and abs(r["weight_change"]) == change]
                summary.append({
                    "dataset": dataset,
                    "steps": steps,
                    "weight_change_magnitude": change,
                    "comparisons": len(group),
                    "mean_absolute_error": mean(r["absolute_error"] for r in group),
                    "max_absolute_error": max(r["absolute_error"] for r in group),
                    "no_change_mean_absolute_error": mean(abs(r["actual_change"]) for r in group),
                })
    return summary


def draw(rows, summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    fig.suptitle("How well does example influence predict retraining?", fontsize=16)
    colors = ("#8b9ba7", "#3986a3", "#16675f")

    for column, dataset in enumerate(("xor", "regression")):
        selected = [r for r in rows if r["dataset"] == dataset
                    and r["seed"] == 7 and r["steps"] == 1000 and r["weight_change"] == 0]
        examples = sorted({int(r["example_index"]) for r in selected})
        queries = sorted({int(r["query_index"]) for r in selected})
        values = {(int(r["query_index"]), int(r["example_index"])): r["sensitivity"]
                  for r in selected}
        coordinates = {int(r["query_index"]): f'({r["query_x0"]:g}, {r["query_x1"]:g})'
                       for r in selected}
        matrix = [[values[q, e] for e in examples] for q in queries]
        limit = max(abs(value) for row in matrix for value in row) or 1.0
        ax = axes[0, column]
        heatmap = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(xticks=examples, yticks=queries, yticklabels=[coordinates[q] for q in queries],
               xlabel="training example index", ylabel="unseen query (x, y)",
               title=f"{dataset.upper() if dataset == 'xor' else 'Smooth regression'}\n"
                     "sensitivity · seed 7 · 1,000 updates")
        fig.colorbar(heatmap, ax=ax, shrink=0.8, label="prediction derivative w.r.t. example weight")

        ax = axes[1, column]
        for steps, color in zip((10, 100, 1000), colors):
            group = [r for r in summary if r["dataset"] == dataset and r["steps"] == steps]
            magnitudes = [r["weight_change_magnitude"] for r in group]
            ax.loglog(magnitudes, [r["mean_absolute_error"] for r in group],
                      "o-", color=color, label=f"{steps:,} updates")
            if steps == 1000:
                ax.loglog(magnitudes, [r["no_change_mean_absolute_error"] for r in group],
                          "--", color="#a66842", label="no-change baseline (1,000)")
        ax.set(xlabel="absolute example-weight change", ylabel="mean absolute prediction-change error",
               title="Approximation error across all examples and queries")
        ax.grid(True, which="major", alpha=0.2)
        ax.legend(fontsize=8)

    fig.supxlabel("Errors average both change signs and all three seeds. Heatmap scales differ by dataset.",
                  fontsize=10)
    fig.savefig(output / "influence.png", dpi=180)
    plt.close(fig)



def main(source="influence.csv", output="results"):
    rows = read_rows(source)
    summary = summarize(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    draw(rows, summary, output)


if __name__ == "__main__":
    main()
