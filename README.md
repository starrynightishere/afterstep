# afterstep

A tiny neural network that tracks how a training example affects a prediction.
It differentiates through gradient descent, estimates the effect of changing an
example's weight, and checks the estimate by retraining from the same initialization.

Built around a model with two hidden neurons in plain Python. Experiments cover
XOR and smooth regression, checking every training example against unseen query
points. The estimates describe small weight changes; larger changes can disagree
with retraining.

## run

Python 3. No packages to install.

```sh
git clone https://github.com/starrynightishere/afterstep.git
cd afterstep
python3 experiment.py
```

Writes `influence.csv` with 14,256 comparisons across two datasets, every training
example, nine unseen queries per dataset, three seeds, three training lengths,
and eleven weight changes. Each row records the settings, local sensitivity,
baseline prediction, predicted change, actual change, and absolute error.

Example weights scale the loss terms; the averaging denominator stays fixed.

Check the gradients and influence calculation against finite differences:

```sh
python3 -m unittest
```

## results

[Plot](results/influence.png) · [Summary](results/summary.csv)

Small weight changes track retraining closely. Large changes can miss badly,
especially when an example's weight is set to zero.

To regenerate the figure and summary after running the experiment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-plot.txt
.venv/bin/python plot.py
```

Matplotlib is only needed for plotting.
