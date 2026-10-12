# afterstep

A tiny neural network that tracks how a training example affects a prediction.
It differentiates through gradient descent, estimates the effect of changing an
example's weight, and checks the estimate by retraining from the same initialization.

Built around an XOR model with two hidden neurons in plain Python. The estimates
describe small weight changes; larger changes can disagree with retraining.

## run

Python 3. No packages to install.

```sh
git clone https://github.com/starrynightishere/afterstep.git
cd afterstep
python3 experiment.py
```

Writes `influence.csv` with 99 comparisons across three seeds, three training
lengths, and eleven weight changes. Each row includes the predicted change,
actual change, and absolute error.

The default experiment changes one example and queries that same training input.
Example weights scale the loss terms; the averaging denominator stays fixed.

Check the gradients and influence calculation against finite differences:

```sh
python3 -m unittest
```
