# afterstep

A tiny neural network that tracks how a training example affects a prediction.
It differentiates through gradient descent, estimates the effect of changing an
example's weight, and checks the estimate by retraining from the same initialization.

Built around a two-neuron XOR model in plain Python. The estimates describe small
weight changes; larger changes can disagree with retraining.

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
