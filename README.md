# afterstep

A learning project about how training examples change a model's predictions.

Current stage: one linear neuron and one gradient-descent update. Training-data
influence, multiple neurons, and differentiation through training are future work.

## Lesson 1: one prediction, one loss, one update

We have one observation: input `x = 2`, desired output `target = 1`.
Our model has two adjustable numbers: a weight `w` and a bias `b`.

```text
prediction = w*x + b
error      = prediction - target
loss       = error² / 2
```

The weight controls the slope; the bias shifts the prediction. This first model
is linear, with no activation function. The loss measures prediction error.
The factor of one half makes its derivative simpler.

Start with `w = 0.5` and `b = 0.5`. The prediction is `1.5`, the error is `0.5`,
and the loss is `0.125`.

The chain rule tells us how each parameter changes the loss:

```text
dL/dprediction = prediction - target = error
dprediction/dw = x
dprediction/db = 1

dL/dw = error*x = 1.0
dL/db = error   = 0.5
```

These derivatives describe local changes, holding the other parameter fixed.
Gradient descent moves opposite the gradient:

```text
new_w = w - learning_rate * dL/dw
new_b = b - learning_rate * dL/db
```

At a learning rate of `0.1`, the new parameters are `0.4` and `0.45`.
The new prediction is `1.25`, and the loss is `0.03125`.
The update improves this example. A larger learning rate is not guaranteed to help.
Fitting one observation does not demonstrate generalization.

## Run

From this folder, using Python 3 (no packages to install):

```sh
python3 neuron.py
python3 -m unittest -v
```

The test estimates each derivative by slightly increasing and decreasing one
parameter and measuring the loss change. It compares this independent estimate
with the handwritten derivative, including negative and zero input values.

## Before moving on

1. Explain the difference between a prediction, a loss, and a gradient.
2. Explain why `x` appears in the weight derivative but not the bias derivative.
3. Predict the weight derivative if `x = 0` before running that experiment.

Related inspiration: [micrograd](https://github.com/karpathy/micrograd).
This initial exercise was developed with AI assistance as part of a guided
learning session. Its mathematics is standard; no novelty is claimed.
