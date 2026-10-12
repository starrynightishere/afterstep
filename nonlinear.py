from math import tanh


def predict(weight, bias, x):
    return tanh(weight * x + bias)


def loss(weight, bias, x, target):
    error = predict(weight, bias, x) - target
    return 0.5 * error**2


def gradients(weight, bias, x, target):
    prediction = predict(weight, bias, x)
    error = prediction - target
    slope = error * (1.0 - prediction**2)
    return slope * x, slope
