def predict(weight, bias, x):
    return weight * x + bias


def loss(weight, bias, x, target):
    error = predict(weight, bias, x) - target
    return 0.5 * error**2


def gradients(weight, bias, x, target):
    error = predict(weight, bias, x) - target
    d_weight = error * x
    d_bias = error
    return d_weight, d_bias


