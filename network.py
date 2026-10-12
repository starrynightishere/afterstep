from math import tanh


def forward(params, x):
    w00, w01, b0, w10, w11, b1, v0, v1, b2 = params
    x0, x1 = x
    h0 = tanh(w00 * x0 + w01 * x1 + b0)
    h1 = tanh(w10 * x0 + w11 * x1 + b1)
    prediction = v0 * h0 + v1 * h1 + b2
    return (h0, h1), prediction


def predict(params, x):
    return forward(params, x)[1]


def loss(params, x, target):
    error = predict(params, x) - target
    return 0.5 * error**2


def gradients(params, x, target):
    (h0, h1), prediction = forward(params, x)
    x0, x1 = x
    v0, v1 = params[6:8]
    error = prediction - target
    d0 = error * v0 * (1.0 - h0**2)
    d1 = error * v1 * (1.0 - h1**2)
    return [
        d0 * x0, d0 * x1, d0,
        d1 * x0, d1 * x1, d1,
        error * h0, error * h1, error,
    ]
