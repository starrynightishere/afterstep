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


def prediction_gradients(params, x):
    (h0, h1), _ = forward(params, x)
    x0, x1 = x
    v0, v1 = params[6:8]
    d0 = v0 * (1.0 - h0**2)
    d1 = v1 * (1.0 - h1**2)
    return [
        d0 * x0, d0 * x1, d0,
        d1 * x0, d1 * x1, d1,
        h0, h1, 1.0,
    ]


def gradients(params, x, target):
    error = predict(params, x) - target
    return [error * g for g in prediction_gradients(params, x)]


def hessian_vector_product(params, x, target, vector):
    (h0, h1), prediction = forward(params, x)
    x0, x1 = x
    v0, v1 = params[6:8]
    dw00, dw01, db0, dw10, dw11, db1, dv0, dv1, db2 = vector

    slope0 = 1.0 - h0**2
    slope1 = 1.0 - h1**2
    dh0 = slope0 * (dw00 * x0 + dw01 * x1 + db0)
    dh1 = slope1 * (dw10 * x0 + dw11 * x1 + db1)
    d_prediction = dv0 * h0 + v0 * dh0 + dv1 * h1 + v1 * dh1 + db2

    d0 = dv0 * slope0 - 2.0 * v0 * h0 * dh0
    d1 = dv1 * slope1 - 2.0 * v1 * h1 * dh1
    gradient_change = [
        d0 * x0, d0 * x1, d0,
        d1 * x0, d1 * x1, d1,
        dh0, dh1, 0.0,
    ]

    error = prediction - target
    return [
        d_prediction * g + error * dg
        for g, dg in zip(prediction_gradients(params, x), gradient_change)
    ]
