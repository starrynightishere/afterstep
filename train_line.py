from neuron import gradients, loss


def mean_loss(weight, bias, data):
    total = 0.0
    for x, target in data:
        total += loss(weight, bias, x, target)
    return total / len(data)


def mean_gradients(weight, bias, data):
    total_weight = 0.0
    total_bias = 0.0
    for x, target in data:
        d_weight, d_bias = gradients(weight, bias, x, target)
        total_weight += d_weight
        total_bias += d_bias
    return total_weight / len(data), total_bias / len(data)


def main():
    data = [(-2.0, -3.0), (-1.0, -1.0), (0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]
    weight = 0.0
    bias = 0.0
    learning_rate = 0.1

    for _ in range(100):
        d_weight, d_bias = mean_gradients(weight, bias, data)
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

    return weight, bias


if __name__ == "__main__":
    main()
