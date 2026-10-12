from neuron import gradients


def main():
    x = 2.0
    target = 1.0
    weight = 0.5
    bias = 0.5
    learning_rate = 0.1

    for _ in range(10):
        d_weight, d_bias = gradients(weight, bias, x, target)
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

    return weight, bias


if __name__ == "__main__":
    main()
