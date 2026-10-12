from neuron import gradients, loss, predict


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

    print("step  weight    bias      mean loss")
    print(f"{0:4d}  {weight:.6f}  {bias:.6f}  {mean_loss(weight, bias, data):.3e}")

    for step in range(1, 101):
        d_weight, d_bias = mean_gradients(weight, bias, data)
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

        if step == 1 or step % 10 == 0:
            print(
                f"{step:4d}  {weight:.6f}  {bias:.6f}  "
                f"{mean_loss(weight, bias, data):.3e}"
            )

    print(f"\nAt x=3: prediction={predict(weight, bias, 3.0):.6f}, expected=7")


if __name__ == "__main__":
    main()
