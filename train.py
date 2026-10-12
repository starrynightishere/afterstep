from neuron import gradients, loss, predict

def main():
    x = 2.0
    target = 1.0
    weight = 0.5
    bias = 0.5
    learning_rate = 0.1

    print("step  prediction  loss")
    print(f"{0:4d}  {predict(weight, bias, x):10.6f}  "
          f"{loss(weight, bias, x, target):.8f}")

    for step in range(1, 11):
        d_weight, d_bias = gradients(weight, bias, x, target)
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

        print(f"{step:4d}  {predict(weight, bias, x):10.6f}  "
              f"{loss(weight, bias, x, target):.8f}")

if __name__ == "__main__":
    main()
