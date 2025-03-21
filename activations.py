import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_derivative(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def test_activation_functions():
    """
    Test the activation functions and print their outputs.
    """
    Z_test = np.array([-5, -1, 0, 1, 5])
    A_sigmoid, _ = sigmoid(Z_test)
    print("Sigmoid(", Z_test, ") =", A_sigmoid)

    A_relu, _ = relu(Z_test)
    print("ReLU(", Z_test, ") =", A_relu)


if __name__ == "__main__":
    test_activation_functions()
