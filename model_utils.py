import numpy as np

from activations import sigmoid, relu, relu_derivative

def initialize_parameters_deep(n_x, n_h):
    """
    Initializes parameters for a 2-layer neural network.

    Arguments:
      n_x -- number of input features
      n_h -- number of neurons in the hidden layer

    Returns:
      parameters -- dictionary containing:
          W1 -- weight matrix for hidden layer of shape (n_h, n_x)
          b1 -- bias vector for hidden layer of shape (n_h, 1)
          W2 -- weight matrix for output layer of shape (1, n_h)
          b2 -- bias for output layer (scalar)
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(1, n_h) * 0.01
    b2 = 0
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    """
    Implements forward propagation for a 2-layer neural network.

    Arguments:
        X -- input data of shape (n_x, m)
        parameters -- dictionary containing parameters (W1, b1, W2, b2)

    Returns:
        A2 -- output of the sigmoid activation function
        cache -- dictionary containing "Z1", "A1", "Z2", and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1   # (n_h, m)
    A1, cache1 = relu(Z1)     # (n_h, m)
    Z2 = np.dot(W2, A1) + b2  # (1, m)
    A2, cache2 = sigmoid(Z2)  # (1, m)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost given in equation (13).

    Arguments:
        A2 -- output of the sigmoid activation function
        Y -- true "label" vector of shape (1, number of examples)

    Returns:
        cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    cost = np.squeeze(cost)
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation.
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each cache from the dictionary "cache"
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    m = X.shape[1]

    # Backward propagation: calculate dW1, db1, dW2, db2
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1, cache)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate=0.01):
    """
    Updates parameters using gradient descent.

    Arguments:
      parameters -- dictionary containing current parameters
      grads -- dictionary containing gradients
      learning_rate -- learning rate

    Returns:
      parameters -- dictionary with updated parameters
    """
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters

def optimize(parameters, X, Y, num_iterations=2000, learning_rate=0.01, print_cost=False):
    """
    Runs gradient descent to optimize parameters.

    Arguments:
      parameters -- dictionary with initial parameters
      X -- input data of shape (n_x, m)
      Y -- true labels of shape (1, m)
      num_iterations -- number of iterations
      learning_rate -- learning rate for gradient descent
      print_cost -- if True, prints cost every 100 iterations

    Returns:
      parameters -- dictionary with optimized parameters
      grads -- gradients from last iteration
      costs -- list of costs recorded every 100 iterations
    """
    costs = []

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")
    return parameters, grads, costs

def predict(parameters, X):
    """
    Predicts binary labels for input data X using the trained parameters.

    Arguments:
      parameters -- dictionary containing parameters
      X -- input data of shape (n_x, m)

    Returns:
      predictions -- numpy array of shape (1, m) with predictions (0 or 1)
    """
    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 > 0.5).astype(int)
    return predictions

if __name__ == "__main__":
    # Test the network with random data
    np.random.seed(1)
    X = np.random.randn(3, 5)  # 3 features, 5 examples
    Y = (np.random.rand(1, 5) > 0.5).astype(int)
    parameters = initialize_parameters_deep(n_x=3, n_h=4)
    A2, cache = forward_propagation(X, parameters)
    cost = compute_cost(A2, Y)
    print("Initial cost:", cost)