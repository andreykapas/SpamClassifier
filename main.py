# main.py
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from data_utils import load_and_preprocess_data, print_data_shapes
from model_utils import initialize_parameters_deep, optimize, predict

def load_model_parameters(cache_file="model_params_mlp.pkl"):
    """
    Loads trained model parameters if they exist.
    Returns:
      parameters -- dictionary with parameters, or None if not found.
    """
    try:
        with open(cache_file, "rb") as f:
            parameters = pickle.load(f)
        print("Loaded model parameters from cache.")
        return parameters
    except FileNotFoundError:
        return None

def main():
    # Step 1: Load and preprocess the data (with caching)
    start_time = time.time()
    # Используем, например, Spambase датасет, кешируем в "spam_data.pkl"
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data("spam_data.pkl", data_file="spambase.data")
    end_time = time.time()
    print(f"Data loaded and preprocessed in {end_time - start_time:.2f} seconds.")

    print_data_shapes(X_train, Y_train, X_test, Y_test)
    # Expected X_train shape: (n_x, m_train) e.g. (57, 3680)

    n_x = X_train.shape[0]  # число признаков
    # Set hidden layer size, например, 10 нейронов
    n_h = 10

    # Step 2: Try to load trained model parameters
    model_file = "model_params_mlp.pkl"
    parameters = load_model_parameters(model_file)

    # Step 3: If not loaded, initialize and train the model
    if parameters is None:
        print("No trained model found, training new model...")
        parameters = initialize_parameters_deep(n_x, n_h)
        parameters, grads, costs = optimize(parameters, X_train, Y_train,
                                            num_iterations=2000,
                                            learning_rate=0.05,
                                            print_cost=True)
        # Save the trained parameters
        with open(model_file, "wb") as f:
            pickle.dump(parameters, f)
        print("Model parameters saved to", model_file)

        # Plot cost curve
        plt.plot(costs)
        plt.xlabel("Iterations (per hundreds)")
        plt.ylabel("Cost")
        plt.title("Cost Reduction Over Iterations")
        plt.grid(True)
        plt.show()
    else:
        print("Using loaded model parameters.")

    # Step 4: Evaluate the model on training and test sets
    train_predictions = predict(parameters, X_train)
    test_predictions = predict(parameters, X_test)

    train_accuracy = 100 - np.mean(np.abs(train_predictions - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(test_predictions - Y_test)) * 100

    print("Train Accuracy: {:.2f}%".format(train_accuracy))
    print("Test Accuracy: {:.2f}%".format(test_accuracy))

if __name__ == "__main__":
    main()
