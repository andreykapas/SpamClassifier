# data_utils.py

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_spambase(data_file="spambase.data"):
    """
    Loads the Spambase dataset from a CSV file.
    Returns:
      X -- numpy array of features, shape (m, n_x)
      y -- numpy array of labels, shape (m,)
    """
    # Spambase doesn't have headers, so we read it as CSV with no header.
    # The last column is the label (1 = spam, 0 = not spam).
    df = pd.read_csv(data_file, header=None)
    X = df.iloc[:, :-1].values  # all columns except the last
    y = df.iloc[:, -1].values  # the last column
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into train/test sets and standardizes features.

    Returns:
      X_train -- shape (n_x, m_train)
      X_test -- shape (n_x, m_test)
      y_train -- shape (1, m_train)
      y_test -- shape (1, m_test)
    """
    # X has shape (m, n_x), y has shape (m,)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Transpose X to get (n_x, m)
    X_train = X_train.T
    X_test = X_test.T

    # Reshape y to (1, m)
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    # Standardize the data: the scaler expects shape (m, n_x), so transpose, fit_transform, transpose back
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T

    return X_train, X_test, y_train, y_test


def load_and_preprocess_data(
    cache_file="spam_data.pkl", data_file="spambase.data"
):
    """
    Loads and preprocesses the Spambase data with caching.
    """
    if os.path.exists(cache_file):
        print("Loading data from cache...")
        with open(cache_file, "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
    else:
        print("Cache not found, loading data from file and preprocessing...")
        X, y = load_data_spambase(data_file)
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        with open(cache_file, "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"Data saved to cache ('{cache_file}').")
    return X_train, X_test, y_train, y_test


def print_data_shapes(X_train, y_train, X_test, y_test):
    """
    Prints shapes and info about the dataset.
    """
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Number of training examples:", X_train.shape[1])
    print("Number of test examples:", X_test.shape[1])
    print("Number of features:", X_train.shape[0])
    print("Number of classes:", len(np.unique(y_train)))
    print("Classes:", np.unique(y_train))
    print()


if __name__ == "__main__":
    # Simple test
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print_data_shapes(X_train, y_train, X_test, y_test)
