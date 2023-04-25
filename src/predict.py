import numpy as np
import pickle


def predict(regressor, n_features, X_np):
    y_pred_np = regressor.predict(X_np.reshape(-1, n_features))
    return y_pred_np


def main():
    regressor = pickle.load(open("target/models/Miles_per_Gallon_Regressor.pickle", "rb"))

    X_np = np.asarray([6.0, 200.0, 2500.0, 9.5, 99.0])

    y_pred_np = regressor.predict(X_np.reshape(-1, 5))

    print(y_pred_np)


if __name__ == "__main__":
    main()
