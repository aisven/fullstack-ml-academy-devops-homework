import numpy as np
import pickle


def predict(regressor, n_features, X_np):
    y_pred_np = regressor.predict(X_np.reshape(-1, n_features))
    return y_pred_np


def main():
    regressor = pickle.load(open("target/models/Miles_per_Gallon_Regressor.pickle", "rb"))
    scaler_X = pickle.load(open("target/models/Miles_per_Gallon_Scaler_X.pickle", "rb"))
    scaler_y = pickle.load(open("target/models/Miles_per_Gallon_Scaler_y.pickle", "rb"))

    X_original_np = np.asarray([6.0, 200.0, 2500.0, 9.5, 99.0])
    X_np = scaler_X.transform(X_original_np.reshape(-1, 5))

    y_pred_np = regressor.predict(X_np.reshape(-1, 5))

    y_pred_original_np = scaler_y.inverse_transform(y_pred_np.reshape(-1, 1))

    print(y_pred_np)
    print(y_pred_original_np)


if __name__ == "__main__":
    main()
