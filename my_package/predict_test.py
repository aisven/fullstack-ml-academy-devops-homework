import pickle
import numpy as np

from train import train
from predict import predict


def test_train_then_predict():
    train()

    regressor = pickle.load(open("target/models/Miles_per_Gallon_Regressor.pickle", "rb"))

    X_np = np.asarray([5.0, 180.0, 2400.0, 8.7, 99.0])

    y_pred_np = predict(regressor, 5, X_np)

    assert y_pred_np.shape == (1,)
