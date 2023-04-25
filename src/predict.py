import numpy as np
import pickle


def main():
    regressor = pickle.load(open("target/models/Miles_per_Gallon_Regressor.pickle", "rb"))

    y_pred = regressor.predict(np.asarray([6.0, 200.0, 2500.0, 9.5, 99.0]).reshape(-1, 5))

    print(y_pred)


if __name__ == "__main__":
    main()
