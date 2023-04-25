# to create directory
import os

# to read CSV and for use with the library ppscore
import pandas as pd
from pandas.api.types import CategoricalDtype

# for use with sklearn and for EDA
import numpy as np

# Data Split
from sklearn.model_selection import train_test_split

# Normalization and Standardization
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# from sklearn.preprocessing import RobustScaler

# Regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor

# Plots
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm

# EDA
import ppscore as pps
from scipy import stats

# PCA
from sklearn.decomposition import PCA

# Tensors and Artificial Neural Networks
import torch
import torch.nn as nn

#  Model Dump
import pickle


def load_dataset_with_regression_target(
    path: str,
    sep: str,
    target_column_name: str,
    columns_to_drop: list,
    column_name_mapping: dict,
):
    #    target_column_is_index = False
    #    target_column_index = -1
    #    if target_column_name.isnumeric():
    #        target_column_is_index = True
    #        target_column_index = int(target_column_name)

    # read the CSV file
    # using separator character semicolon
    X_original_y_pd = pd.read_csv(path, sep=sep, skipinitialspace=True).astype(float)

    # make column names pythonic
    # so that they can be used in code where applicable
    X_original_y_pd.columns = X_original_y_pd.columns.str.replace(" ", "_")

    for column_to_drop in columns_to_drop:
        X_original_y_pd.drop(column_to_drop, axis=1, inplace=True)

    X_original_y_pd.rename(column_name_mapping, axis=1, inplace=True)

    print(f"X_original_y_pd=\n{X_original_y_pd}")

    X_original_y_np = X_original_y_pd.to_numpy()

    # number of instances often referred to as just n
    n_samples = X_original_y_np.shape[0]
    print(f"n_samples={n_samples}")

    # number of target variables
    n_targets = 1
    print(f"n_targets={n_targets}")

    # number of features
    n_features = X_original_y_np.shape[1] - n_targets
    print(f"n_features={n_features}")

    assert X_original_y_np.shape == (n_samples, n_features + n_targets)

    X_original_pd = X_original_y_pd.copy().drop(target_column_name, axis=1)
    X_original_np = X_original_pd.to_numpy()
    assert X_original_np.shape == (n_samples, n_features)

    y_original_pd = X_original_y_pd[target_column_name].copy()
    y_original_np = y_original_pd.to_numpy()
    assert y_original_np.shape == (n_samples,)

    X_original = torch.from_numpy(X_original_np)
    y_original = torch.from_numpy(y_original_np)

    print(X_original.dtype)
    print(y_original.dtype)

    assert X_original.dtype == torch.float64
    assert y_original.dtype == torch.float64

    # also create a tensor that contains the 4 features and the target
    y_original_unsqueezed = y_original.unsqueeze(1)
    X_original_y = torch.cat((X_original, y_original_unsqueezed), 1)
    assert X_original_y.shape == (n_samples, n_features + n_targets)
    assert X_original_y.dtype == torch.float64

    return (
        X_original_pd,
        X_original_np,
        X_original,
        y_original_pd,
        y_original_np,
        y_original,
        X_original_y_pd,
        X_original_y_np,
        X_original_y,
    )


def standardize_dataset(X_original_np, y_original_np, target_column_name, column_name_mapping):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_np = scaler_X.fit_transform(X_original_np)
    y_original_tmp_np = y_original_np.reshape(-1, 1)
    y_tmp_np = scaler_y.fit_transform(y_original_tmp_np)
    y_np = y_tmp_np.reshape(len(y_tmp_np))

    X_column_names = list(column_name_mapping.values())
    X_column_names.remove(target_column_name)
    # print(X_column_names)

    X_pd = pd.DataFrame(X_np, columns=X_column_names)
    y_pd = pd.DataFrame(y_np, columns=[target_column_name])

    # print(f"X_pd=\n{X_pd}")
    # print(f"y_pd=\n{y_pd}")

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    return (X_pd, X_np, X, scaler_X, y_pd, y_np, y, scaler_y)


def split_dataset_with_regression_target(X, y, n_targets):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # we first attempted to do this purely in PyTorch which is still a bit difficult
    # dataset_known = torch.from_numpy(dataset_known_np)
    # dataset_known_subsets = torch.utils.data.random_split(dataset_known, [int(n_samples * 0.7), int(n_samples * 0.3)])
    # dataset_known_train_subset = dataset_known_subsets[0]
    # dataset_known_test_subset = dataset_known_subsets[1]
    # assert len(dataset_known_train_subset) == 105
    # assert len(dataset_known_test_subset) == 45

    # however many people still use pandas and sklearn which we follow for now
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, random_state=77)

    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.20, random_state=77)

    del X_tmp
    del y_tmp

    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == n_samples
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == n_samples

    assert X_train.ndim == 2
    assert X_train.shape[1] == n_features
    assert X_train.dtype == torch.float64

    assert X_val.ndim == 2
    assert X_val.shape[1] == n_features
    assert X_val.dtype == torch.float64

    assert X_test.ndim == 2
    assert X_test.shape[1] == n_features
    assert X_test.dtype == torch.float64

    assert (n_targets == 1 and y_train.ndim == 1) or (n_targets > 1 and y_train.ndim == 2)
    assert y_train.dtype == torch.float64

    assert (n_targets == 1 and y_val.ndim == 1) or (n_targets > 1 and y_val.ndim == 2)
    assert y_val.dtype == torch.float64

    assert (n_targets == 1 and y_test.ndim == 1) or (n_targets > 1 and y_test.ndim == 2)
    assert y_test.dtype == torch.float64

    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_val_np = X_val.numpy()
    y_val_np = y_val.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    return (
        X_train_np,
        X_train,
        y_train_np,
        y_train,
        X_val_np,
        X_val,
        y_val_np,
        y_val,
        X_test_np,
        X_test,
        y_test_np,
        y_test,
    )


def explore_dataset_with_regression_target(X_y_pd, X_y_np, target_column_name):
    # number of instances
    print(f"n={X_y_np.shape[0]}")

    # location parameters
    print(f"mean={X_y_np.mean(axis=0)}")
    print(f"trimmed_mean={stats.trim_mean(X_y_np.astype('float32'), proportiontocut=0.10, axis=0)}")
    print(f"mode={stats.mode(X_y_np, keepdims=True)}")

    # statistical dispersion measures
    def range_np(a: np.ndarray) -> np.ndarray:
        result = a.max(axis=0) - a.min(axis=0)
        return result

    print(f"range={range_np(X_y_np)}")
    print(f"iqr={stats.iqr(X_y_np, axis=0)}")

    print(f"percentile_10={np.percentile(X_y_np, 10.0, axis=0)}")
    print(f"percentile_25={np.percentile(X_y_np, 25.0, axis=0)}")
    print(f"median={np.percentile(X_y_np, 50.0, axis=0)}")
    print(f"percentile_75={np.percentile(X_y_np, 75.0, axis=0)}")
    print(f"percentile_90={np.percentile(X_y_np, 90.0, axis=0)}")

    def mad_np(a: np.ndarray) -> np.ndarray:
        result = np.mean(np.absolute(a - np.mean(a, axis=0)), axis=0)
        return result

    print(f"mad={mad_np(X_y_np)}")

    print(f"std={X_y_np.std(axis=0)}")
    print(f"var={X_y_np.var(axis=0)}")

    # association measures
    print(f"\ncorrelation_matrix=\n{np.corrcoef(X_y_np, rowvar=False).round(decimals=2)}")

    # we have a look at a scatter matrix
    pd.plotting.scatter_matrix(
        X_y_pd,
        c=X_y_pd[target_column_name],
        figsize=(17, 17),
        cmap=cm["cool"],
        diagonal="kde",
    )

    # predictive_power_score_matrix_all_pd = pps.matrix(df_pd_all, output='df')
    predictive_power_scores_pd = pps.predictors(X_y_pd, y=target_column_name, output="df")
    predictive_power_scores_pd.style.background_gradient(cmap="twilight", low=0.0, high=1.0)
    print(predictive_power_scores_pd)


def train_regressor(X_train_np, y_train_np):
    reg1 = GradientBoostingRegressor(random_state=42)
    reg2 = RandomForestRegressor(random_state=42)
    reg3 = LinearRegression()
    reg4 = DecisionTreeRegressor(max_depth=16)
    ereg = VotingRegressor(estimators=[("gb", reg1), ("rf", reg2), ("lr", reg3), ("dt", reg4)])
    ereg = ereg.fit(X_train_np, y_train_np)

    return ereg


def assert_elements_are_zero_or_one(a):
    assert len(np.ma.masked_where(((a == 0) | (a == 1)).all(), a).compressed()) == 0


def sse(y, y_hat):
    return np.square(y - y_hat).sum()


def mse(y, y_pred):
    return sse(y, y_pred) / len(y)


def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))


def mae(y, y_pred):
    return np.abs(y - y_pred).sum() / len(y)


def main():
    n_targets = 1
    target_column_name = "mpg"
    columns_to_drop = []
    column_name_mapping = {
        "mpg": "mpg",
        "zylinder;": "zylinder",
        "ps": "ps",
        "gewicht": "gewicht",
        "beschleunigung": "beschleunigung",
        "baujahr": "baujahr",
    }

    (
        X_original_pd,
        X_original_np,
        X_original,
        y_original_pd,
        y_original_np,
        y_original,
        X_original_y_pd,
        X_original_y_np,
        X_original_y,
    ) = load_dataset_with_regression_target(
        path="data/auto-mpg.csv",
        sep=";",
        target_column_name=target_column_name,
        columns_to_drop=columns_to_drop,
        column_name_mapping=column_name_mapping,
    )

    X_pd, X_np, X, scaler_X, y_pd, y_np, y, scaler_y = standardize_dataset(
        X_original_np, y_original_np, target_column_name, column_name_mapping
    )

    (
        X_train_np,
        X_train,
        y_train_np,
        y_train,
        X_val_np,
        X_val,
        y_val_np,
        y_val,
        X_test_np,
        X_test,
        y_test_np,
        y_test,
    ) = split_dataset_with_regression_target(X, y, n_targets)

    # explore_dataset_with_regression_target(X_y_pd, X_y_np, target_column_name)

    regressor = train_regressor(X_train_np, y_train_np)

    y_pred_test_np = regressor.predict(X_test_np)

    # print(y_test_np)
    # print(y_pred_test_np)

    current_rmse = rmse(y_test_np, y_pred_test_np)
    print(f"current_rmse={current_rmse}")

    current_mae = mae(y_test_np, y_pred_test_np)
    print(f"current_mae={current_mae}")

    os.makedirs("target/models", exist_ok=True)

    file_to_write = open("target/models/Miles_per_Gallon_Regressor.pickle", "wb")
    pickle.dump(regressor, file_to_write)


if __name__ == "__main__":
    main()
