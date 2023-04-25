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
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# Decision Tree Classifier
from sklearn import tree

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
    X_original_y_pd = pd.read_csv(path, sep=sep, skipinitialspace=True)

    # make column names pythonic
    # so that they can be used in code where applicable
    X_original_y_pd.columns = X_original_y_pd.columns.str.replace(" ", "_")

    for column_to_drop in columns_to_drop:
        X_original_y_pd.drop(column_to_drop, axis=1, inplace=True)

    X_original_y_pd.rename(column_name_mapping, axis=1, inplace=True)

    X_y_np = X_original_y_pd.to_numpy()

    # number of instances often referred to as just n
    n_samples = X_y_np.shape[0]
    print(f"n_samples={n_samples}")

    # number of target variables
    n_targets = 1
    print(f"n_targets={n_targets}")

    # number of features
    n_features = X_y_np.shape[1] - n_targets
    print(f"n_features={n_features}")

    assert X_y_np.shape == (n_samples, n_features + n_targets)
    assert X_y_np.shape == (n_samples, n_features + n_targets)

    X_original_pd = X_original_y_pd.copy().drop(target_column_name, axis=1)
    X_original_np = X_original_pd.to_numpy()
    assert X_original_np.shape == (n_samples, n_features)

    y_pd = X_original_y_pd[target_column_name].copy()
    y_np = y_pd.to_numpy()
    assert y_np.shape == (n_samples,)

    X_original = torch.from_numpy(X_original_np)
    y = torch.from_numpy(y_np)

    # we need the target data to be of data type long for the loss function to work
    if y.dtype != torch.int64:
        y = y.long()
    assert X_original.dtype == torch.float64
    assert y.dtype == torch.int64

    # also create a tensor that contains the 4 features and the target
    X_original_y_np = X_original_y_pd.to_numpy()
    y_unsqueezed = y.unsqueeze(1)
    X_original_y = torch.cat((X_original, y_unsqueezed), 1)
    assert X_original_y.shape == (n_samples, n_features + n_targets)
    assert X_original_y.dtype == torch.float64

    return (
        X_original_pd,
        X_original_np,
        X_original,
        y_pd,
        y_np,
        y,
        X_original_y_pd,
        X_original_y_np,
        X_original_y,
    )


def standardize_dataset(X_original_np):
    scaler = RobustScaler(unit_variance=True)
    X_np = scaler.fit_transform(X_original_np)

    return (scaler, X_np)


def split_dataset(X, y, n_targets):
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
    assert y_train.dtype == torch.int64

    assert (n_targets == 1 and y_val.ndim == 1) or (n_targets > 1 and y_val.ndim == 2)
    assert y_val.dtype == torch.int64

    assert (n_targets == 1 and y_test.ndim == 1) or (n_targets > 1 and y_test.ndim == 2)
    assert y_test.dtype == torch.int64

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


def train_decision_tree_regressor(X_train_np, y_train_np, max_depth):
    dtr = tree.DecisionTreeRegressor(random_state=42, max_depth=max_depth)
    dtr.fit(X_train_np, y_train_np)
    return dtr


def assert_elements_are_zero_or_one(a):
    assert len(np.ma.masked_where(((a == 0) | (a == 1)).all(), a).compressed()) == 0


def regression_accuracy_np(y_reference_np, y_pred_np):
    assert y_reference_np.shape == y_pred_np.shape
    y_reference_rounded_np = np.round(y_reference_np.astype(float), 2)
    y_pred_rounded_np = np.round(y_pred_np.astype(float), 2)
    # the following tensor will contain True for equal values and False for different values
    comparison = y_reference_rounded_np == y_pred_rounded_np
    # the following tensor will contain 1.0 for true positives and 0.0 for true negatives
    comparison_float = comparison.astype(float)
    # the mean of that tensor will thus represent the percentage of true positives, e.g. 97.5
    comparison_mean = comparison_float.mean()
    # we scale and round it to obtain the value in percent
    comparison_percent = np.round(comparison_mean, decimals=2) * 100
    # print(f"comparison={comparison}")
    # print(f"comparison_float={comparison_float}")
    # print(f"comparison_mean={comparison_mean}")
    # print(f"comparison_percent={comparison_percent}")
    assert 0.00 <= comparison_percent and comparison_percent <= 100.00
    return comparison_percent


def main():
    n_targets = 1

    (
        X_original_pd,
        X_original_np,
        X_original,
        y_pd,
        y_np,
        y,
        X_original_y_pd,
        X_original_y_np,
        X_original_y,
    ) = load_dataset_with_regression_target(
        path="data/auto-mpg.csv",
        sep=";",
        target_column_name="mpg",
        columns_to_drop=[],
        column_name_mapping={},
    )

    # scaler, X_np = standardize_dataset(X_original_np)

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
    ) = split_dataset(X_original, y, n_targets)

    dtr = train_decision_tree_regressor(X_train_np, y_train_np, 10)

    y_pred_test_np = dtr.predict(X_test_np)

    acc_test = regression_accuracy_np(y_test_np, y_pred_test_np)

    print(f"acc_test={acc_test}")

    os.mkdir("target")
    os.mkdir("target/models")

    file_to_write = open("target/models/Miles_per_Gallon_DecisionTreeRegressor.pickle", "wb")
    pickle.dump(dtr, file_to_write)


if __name__ == "__main__":
    main()
