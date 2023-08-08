#!/usr/bin/python3
#
# Copyright (C) 2019--2023 Richard Preen <rpreen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
This example demonstrates the XCSF supervised learning mechanisms to perform
regression on the kin8nm dataset. Classifiers are composed of tree GP
conditions and neural network predictions. A single dummy action is performed
such that [A] = [M].
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from libs import xcsf

np.set_printoptions(suppress=True)

RANDOM_STATE: int = 101

###########################################################
# Load data
###########################################################

# Loads the kin8nm dataset from: https://www.openml.org/d/189
data = fetch_openml(data_id=189, as_frame=True, parser="auto")

# numpy
X = np.asarray(data.data, dtype=np.float64)
y = np.asarray(data.target, dtype=np.float64)

# normalise inputs (zero mean and unit variance)
feature_scaler = StandardScaler()
X = feature_scaler.fit_transform(X)

# reshape outputs into 2D arrays
if y.ndim == 1:
    y = y.reshape(-1, 1)

# normalise outputs (zero mean and unit variance)
output_scaler = StandardScaler()
y = output_scaler.fit_transform(y)

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE
)

# 10% of training for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
)

print(f"X_train shape = {np.shape(X_train)}")
print(f"y_train shape = {np.shape(y_train)}")
print(f"X_val shape = {np.shape(X_val)}")
print(f"y_val shape = {np.shape(y_val)}")
print(f"X_test shape = {np.shape(X_test)}")
print(f"y_test shape = {np.shape(y_test)}")

X_DIM: int = np.shape(X_train)[1]
Y_DIM: int = np.shape(y_train)[1]

###########################################################
# Initialise XCSF
###########################################################

PERF_TRIALS: int = 5000
MAX_TRIALS: int = 200000
E0: float = 0.1

xcs = xcsf.XCS(
    x_dim=X_DIM,
    y_dim=Y_DIM,
    n_actions=1,
    omp_num_threads=12,
    random_state=RANDOM_STATE,
    pop_init=True,
    max_trials=MAX_TRIALS,
    perf_trials=PERF_TRIALS,
    pop_size=500,
    loss_func="mse",
    set_subsumption=False,
    theta_sub=100,
    e0=E0,
    alpha=1,
    nu=20,
    beta=0.1,
    delta=0.1,
    theta_del=50,
    init_fitness=0.01,
    init_error=0,
    m_probation=10000,
    stateful=True,
    compaction=False,
    ea={
        "select_type": "roulette",
        "theta_ea": 50,
        "lambda": 2,
        "p_crossover": 0.8,
        "err_reduc": 1,
        "fit_reduc": 0.1,
        "subsumption": False,
        "pred_reset": False,
    },
    condition={
        "type": "hyperrectangle_ubr",
        "args": {
            "min": 0.0,
            "max": 1.0,
            "spread_min": 1.0,
        },
    },
    prediction={
        "type": "nlms_linear",
        "args": {
            "eta": 1.0,
            "eta_min": 0.0001,
            "evolve_eta": True,
        },
    },
)

print(json.dumps(xcs.internal_params(), indent=4))

###########################################################
# Fit XCSF
###########################################################

callback = xcsf.EarlyStoppingCallback(
    # note: PERF_TRIALS is considered an "epoch" for callbacks
    monitor="val",  # which metric to monitor: {"train", "val"}
    patience=20000,  # trials with no improvement after which training will be stopped
    restore_best=True,  # whether to make checkpoints and restore best population
    min_delta=0,  # minimum change to qualify as an improvement
    start_from=0,  # trials to wait before starting to monitor improvement
    verbose=True,  # whether to display when checkpoints are made
)

xcs.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[callback])

###########################################################
# Plot XCSF learning performance
###########################################################

metrics: dict = xcs.get_metrics()
plt.figure(figsize=(10, 6))
plt.plot(metrics["trials"], metrics["train"], label="Train MSE")
plt.plot(metrics["trials"], metrics["val"], label="Validation MSE")
plt.grid(linestyle="dotted", linewidth=1)
plt.axhline(y=E0, xmin=0, xmax=1, linestyle="dashed", color="k")
plt.title("XCSF Training Performance", fontsize=14)
plt.xlabel("Trials", fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.ylim([0, 0.5])
plt.xlim([PERF_TRIALS, MAX_TRIALS])
plt.legend()
plt.show()

###########################################################
# Final XCSF test score
###########################################################

print("*****************************")
xcsf_pred = xcs.predict(X_test)
xcsf_mse = mean_squared_error(xcsf_pred, y_test)
print(f"XCSF Test MSE = {xcsf_mse:.4f}")

###########################################################
# Compare with alternatives
###########################################################

X_train = np.vstack((X_train, X_val))
y_train = np.vstack((y_train, y_val))

regressors = []

regressors.append(LinearRegression())

regressors.append(
    MLPRegressor(
        random_state=RANDOM_STATE,
        hidden_layer_sizes=(40,),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=0.01,
        max_iter=1000,
        alpha=0.01,
        validation_fraction=0.1,
    )
)

regressors.append(RandomForestRegressor(random_state=RANDOM_STATE))

regressors.append(GaussianProcessRegressor(random_state=RANDOM_STATE))

for model in regressors:
    model.fit(X_train, y_train.ravel())
    pred = model.predict(X_test)
    mse = mean_squared_error(pred, y_test)
    print(f"{model.__class__.__name__} Test MSE = {mse:.4f}")
