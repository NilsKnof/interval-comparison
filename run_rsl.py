from libs import xcsf
import click
import os
import numpy as np
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

def get_train(data):
    X, y = data["X"], data["y"]
    y = y.reshape(len(X), -1)
    return X, y


def get_test(data):
    X_test = data["X_test"]
    try:
        y_test = data["y_test_true"]
    except KeyError:
        y_test = data["y_test"]
    y_test = y_test.reshape(len(X_test), -1)
    return X_test, y_test


@click.group()
def cli():
    pass


@click.command()
@click.argument('NPZFILE')
def run(npzfile):
    # training data
    data = np.load(npzfile)
    X, y = get_train(data)
    N, DX = X.shape

    # test data
    X_test, y_test = get_test(data)

    # ground truth
    centers_true = data["centers"]
    spreads_true = data["spreads"]
    lowers_true = centers_true - spreads_true
    uppers_true = centers_true + spreads_true

    K = len(centers_true)

    def eval_model(model, label):
        pipe = make_pipeline(
            MinMaxScaler(feature_range=(-1.0, 1.0)),
            TransformedTargetRegressor(regressor=model,
                                       transformer=StandardScaler()))
        pipe.fit(X, y)
        print("Performing predictions on test data")
        y_test_pred = pipe.predict(X_test)
        print("Performing predictions on training data")
        y_pred = pipe.predict(X)
        mse_test = mean_squared_error(y_test_pred, y_test)
        print(f"MSE test ({label}):", mse_test)
        mse_train = mean_squared_error(y_pred, y)
        print(f"MSE train ({label}):", mse_train)
        return y_pred, y_test_pred

    model_ubr = xcsf.XCS(
        x_dim=DX,
        y_dim=1,
        n_actions=1,
        alpha=0.1,
        beta=0.2,
        delta=0.1,
        e0=0.01,
        init_error=0,
        init_fitness=0.01,
        m_probation=10000,
        nu=5,
        omp_num_threads=12,
        perf_trials=1000,
        max_trials=100000,
        pop_init=True,
        pop_size=200,
        random_state=1234,
        set_subsumption=True,
        theta_del=20,
        theta_sub=100,
        loss_func="mse",
        ea={
            "select_type": "tournament",
            "select_size": 0.4,
            "theta_ea": 25,
            "lambda": 2,
            "p_crossover": 0.8,
            "err_reduc": 0.25,
            "fit_reduc": 0.1,
            "subsumption": True,
            "pred_reset": False,
        },
        action={
            "type": "integer",
        },
        condition={
            "type": "hyperrectangle_ubr",
            "args": {
                "min": -1.0,
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

    model_csr = xcsf.XCS(
        x_dim=DX,
        y_dim=1,
        n_actions=1,
        alpha=0.1,
        beta=0.2,
        delta=0.1,
        e0=0.01,
        init_error=0,
        init_fitness=0.01,
        m_probation=10000,
        nu=5,
        omp_num_threads=12,
        perf_trials=1000,
        max_trials=100000,
        pop_init=False,
        pop_size=200,
        random_state=23423543,
        set_subsumption=True,
        theta_del=20,
        theta_sub=100,
        loss_func="mse",
        ea={
            "select_type": "tournament",
            "select_size": 0.4,
            "theta_ea": 25,
            "lambda": 2,
            "p_crossover": 0.8,
            "err_reduc": 0.25,
            "fit_reduc": 0.1,
            "subsumption": True,
            "pred_reset": False,
        },
        action={
            "type": "integer",
        },
        condition={
            "type": "hyperrectangle_csr",
            "args": {
                "min": -1.0,
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

    model_mmr = xcsf.XCS(
        x_dim=DX,
        y_dim=1,
        n_actions=1,
        alpha=0.1,
        beta=0.2,
        delta=0.1,
        e0=0.01,
        init_error=0,
        init_fitness=0.01,
        m_probation=10000,
        nu=5,
        omp_num_threads=12,
        perf_trials=1000,
        max_trials=100000,
        pop_init=False,
        pop_size=200,
        random_state=23423543,
        set_subsumption=True,
        theta_del=20,
        theta_sub=100,
        loss_func="mse",
        ea={
            "select_type": "tournament",
            "select_size": 0.4,
            "theta_ea": 25,
            "lambda": 2,
            "p_crossover": 0.8,
            "err_reduc": 0.25,
            "fit_reduc": 0.1,
            "subsumption": True,
            "pred_reset": False,
        },
        action={
            "type": "integer",
        },
        condition={
            "type": "hyperrectangle_mmr",
            "args": {
                "min": -1.0,
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
    model_mpr = xcsf.XCS(
        x_dim=DX,
        y_dim=1,
        n_actions=1,
        alpha=0.1,
        beta=0.2,
        delta=0.1,
        e0=0.01,
        init_error=0,
        init_fitness=0.01,
        m_probation=10000,
        nu=5,
        omp_num_threads=12,
        perf_trials=1000,
        max_trials=100000,
        pop_init=False,
        pop_size=200,
        random_state=23423543,
        set_subsumption=True,
        theta_del=20,
        theta_sub=100,
        loss_func="mse",
        ea={
            "select_type": "tournament",
            "select_size": 0.4,
            "theta_ea": 25,
            "lambda": 2,
            "p_crossover": 0.8,
            "err_reduc": 0.25,
            "fit_reduc": 0.1,
            "subsumption": True,
            "pred_reset": False,
        },
        action={
            "type": "integer",
        },
        condition={
            "type": "hyperrectangle_mpr",
            "args": {
                "min": -1.0,
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
    model_ubr.json_write("data/pop/ubr.json")
    model_csr.json_read("data/pop/ubr.json")
    model_mmr.json_read("data/pop/ubr.json")
    model_mpr.json_read("data/pop/ubr.json")

    model_csr.json_write("data/pop/csr.json")
    model_mmr.json_write("data/pop/mmr.json")
    model_mpr.json_write("data/pop/mpr.json")

    eval_model(model_ubr, "ubr")
    eval_model(model_csr, "csr")
    eval_model(model_mmr, "mmr")
    eval_model(model_mpr, "mpr")

    # Sort test data for more straightforward prediction plotting.
    # X_test = X_test[:, 0:1]
    # perm = np.argsort(X_test.ravel())
    #
    # X_test = X_test[perm]
    # y_test = y_test[perm]
    # y_test_pred_ubr = y_test_pred_ubr[perm]
    #
    # fig, ax = plt.subplots(2, layout="constrained")
    # ax[0].plot(X_test, y_test, color="C2", marker="+")
    # ax[0].plot(X_test, y_test_pred_ubr, color="C1")
    # ax[0].set_title("ubr")
    # plt.show()


cli.add_command(run)

if __name__ == '__main__':
    cli()
