import json

from libs import xcsf
import click
import os
import numpy as np
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow


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
@click.option("-s",
              "--startseed",
              type=click.IntRange(min=0),
              default=0,)
@click.option("-e",
              "--endseed",
              type=click.IntRange(min=0),
              default=0,)
@click.option("--experiment-name", type=str, required=True)
@click.argument('NPZFILE')
@click.pass_context
def runmany(ctx, startseed, endseed, experiment_name, npzfile):
    for seed in range(startseed, endseed + 1):
        ctx.invoke(
            run,
            experiment_name=experiment_name,
            npzfile=npzfile,
            seed=seed,
        )


@click.command()
@click.option("-s",
              "--seed",
              type=click.IntRange(min=0),
              default=0,)
@click.option("--experiment-name", type=str, required=True)
@click.argument('NPZFILE')
def run(experiment_name, npzfile, seed):
    y_dim = 1
    n_actions = 1
    alpha = 0.1
    beta = 0.2
    delta = 0.1
    e0 = 0.01
    init_error = 0
    init_fitness = 0.01
    m_probation = 10000
    nu = 5
    omp_num_threads = 8
    perf_trials = 1000
    max_trials = 100000
    pop_init = True
    pop_size = 200
    random_state = seed
    set_subsumption = True
    theta_del = 20
    theta_sub = 100
    loss_func = "mse"
    ea = {
        "select_type": "tournament",
        "select_size": 0.4,
        "theta_ea": 25,
        "lambda": 2,
        "p_crossover": 0.8,
        "err_reduc": 0.25,
        "fit_reduc": 0.1,
        "subsumption": True,
        "pred_reset": False,
    }
    action = {
        "type": "integer",
    }
    spread_min = 1.0
    conditionArgs = {
        "min": -1.0,
        "max": 1.0,
        "spread_min": 0.1,
        "eta": 0,
    }
    prediction = {
        "type": "nlms_linear",
        "args": {
            "eta": 1.0,
            "eta_min": 0.0001,
            "evolve_eta": True,
        }
    }

    mlflow.set_tracking_uri('data/results/')
    mlflow.set_experiment("rsl-K5-DX5-N1000")
    print("Tracking URI:", mlflow.get_tracking_uri())
    with mlflow.start_run(run_name=npzfile[5:-4]) as run:
        # training data
        data = np.load(npzfile)
        X, y = get_train(data)
        N, DX = X.shape
        mlflow.log_params({
            "data.N": N,
            "data.DX": DX,
            "seed": seed,
            "pop_size": pop_size,
            "max_trials": max_trials,
        })

        # test data
        X_test, y_test = get_test(data)

        K = len(data["centers"])
        mlflow.log_params({
            "data.K": K,
            "data.linear_model_mse":        data["linear_model_mse"],
            "data.linear_model_mae":        data["linear_model_mae"],
            "data.linear_model_rsquared":   data["linear_model_rsquared"],
            "data.rsl_model_mse":           data["rsl_model_mse"],
            "data.rsl_model_mae":           data["rsl_model_mae"],
            "data.rsl_model_rsquared":      data["rsl_model_rsquared"],
        })

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
            mae_test = mean_absolute_error(y_test_pred, y_test)
            print(f"MSE test ({label}):", mse_test)
            print(f"MAE test ({label}):", mae_test)
            mse_train = mean_squared_error(y_pred, y)
            mae_train = mean_absolute_error(y_pred, y)
            print(f"MSE train ({label}):", mse_train)
            print(f"MAE train ({label}):", mae_train)
            mlflow.log_metrics({
                f"mse.test.{label}": mse_test,
                f"mae.test.{label}": mae_test,
                f"mse.train.{label}": mse_train,
                f"mae.train.{label}": mae_train,
            })

            return y_pred, y_test_pred

        model_ubr = xcsf.XCS(
            x_dim=DX,
            y_dim=y_dim,
            n_actions=n_actions,
            alpha=alpha,
            beta=beta,
            delta=delta,
            e0=e0,
            init_error=init_error,
            init_fitness=init_fitness,
            m_probation=m_probation,
            nu=nu,
            omp_num_threads=omp_num_threads,
            perf_trials=perf_trials,
            max_trials=max_trials,
            pop_init=pop_init,
            pop_size=pop_size,
            random_state=random_state,
            set_subsumption=set_subsumption,
            theta_del=theta_del,
            theta_sub=theta_sub,
            loss_func=loss_func,
            ea=ea,
            action=action,
            condition={
                "type": "hyperrectangle_ubr",
                "args": conditionArgs,
            },
            prediction=prediction,
        )

        pop_init = False

        model_csr = xcsf.XCS(
            x_dim=DX,
            y_dim=y_dim,
            n_actions=n_actions,
            alpha=alpha,
            beta=beta,
            delta=delta,
            e0=e0,
            init_error=init_error,
            init_fitness=init_fitness,
            m_probation=m_probation,
            nu=nu,
            omp_num_threads=omp_num_threads,
            perf_trials=perf_trials,
            max_trials=max_trials,
            pop_init=pop_init,
            pop_size=pop_size,
            random_state=random_state,
            set_subsumption=set_subsumption,
            theta_del=theta_del,
            theta_sub=theta_sub,
            loss_func=loss_func,
            ea=ea,
            action=action,
            condition={
                "type": "hyperrectangle_csr",
                "args": conditionArgs,
            },
            prediction=prediction,
        )

        model_mmr = xcsf.XCS(
            x_dim=DX,
            y_dim=y_dim,
            n_actions=n_actions,
            alpha=alpha,
            beta=beta,
            delta=delta,
            e0=e0,
            init_error=init_error,
            init_fitness=init_fitness,
            m_probation=m_probation,
            nu=nu,
            omp_num_threads=omp_num_threads,
            perf_trials=perf_trials,
            max_trials=max_trials,
            pop_init=pop_init,
            pop_size=pop_size,
            random_state=random_state,
            set_subsumption=set_subsumption,
            theta_del=theta_del,
            theta_sub=theta_sub,
            loss_func=loss_func,
            ea=ea,
            action=action,
            condition={
                "type": "hyperrectangle_mmr",
                "args": conditionArgs,
            },
            prediction=prediction,
        )

        model_mpr = xcsf.XCS(
            x_dim=DX,
            y_dim=y_dim,
            n_actions=n_actions,
            alpha=alpha,
            beta=beta,
            delta=delta,
            e0=e0,
            init_error=init_error,
            init_fitness=init_fitness,
            m_probation=m_probation,
            nu=nu,
            omp_num_threads=omp_num_threads,
            perf_trials=perf_trials,
            max_trials=max_trials,
            pop_init=pop_init,
            pop_size=pop_size,
            random_state=random_state,
            set_subsumption=set_subsumption,
            theta_del=theta_del,
            theta_sub=theta_sub,
            loss_func=loss_func,
            ea=ea,
            action=action,
            condition={
                "type": "hyperrectangle_mpr",
                "args": conditionArgs,
            },
            prediction=prediction,
        )

        model_ubr.json_write("data/pop/ubr.json")
        model_csr.json_read("data/pop/ubr.json")
        model_mmr.json_read("data/pop/ubr.json")
        model_mpr.json_read("data/pop/ubr.json")

        eval_model(model_ubr, "ubr")
        eval_model(model_csr, "csr")
        eval_model(model_mmr, "mmr")
        eval_model(model_mpr, "mpr")


cli.add_command(run)
cli.add_command(runmany)

if __name__ == '__main__':
    cli()
