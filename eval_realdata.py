import csv
import json

from libs import xcsf
import click
import os
import numpy as np
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import seaborn as sns
import pandas as pd


@click.group()
def cli():
    pass


@click.command()
@click.argument('NPZFILE')
def run(npzfile):
    mlflow.set_tracking_uri('data/results/')
    df = mlflow.search_experiments("run")
    pass


# copied from the run-rsl-bench repository: https://doi.org/10.5281/zenodo.7915821
@cli.command()
@click.argument("PATH")
def datasets(path):
    print("test")
    fnames = os.listdir(path)

    def entry(fname):
        data = np.load(f"{path}/{fname}", allow_pickle=True)
        X = data["X"]
        N, DX = X.shape
        centers = data["centers"]
        K = len(centers)

        return dict(N=N,
                    DX=DX,
                    K=K,
                    linear_model_mse=data["linear_model_mse"][()],
                    linear_model_rsquared=data["linear_model_rsquared"][()],
                    rsl_model_mse=data["rsl_model_mse"][()],
                    rsl_model_rsquared=data["rsl_model_rsquared"][()])

    index = ["DX", "K", "N"]
    df = pd.DataFrame(map(entry, fnames))

    summary = df.set_index(index).sort_index().index
    print(np.unique(summary, return_counts=True))

    fig, ax = plt.subplots(2, layout="constrained", figsize=(10, 3 * 10))
    sns.boxplot(
        data=df,
        x="DX",
        hue="K",
        y="linear_model_mse",
        ax=ax[0],
    )
    sns.boxplot(
        data=df,
        x="DX",
        hue="K",
        y="rsl_model_mse",
        ax=ax[1],
    )
    sns.stripplot(data=df,
                  x="DX",
                  hue="K",
                  y="linear_model_mse",
                  ax=ax[0],
                  dodge=True,
                  palette="dark:0",
                  size=3)
    sns.stripplot(data=df,
                  x="DX",
                  hue="K",
                  y="rsl_model_mse",
                  ax=ax[1],
                  dodge=True,
                  palette="dark:0",
                  size=3)
    plt.show()

    means = df.groupby(index).mean().round(2)
    n_datasets = df.groupby(index).apply(len).unique()
    print(f"Mean of the {n_datasets} datasets per {index} combination")
    print(means[["linear_model_mse", "rsl_model_mse"]].to_latex())

    # A more space saving representation.
    means = means[["linear_model_mse", "rsl_model_mse"
                   ]].reset_index().set_index(["DX", "N", "K"]).unstack("K")
    print(means.to_latex())

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


cli.add_command(run)

cli.add_command(datasets)
if __name__ == '__main__':
    runName="real_data_sd"
    name="sd"
    # cli()
    # copied from the run-rsl-bench repository: https://doi.org/10.5281/zenodo.7915821
    mlflow.set_tracking_uri('data/realworld/')
    df = mlflow.search_runs(experiment_names=[runName])
    df = df[df.status == "FINISHED"]
    # pass
    index = ["params.seed"]
    metrics = ["metrics.mse.test.csr", "metrics.mse.test.ubr", "metrics.mse.test.mmr", "metrics.mse.test.mpr"]
    # df['tags.mlflow.runName'] = [s[-1] for s in df['tags.mlflow.runName']]
    df_metrics = df.reset_index()[index + metrics]

    pretty = {
        "params.seed": "Data Seed"
    }

    df_metrics = df_metrics.set_index(index).stack().reset_index().rename(
        columns={
                    0: "Test MSE",
                    "level_1": "Algorithm",
                } | pretty)

    labels = {
        "metrics.mse.test.ubr": "2UBR",
        "metrics.mse.test.csr": "1CSR",
        "metrics.mse.test.mmr": "3MMR",
        "metrics.mse.test.mpr": "4MPR",
    }

    df_metrics["Algorithm"] = df_metrics["Algorithm"].apply(
       lambda s: labels[s])

    # df_metrics = pd.read_csv('Auswertung/results_test_mse.CSV')
    # dx = 50
    # k = 'all'
    # df_metrics = df_metrics[df_metrics["$\mathcal{D}_\mathcal{X}$"] == dx]
    # df_metrics = df_metrics[df_metrics["$K$"] == k]

    # df_metrics_mean = df_metrics.groupby(["$\mathcal{D}_\mathcal{X}$","$K$", "Data Seed", "Algorithm"])["Test MSE"].mean().reset_index()
    # df_metrics_min = df_metrics.groupby(["Algorithm"])["Test MSE"].min().reset_index()
    df_metrics_low = df_metrics.groupby(["Algorithm"])["Test MSE"].quantile(.25).reset_index()
    # df_metrics_med = df_metrics.groupby(["$\mathcal{D}_\mathcal{X}$","$K$", "Data Seed", "Algorithm"])["Test MSE"].quantile(.5).reset_index()
    # df_metrics_high = df_metrics.groupby(["$\mathcal{D}_\mathcal{X}$","$K$", "Data Seed", "Algorithm"])["Test MSE"].quantile(.75).reset_index()
    # df_metrics_low2 = df_metrics
    group = df_metrics.groupby(["Algorithm"])
    quantiles = group['Test MSE'].quantile(.25)
    result = df_metrics[df_metrics.apply(lambda row: row['Test MSE'] < quantiles[row['Algorithm']], axis=1)]
    result = result.sort_values(by=["Algorithm"])
    labels = {
        "2UBR": "UBR",
        "1CSR": "CSR",
        "3MMR": "MMR",
        "4MPR": "MPR",
    }

    result["Algorithm"] = result["Algorithm"].apply(
        lambda s: labels[s])
    # g = sns.pointplot(    #     data=df_metrics,
    #     x=pretty["params.seed"],
    #     y="Test MSE",
    #     hue="Algorithm",
    #     order=np.flip(df_metrics[pretty["params.seed"]].unique()),
    #     errorbar=("ci", 95),
    #     capsize=0.3,
    #     errwidth=2.0,
    # )
    # plt.savefig(f"Auswertung/plots/{name}_all.pdf")

    g = sns.swarmplot(
        data=result,
        x='Algorithm',
        y="Test MSE",
        hue="Algorithm",
    )

    # plt.show()
    plt.savefig(f"Auswertung/plots/{name}_low.pdf")
    pass
