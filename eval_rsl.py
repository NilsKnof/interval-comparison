import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import pandas as pd

# copied from the run-rsl-bench repository to generate the table, summarizing the used datasets: https://doi.org/10.5281/zenodo.7915821
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

if __name__ == '__main__':
    # copied parts from the run-rsl-bench repository: https://doi.org/10.5281/zenodo.7915821
    mlflow.set_tracking_uri('data/results/')
    df = mlflow.search_runs(experiment_names=["rsl"])
    df = df[df.status == "FINISHED"]
    index = ["params.data.DX", "params.data.K", "tags.mlflow.runName"]
    metrics = ["metrics.mse.test.csr", "metrics.mse.test.ubr", "metrics.mse.test.mmr", "metrics.mse.test.mpr"]
    df['tags.mlflow.runName'] = [s[-1] for s in df['tags.mlflow.runName']]
    df_metrics = df.reset_index()[index + metrics]

    pretty = {
        "params.data.DX": "$\mathcal{D}_\mathcal{X}$",
        "params.data.K": "$K$",
        "tags.mlflow.runName": "Data Seed"
    }

    df_metrics = df_metrics.set_index(index).stack().reset_index().rename(
        columns={
                    0: "Test MSE",
                    "level_3": "Algorithm",
                } | pretty)

    labels = {
        "metrics.mse.test.ubr": "UBR",
        "metrics.mse.test.csr": "CSR",
        "metrics.mse.test.mmr": "MMR",
        "metrics.mse.test.mpr": "MPR",
    }

    df_metrics["Algorithm"] = df_metrics["Algorithm"].apply(
        lambda s: labels[s])


    # df_metrics = pd.read_csv('Auswertung/results_test_mse.CSV')

    dx = 50
    k = 'all'
    df_metrics = df_metrics[df_metrics["$\mathcal{D}_\mathcal{X}$"] == dx]

    # needed for some of the used diagrams
    # df_metrics = df_metrics[df_metrics["$K$"] == k]

    df_metrics_mean = df_metrics.groupby(["$\mathcal{D}_\mathcal{X}$","$K$", "Data Seed", "Algorithm"])["Test MSE"].mean().reset_index()
    df_metrics_min = df_metrics.groupby(["$\mathcal{D}_\mathcal{X}$","$K$", "Data Seed", "Algorithm"])["Test MSE"].min().reset_index()
    df_metrics_low = df_metrics.groupby(["$\mathcal{D}_\mathcal{X}$","$K$", "Data Seed", "Algorithm"])["Test MSE"].quantile(.25).reset_index()
    df_metrics_med = df_metrics.groupby(["$\mathcal{D}_\mathcal{X}$","$K$", "Data Seed", "Algorithm"])["Test MSE"].quantile(.5).reset_index()
    df_metrics_high = df_metrics.groupby(["$\mathcal{D}_\mathcal{X}$","$K$", "Data Seed", "Algorithm"])["Test MSE"].quantile(.75).reset_index()

    g = sns.FacetGrid(data=df_metrics_mean,
                      row=pretty["params.data.DX"],
                      col=pretty["params.data.K"],
                      hue="Algorithm",
                      hue_order=list(labels.values()),
                      sharey=False,
                      margin_titles=True)

    g.map(
        sns.pointplot,
        pretty["tags.mlflow.runName"],
        "Test MSE",
        order=np.sort(df_metrics[pretty["tags.mlflow.runName"]].unique()),
        errorbar=("ci", 95),
        capsize=0.3,
        errwidth=2.0,
    )
    g.add_legend()
    plt.savefig(f"Auswertung/plots/dim_{dx}_pop_{k}_mean.pdf")

    g = sns.FacetGrid(data=df_metrics_min,
                      row=pretty["params.data.DX"],
                      col=pretty["params.data.K"],
                      hue="Algorithm",
                      hue_order=list(labels.values()),
                      sharey=False,
                      margin_titles=True)

    g.map(
        sns.pointplot,
        pretty["tags.mlflow.runName"],
        "Test MSE",
        order=np.sort(df_metrics[pretty["tags.mlflow.runName"]].unique()),
        errorbar=("ci", 95),
        capsize=0.3,
        errwidth=2.0,
    )
    g.add_legend()
    plt.savefig(f"Auswertung/plots/dim_{dx}_pop_{k}_min.pdf")

    g = sns.FacetGrid(data=df_metrics_low,
                      row=pretty["params.data.DX"],
                      col=pretty["params.data.K"],
                      hue="Algorithm",
                      hue_order=list(labels.values()),
                      sharey=False,
                      margin_titles=True)

    g.map(
        sns.pointplot,
        pretty["tags.mlflow.runName"],
        "Test MSE",
        order=np.sort(df_metrics[pretty["tags.mlflow.runName"]].unique()),
        errorbar=("ci", 95),
        capsize=0.3,
        errwidth=2.0,
    )
    g.add_legend()
    plt.savefig(f"Auswertung/plots/dim_{dx}_pop_{k}_low.pdf")

    g = sns.FacetGrid(data=df_metrics_med,
                      row=pretty["params.data.DX"],
                      col=pretty["params.data.K"],
                      hue="Algorithm",
                      hue_order=list(labels.values()),
                      sharey=False,
                      margin_titles=True)

    g.map(
        sns.pointplot,
        pretty["tags.mlflow.runName"],
        "Test MSE",
        order=np.sort(df_metrics[pretty["tags.mlflow.runName"]].unique()),
        errorbar=("ci", 95),
        capsize=0.3,
        errwidth=2.0,
    )
    g.add_legend()
    plt.savefig(f"Auswertung/plots/dim_{dx}_pop_{k}_med.pdf")

    g = sns.FacetGrid(data=df_metrics_high,
                      row=pretty["params.data.DX"],
                      col=pretty["params.data.K"],
                      hue="Algorithm",
                      hue_order=list(labels.values()),
                      sharey=False,
                      margin_titles=True)

    g.map(
        sns.pointplot,
        pretty["tags.mlflow.runName"],
        "Test MSE",
        order=np.sort(df_metrics[pretty["tags.mlflow.runName"]].unique()),
        errorbar=("ci", 95),
        capsize=0.3,
        errwidth=2.0,
    )
    g.add_legend()
    plt.savefig(f"Auswertung/plots/dim_{dx}_pop_{k}_high.pdf")
    pass
