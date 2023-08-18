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


@click.group()
def cli():
    pass


@click.command()
@click.argument('NPZFILE')
def run(npzfile):
    mlflow.set_tracking_uri('data/results/')
    df = mlflow.search_experiments("run")
    pass



cli.add_command(run)

if __name__ == '__main__':
    mlflow.set_tracking_uri('data/results/')
    df = mlflow.search_runs(experiment_names=["rsl-K5-DX5-N1000"])
    pass
    cli()
