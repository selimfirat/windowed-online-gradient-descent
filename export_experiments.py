import os

import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient("http://localhost:6161/")

experiments = client.list_experiments()


experiments_path = "mlruns"

base_path = "results"
for experiment in experiments:

    fname = f"{experiment.experiment_id}_{experiment.name}.csv"
    fpath = os.path.join(base_path, fname)

    print(fpath)
#    print(client.search_runs([experiment.experiment_id], max_results=50))

    edir = os.path.join(experiments_path, str(experiment.experiment_id))

    edirs = [o for o in os.listdir(edir) if os.path.isdir(os.path.join(edir,o))]

    dicts = []

    for dd in edirs:
        d = os.path.join(edir, dd)
        params = {}
        metrics = {}
        param_files = [n for n in os.listdir(os.path.join(d, "params"))]
        for pf in param_files:
            params[pf] = open(os.path.join(d, "params", pf)).readlines()[0]

        metric_files = [m for m in os.listdir(os.path.join(d, "metrics"))]

        for mf in metric_files:

            metrics[mf] = float(open(os.path.join(d, "metrics", mf)).readlines()[0].split()[1])

        params.update(metrics)

        params["uuid"] = dd

        if "avg_squared_error" not in params:
            params["avg_squared_error"] = 99999
        dicts.append(params)

    df = pd.DataFrame(dicts)

    if len(df) > 1:
        df = df.sort_values(by="avg_squared_error")

    df.to_csv(os.path.join(fpath))
