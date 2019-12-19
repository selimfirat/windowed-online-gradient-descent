from experiment import Experiment
from argparse import ArgumentParser
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool
import mlflow

if __name__ == "__main__":

    params = {
        "lr": [0.5, 1, 2, 3, 4], #[8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625],
        "recurrent_cell": ["rnn"],
        "model": ["one_step_rnn"],
        "optimizer": ["windowed_ogd"],
        "hidden_size": [16, 32, 64],
        "alpha": [0],
        "clip_ih": [0.95],
        "clip_hh": [0.95],
        "window_size": [120],
        "truncation": [30], # plus window size for windowed_ogd
        "time_decay": [False, True],
        "experiment_name": ["all_datasets"],
        "mlflow": [True],
        "data": ["pumadyn", "puma32f", "elevator", "alcoa", "euro"], # "elevator", "pumadyn", "alcoa", "euro"]
        "num_threads": [1],
        "log_hessian": [False],
        "log_differences": [True],
        "output_decay": [1.0],
        "time_decay_power": [0.5],
        "log_hessian_every": [50]
    }

    param_grid = ParameterGrid(params)


    def experiment(args):
        try:
            experiment = Experiment(**args)

            experiment.run()
        except Exception as e:
            try:
                mlflow.end_run()
            except:
                pass

            print(e)


    pool = Pool(8)

    args_set = []

    for args in param_grid:
        args["lr_oh"] = args["lr"]
        args["lr_ih"] = args["lr"]
        args["lr_hh"] = args["lr"]


        if args["optimizer"] == "windowed_ogd":
            args["lr_oh"] = 1
            args["truncation"] += args["window_size"]

            if not args["time_decay"]:
                continue

        else:
            if args["time_decay"] \
                    or args["alpha"] != 0 \
                    or args["lr_hh"] > 1 \
                    or args["lr_ih"] > 1 \
                    or args["lr_oh"] > 1:
                continue

        args_set.append(args)




    # print(len(args_set))
    pool.map(experiment, args_set)
