import scipy

import mlflow
import numpy as np
import time
import os

import pandas as pd
import torch
from torch import nn
from torch.nn import RNNCell
from torch.optim import SGD, RMSprop
from torch.optim.adam import Adam
from tqdm import tqdm

from data import Data
from one_step_rnn import OneStepRNN
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt

from time_smoothed_ogd import TimeSmoothedOGD
from hessian import hessian

import time

from windowed_ogd import WindowedOGD

class Experiment():

    def __init__(self, **args):

        torch.manual_seed(1)

        if args["num_threads"] > 0:
            torch.set_num_threads(args["num_threads"])

        print(args)

        self.hidden_size = args["hidden_size"]
        self.data = Data().get_data(args["data"])

        self.input_dim = self.data[0][0].shape[1]
        self.output_size = self.data[0][1].shape[1]

        self.optimizer_name = args["optimizer"]

        self.time_decay_power = args["time_decay_power"]

        self.alpha = args["alpha"]
        self.w = args["window_size"] # window size
        self.clip_hh = args["clip_hh"]
        self.clip_ih = args["clip_ih"]
        self.lr_hh = args["lr_hh"]
        self.lr_ih = args["lr_ih"]
        self.lr_oh = args["lr_oh"]

        self.append_input = args["append_input"]

        if self.optimizer_name == "windowed_ogd":
            self.lr_hh /= (4 * self.hidden_size)
            self.lr_ih /= (4 * self.input_dim)

        self.time_decay = args["time_decay"]
        self.recurrent_cell = args["recurrent_cell"]
        self.n = args["truncation"]  #  backpropagation truncation including window. include gradients from last cell to 30th from end cell.
        self.mlflow = args["mlflow"]

        self.model = OneStepRNN(hidden_size=self.hidden_size, input_size=self.input_dim, recurrent_cell=self.recurrent_cell, output_size=self.output_size, append_input=self.append_input)

        if self.optimizer_name == "sgd":
            self.optimizer = SGD([
                {"name": "weight_hh", "params": self.model.weight_hh, "lr": self.lr_hh},
                {"name": "weight_ih", "params": self.model.weight_ih, "lr": self.lr_ih},
                {"name": "weight_oh", "params": self.model.weight_oh, "lr": self.lr_oh}
            ])
        elif self.optimizer_name == "adam":
            self.optimizer = Adam([
                {"name": "weight_hh", "params": self.model.weight_hh, "lr": self.lr_hh},
                {"name": "weight_ih", "params": self.model.weight_ih, "lr": self.lr_ih},
                {"name": "weight_oh", "params": self.model.weight_oh, "lr": self.lr_oh}
            ])
        elif self.optimizer_name == "rmsprop":
            self.optimizer = RMSprop([
                {"name": "weight_hh", "params": self.model.weight_hh, "lr": self.lr_hh},
                {"name": "weight_ih", "params": self.model.weight_ih, "lr": self.lr_ih},
                {"name": "weight_oh", "params": self.model.weight_oh, "lr": self.lr_oh}
            ])
        elif self.optimizer_name == "windowed_ogd":
            self.optimizer = WindowedOGD([
                {"name": "weight_hh", "params": self.model.weight_hh, "lr": self.lr_hh, "clip": self.clip_hh},
                {"name": "weight_ih", "params": self.model.weight_ih, "lr": self.lr_ih, "clip": self.clip_ih},
                {"name": "weight_oh", "params": self.model.weight_oh, "lr": self.lr_oh }
            ])

        self.init_hidden = torch.autograd.Variable(torch.zeros(1, self.hidden_size), requires_grad=False)

        self.final_losses = []
        self.loss_normalization_factor = np.sum(np.fromiter((i**self.alpha for i in range(1, self.w + 1)), np.float))

        if self.optimizer_name == "windowed_ogd" and self.n < self.w:
            raise RuntimeError

        self.experiment_name = args["experiment_name"]

        self.args = args

        self.mlflow_uuid = None

        self.output_decay = args["output_decay"]
        self.log_hessian = args["log_hessian"]
        self.log_difference = args["log_differences"]
        self.log_hessian_every = args["log_hessian_every"]

        if self.log_difference:
            self.log_differences_dict = {
                "W_ih": self.model.weight_ih.clone().data.numpy(),
                "dW_ih": np.zeros(self.model.weight_ih.shape),
                "W_hh": self.model.weight_hh.clone().data.numpy(),
                "dW_hh": np.zeros(self.model.weight_hh.shape),
            }


    def start_mlflow(self, experiment_name, cfg):
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

        run = mlflow.active_run()._info
        uuid = run.run_uuid

        for k, v in cfg.items():
            mlflow.log_param(k, v)

        return uuid

    def end_mlflow(self, avg_squared_error, runtime, predictions, targets, final_losses, max_eigs, differences):
        mlflow.log_metrics({
            "avg_squared_error": avg_squared_error,
            "runtime": runtime
        })

        # df = pd.DataFrame({"step": np.array(list(range(len(predictions))))+1, "prediction": predictions, "target": targets, "loss": final_losses})
        res_dict = {"step": np.array(list(range(len(final_losses))))+1, "loss": final_losses}
        if len(predictions) > 0:
            res_dict["prediction"] = predictions

        if len(targets) > 0:
            res_dict["target"] = targets

        if len(max_eigs) > 0:
            res_dict["max_eig"] = max_eigs

        if len(differences) > 0:
            diffs = {"diff_" + k: np.array([dic[k] for dic in differences]) for k in differences[0]}

            res_dict.update(diffs)

            ratio_hh = diffs["diff_dW_hh"] / diffs["diff_W_hh"]
            ratio_ih = diffs["diff_dW_ih"] / diffs["diff_W_ih"]

            res_dict["diff_ratio"] = np.maximum(ratio_hh, ratio_ih)
            mlflow.log_metric("gt_diff", (res_dict["diff_ratio"] > 1/self.lr_hh).sum())

        df = pd.DataFrame(res_dict)
        tmp_path = f"tmp/"

        try:
            os.makedirs(tmp_path)
        except FileExistsError:
            # directory already exists
            pass

        csv_path = os.path.join(tmp_path, f"{self.mlflow_uuid}.csv")
        df.to_csv(csv_path)

        mlflow.log_artifact(csv_path)

        mlflow.end_run()

    def run(self):

        if self.mlflow:
            self.mlflow_uuid = self.start_mlflow(self.experiment_name, self.args)

        start = time.time()

        runner = self.run_windowed if self.optimizer_name == "windowed_ogd" else self.run_normal

        predictions, targets, final_losses, max_eigs, differences = runner()

        end = time.time()

        avg_squared_error = np.mean(final_losses)
        runtime = end-start

        print(f"Average squared error: {avg_squared_error}")
        print(f"Running time (in seconds): {runtime}")

        if self.mlflow:
            self.end_mlflow(avg_squared_error, runtime, predictions, targets, final_losses, max_eigs, differences)

    def weight_oh_decay(self, timestep):
        # TODO: consider moving this method to windowed_ogd optimizer.

        with torch.no_grad():
            if self.time_decay:
                self.model.weight_oh.grad.data.mul_(torch.pow(torch.tensor(1/timestep), self.time_decay_power))

            self.model.weight_oh.data.mul_(torch.tensor(self.output_decay))

    def run_normal(self):
        final_losses = []
        predictions = []
        ground_truth = []
        max_eigs = []
        differences = []
        last_hidden = self.init_hidden

        for i in tqdm(range(1, len(self.data) + 1)):
            inputs, targets = zip(*self.data[max(i - self.n, 0):i])

            hiddens = []
            hiddens.append(last_hidden)

            def closure():
                self.model.zero_grad()

                for step, inp in enumerate(inputs):

                    prediction, hidden = self.model.forward(inp, hiddens[-1])

                    hiddens.append(hidden)

                    if step == len(inputs) - 1:

                        loss = torch.sum((prediction - targets[step]) ** 2)

                        loss.backward(retain_graph=True)

                        self.weight_oh_decay(i)

                        #predictions.append(prediction.item())
                        #ground_truth.append(targets[step])
                        final_losses.append(loss.item())

                        self.log_hessians(i, loss, max_eigs)

                    if step == 0:
                        last_hidden.data = hidden.data

                return loss

            self.optimizer.step(closure)
            self.log_differences(differences)

        return predictions, ground_truth, final_losses, max_eigs, differences

    def get_max_eig(self, loss):
        """
        grads_ih = torch.autograd.grad(loss, self.model.weight_ih, create_graph=True)[0].flatten()
        grads_hh = torch.autograd.grad(loss, self.model.weight_hh, create_graph=True)[0].flatten()

        grads = torch.cat([grads_ih, grads_hh])

        shape_all = grads.shape[0]

        grads2 = torch.zeros(shape_all, shape_all)

        for i, g in enumerate(grads):
            grads_ih = torch.autograd.grad(g, self.model.weight_ih, create_graph=True)[0].flatten()
            grads_hh = torch.autograd.grad(g, self.model.weight_hh, create_graph=True)[0].flatten()
            grads2[i, :] = torch.cat([grads_ih, grads_hh])

        h = grads2.detach().cpu().numpy()
        eigs = np.linalg.eigvals(h)


        """
        h = hessian(loss, [self.model.weight_hh, self.model.weight_ih])

        eigs = scipy.linalg.eigh(h, eigvals_only=True, eigvals=(h.shape[0]-1, h.shape[0]-1))
        eig_norms = np.absolute(eigs)

        max_eig = eig_norms.max()

        return max_eig

    def log_hessians(self, i, loss, max_eigs):
        if self.log_hessian:
            if (i - 1) % self.log_hessian_every == 0:
                max_eigs.append(self.get_max_eig(loss))
            else:
                max_eigs.append(None)

    def log_differences(self, differences):
        if self.log_difference:

            tmp_log_differences_dict = {
                "W_ih": self.model.weight_ih.clone().data.numpy(),
                "dW_ih": self.model.weight_ih.grad.clone().data.numpy(),
                "W_hh": self.model.weight_hh.clone().data.numpy(),
                "dW_hh": self.model.weight_hh.grad.clone().data.numpy(),
            }

            differences_dict = {
                "W_ih": np.linalg.norm(tmp_log_differences_dict["W_ih"] - self.log_differences_dict["W_ih"], ord="fro"),
                "dW_ih": np.linalg.norm(tmp_log_differences_dict["dW_ih"] - self.log_differences_dict["dW_ih"], ord="fro"),
                "W_hh": np.linalg.norm(tmp_log_differences_dict["W_hh"] - self.log_differences_dict["W_hh"], ord="fro"),
                "dW_hh": np.linalg.norm(tmp_log_differences_dict["dW_hh"] - self.log_differences_dict["dW_hh"], ord="fro"),
            }

            differences.append(differences_dict)

            self.log_differences_dict = tmp_log_differences_dict

    def run_windowed(self):
        final_losses = []
        predictions = []
        ground_truth = []
        max_eigs = []
        differences = []

        last_hidden = self.init_hidden

        for i in tqdm(range(1, len(self.data) + 1)):
            start_idx = max(i - self.n, 0)
            inputs, targets = zip(*self.data[start_idx:i])

            hiddens = [last_hidden]

            def closure():
                self.model.zero_grad()

                loss = 0.0

                hidden = hiddens[-1]
                for step, inp in enumerate(inputs):

                    prediction, hidden = self.model.forward(inp, hidden)

                    ws = step + 1 - min(start_idx, self.n - self.w)

                    if ws > 0:
                        loss += (ws ** self.alpha) * torch.sum((prediction - targets[step])**2)

                    if step + 1 == len(inputs):
                        last_squared_error = torch.sum((prediction - targets[step])**2)
                        final_losses.append(last_squared_error.item())
                        # predictions.append(prediction.item())
                        # ground_truth.append(targets[step])

                    if step == 0:
                        last_hidden.data = hidden.data

                loss /= self.loss_normalization_factor

                loss.backward(retain_graph=True)

                self.weight_oh_decay(i)

                self.log_hessians(i, loss, max_eigs)

                return last_squared_error

            self.optimizer.step(closure)

            self.log_differences(differences)

        return predictions, ground_truth, final_losses, max_eigs, differences
