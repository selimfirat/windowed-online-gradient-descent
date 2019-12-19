import numpy as np
import time

import torch
from torch import nn
from torch.nn import RNNCell
from torch.optim import SGD

from one_step_rnn import OneStepRNN
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt

from time_smoothed_gd import TimeSmoothedGD

import time

torch.manual_seed(1)

class Main():

    def __init__(self):
        self.prev_inputs = 18

        self.hidden_size = 10

        self.input_dim  = self.prev_inputs + 1 # +1 because of the augmentation

        self.alpha = 10
        self.w = 10 # window size
        self.clip_hh = 0.9
        self.clip_ih = 0.9
        self.lr_hh = 0.5/(4*self.hidden_size)
        self.lr_ih = 0.5/(4*self.input_dim)
        self.threshold_hh = 2
        self.threshold_ih = 2
        self.n = 30 # backpropagation truncation including window. include gradients from last cell to 30th from end cell.

        self.model = OneStepRNN(hidden_size=self.hidden_size, w = self.w, input_size=self.input_dim)

        self.optimizer = TimeSmoothedGD([
            { "name": "weight_hh", "params": self.model.weight_hh, "lr": self.lr_hh, "clip": self.clip_hh, "threshold": self.threshold_hh},
            { "name": "weight_ih",  "params": self.model.weight_ih, "lr": self.lr_ih, "clip": self.clip_ih, "threshold": self.threshold_ih},
            { "name": "weight_oh", "params": self.model.weight_oh }
        ])

        self.init_hidden = torch.zeros(1, self.hidden_size) # torch.cat([torch.zeros(1, self.hidden_size-1), torch.ones(1, 1)], dim=1) # augmented hidden state

        self.last_hidden = self.init_hidden

        self.loss_normalization_factor = np.sum(i**self.alpha for i in range(1, self.w + 1))
        self.ols_normalization_factor = np.sum(i**(self.alpha/2) for i in range(1, self.w + 1))

        self.set_initial_weights(self.model)

        self.final_losses = []

    def set_initial_weights(self, model):
        weights = loadmat("init_cond.mat")
        weight_hh = weights["W"]
        weight_ih = weights["U"]
        weight_oh = weights["c"][:-1, :].T

        self.model.weight_hh.data = torch.from_numpy(weight_hh).float()
        self.model.weight_ih.data = torch.from_numpy(weight_ih).float()

        self.model.weight_oh.data = torch.from_numpy(weight_oh).float()


    def process_data(self):
        matfile = loadmat("elev_data.mat")
        sequence = matfile["nngc_data"]

        sequence = sequence - sequence.mean(axis=0) / sequence.std(axis=0)

        data = []
        for i in range(1000):
            inp = torch.FloatTensor(np.append(sequence[i, :-1],  [[1]])).reshape(1, -1)
            out = torch.FloatTensor([sequence[i, -1]]).reshape(1, -1)
            data.append((inp, out))

        return data

    def process_data_mus(self):

        matfile = loadmat("nngc_mus.mat")
        sequence = matfile["nngc_data"]

        data = []
        for i in range(self.prev_inputs, sequence.shape[0]-1):
            inp = torch.FloatTensor(np.append(sequence[i-self.prev_inputs:i],  [[1]])).reshape(1, -1)
            out = torch.FloatTensor(sequence[i:i+1]).reshape(1, -1)
            data.append((inp, out))

        return data

    def run(self):
        data = self.process_data()

        start = time.time()
        for i in range(self.w, len(data)):
            inputs, targets = zip(*data[max(0, i - self.n):i]) # TODO: optimize this step

            def closure():
                self.optimizer.zero_grad()
                self.model.zero_grad()

                hidden = self.last_hidden

                loss = 0.0

                hiddens = []
                ys = []

                for step, inp in enumerate(inputs):
                    if step == 0:
                        hidden = hidden.detach()

                    prediction, hidden = self.model.forward(inp, hidden)

                    if step + self.w >= len(inputs):
                        ws = self.w - len(inputs) + step + 1

                        loss += (ws ** self.alpha) * torch.sum((prediction - targets[step])**2)
                        hiddens.append(hidden*(ws ** (self.alpha/2))/self.ols_normalization_factor)
                        ys.append(targets[step].item()*(ws ** (self.alpha/2))/self.ols_normalization_factor) # TODO: get as slice for a trivial optimize

                    if step == len(inputs) - 1:
                        last_squared_error = torch.sum((prediction - targets[step])**2).item()

                loss /= self.loss_normalization_factor

                loss.backward(retain_graph=True)


                return last_squared_error, hiddens, ys

            last_squared_error = self.optimizer.step(closure)

            self.final_losses.append(last_squared_error)

            print("Average squared erorr:", np.mean(self.final_losses))
        end = time.time()

        print(f"Average squared error: {np.mean(self.final_losses)}")
        print(f"Running time (in seconds): {end-start}")

    def plot(self):

        ax = sns.lineplot(x="Timestep", y="Squared Error", data = {
            "Timestep": list(range(len(self.final_losses))),
            "Squared Error": self.final_losses
        })

        plt.show()


if __name__ == "__main__":
    main = Main()
    main.run()