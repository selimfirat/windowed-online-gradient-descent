import numpy as np

import torch
from scipy.io import loadmat
import os

class Data:

    def __init__(self, data_path="data"):
        self.data_path = data_path

    def elevator(self):
        matfile = loadmat(os.path.join(self.data_path, "elev_data.mat"))
        sequence = matfile["nngc_data"]

        sequence = (sequence - sequence.mean(axis=0)) / sequence.std(axis=0)

        data = []
        for i in range(len(sequence)):
            inp = torch.FloatTensor(np.append(sequence[i, :-1],  [[1]])).reshape(1, -1)
            out = torch.FloatTensor([sequence[i, -1]]).reshape(1, -1)
            data.append((inp, out))

        return data

    def elevator_1000(self):
        matfile = loadmat(os.path.join(self.data_path, "elev_data.mat"))
        sequence = matfile["nngc_data"]

        sequence = (sequence - sequence.mean(axis=0)) / sequence.std(axis=0)

        data = []
        for i in range(1000):
            inp = torch.FloatTensor(np.append(sequence[i, :-1],  [[1]])).reshape(1, -1)
            out = torch.FloatTensor([sequence[i, -1]]).reshape(1, -1)
            data.append((inp, out))

        return data

    def puma32f(self):
        matfile = loadmat(os.path.join(self.data_path, "puma32f.mat"))
        sequence = matfile["nngc_data"]

        data = []
        for i in range(len(sequence)):
            inp = torch.FloatTensor(np.append(sequence[i, :-1],  [[1]])).reshape(1, -1)
            out = torch.FloatTensor([sequence[i, -1]]).reshape(1, -1)
            data.append((inp, out))

        return data

    def pumadyn(self):
        matfile = loadmat(os.path.join(self.data_path, "puma3_data.mat"))

        sequence = matfile["our_data"]

        data = []
        for i in range(len(sequence)):
            inp = torch.FloatTensor(np.append(sequence[i, :-1],  [[1]])).reshape(1, -1)
            out = torch.FloatTensor([sequence[i, -1]]).reshape(1, -1)
            data.append((inp, out))

        return data

    def alcoa(self):
        matfile = loadmat(os.path.join(self.data_path, "fin_multi_data2.mat"))

        sequence = matfile["nngc_data"]

        data = []
        for i in range(len(sequence)-1):
            inp = torch.FloatTensor(np.append(sequence[i, :],  [[1]])).reshape(1, -1)
            out = torch.FloatTensor([sequence[i+1, 0]]).reshape(1, -1)
            data.append((inp, out))

        return data

    def euro(self):
        matfile = loadmat(os.path.join(self.data_path, "euro_data2.mat"))

        sequence = matfile["nngc_data"]

        data = []
        for i in range(len(sequence)-1):
            inp = torch.FloatTensor(np.append(sequence[i, :],  [[1]])).reshape(1, -1)
            out = torch.FloatTensor([sequence[i+1, 0]]).reshape(1, -1)
            data.append((inp, out))

        return data


    def get_data(self, name = "elevator"):
        data_dict = {
            "elevator": self.elevator,
            "elevator_1000": self.elevator_1000,
            "pumadyn": self.pumadyn,
            "puma32f": self.puma32f,
            "euro": self.euro,
            "alcoa": self.alcoa
        }

        return data_dict[name]()
