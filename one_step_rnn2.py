import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import GRUCell, RNNCell, LSTMCell

torch.manual_seed(0)


class OneStepRNN(nn.Module):

    def __init__(self, hidden_size, input_size, recurrent_cell, output_size):
        super().__init__()

        self.recurrent_cell = recurrent_cell
        self.hidden_size = hidden_size
        self.input_size = input_size

        recurrents = {
            "gru": GRUCell,
            "rnn": RNNCell,
            "lstm": LSTMCell
        }

        self.rnncell = recurrents[recurrent_cell](input_size=self.input_size, hidden_size=self.hidden_size, bias=False)
        self.output_size = output_size

        self.weight_hh = self.rnncell.weight_hh
        self.weight_ih = self.rnncell.weight_ih

        self.weight_oh = nn.Parameter(torch.empty((self.output_size, self.hidden_size)).normal_(0.0, 0.1), requires_grad=True)

    def forward(self, input, hidden):
        hidden = self.rnncell.forward(input, hidden)

        y_hat = self.weight_oh @ hidden.T # torch.cat([hidden.T, torch.ones(1, 1)], dim=0)

        return y_hat, hidden
