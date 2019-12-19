import torch
from torch import optim


class WindowedOGD(optim.Optimizer):

    def __init__(self, params, defaults={}):

        super().__init__(params, defaults)

    def step(self, closure):

        loss = closure()

        for i, group in enumerate(self.param_groups):

            name = group["name"]
            lr = group["lr"]

            if name == "weight_oh":
                W_out = group["params"][0]
                if W_out.grad is None:
                    print("No W_out gradient")
                    continue

                W_out.data = W_out.add(- torch.tensor(lr) *  W_out.grad)
                with torch.no_grad():
                    length_project = 15 / (torch.clamp(torch.norm(W_out, p=2, dim=1), min=15))
                    W_out.data = W_out.data.T.mul(length_project).T

            else:
                clip = group["clip"]

                W = group["params"][0]

                if len(group["params"]) != 1:
                    raise Exception

                if W.grad is None: # check whether required
                    print("none gradient")
                    continue

                d_W = W.grad.data
                W_new = W.add(-lr * d_W)

                U, S, V = torch.svd(W_new)

                S_clipped = torch.clamp(S, max=clip)

                W_new = U @ torch.diag(S_clipped) @ V.T

                W.data = W_new.data

        return loss
