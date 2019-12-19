import torch
from torch import optim


class TimeSmoothedOGD(optim.Optimizer):

    def __init__(self, params, defaults={}):

        super().__init__(params, defaults)

    def step(self, closure):

        res_steps = [float("Inf") for group in self.param_groups if group["name"] != "weight_oh"]

        loss = None

        while any(res_step >= self.param_groups[i]["threshold"] for i, res_step in enumerate(res_steps)):

            if loss == None:
                loss, _, _ = closure()
            else:
                closure()

            for i, group in enumerate(self.param_groups):

                name = group["name"]

                if name == "weight_oh":
                    W_out = group["params"][0]
                else:
                    lr = group["lr"]
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
                    res_steps[i] = torch.sqrt(torch.sum(d_W**2)).item()

                    W.data = W_new.data

            _, hiddens, targets = closure()

            """

            Y = torch.tensor(targets, dtype=torch.float).reshape(len(targets), 1)

            H = torch.cat(hiddens, dim=0)
            W, _ = torch.lstsq(Y, H + 0.001 * torch.eye(H.shape[0]))

            fnorm = torch.sqrt(torch.sum(W ** 2)).item()
            if fnorm > 5:
                W *= 5/fnorm

            W_out.data = W.T
            """

            Y = torch.tensor(targets, dtype=torch.float).reshape(len(targets), 1)

            H = torch.cat(hiddens, dim=0)

            W = torch.inverse(H.T @ H + 0.001 * torch.eye(H.T.shape[0])) @ H.T @ Y

            fnorm = torch.sqrt(torch.sum(W ** 2)).item()
            if fnorm > 5:
                W *= 5/fnorm

            W_out.data = W.T

            """

            W_out.data = W_out.add(-lr * W_out.grad)
            """
        print(res_steps)

        return loss