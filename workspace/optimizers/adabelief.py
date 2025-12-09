import math
from torch.optim import Optimizer
import torch


class AdaBelief(Optimizer):
    """PyTorch implementation of AdaBelief optimizer.
    Reference: AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients (NeurIPS 2020).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-16,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        amsgrad=False,
        rectify=True,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 < betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 < betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
            amsgrad=amsgrad,
            rectify=rectify,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_var"] = torch.zeros_like(p.data)
                    if group["amsgrad"]:
                        state["max_exp_avg_var"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_var = state["exp_avg_var"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] != 0:
                    if group["weight_decouple"]:
                        if not group["fixed_decay"]:
                            p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                        else:
                            p.data.add_(p.data, alpha=-group["weight_decay"])
                    else:
                        grad = grad.add(p.data, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if group["amsgrad"]:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom_var = max_exp_avg_var
                else:
                    denom_var = exp_avg_var

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group["lr"] / bias_correction1

                if group["rectify"]:
                    sma_max = 2 / (1 - beta2) - 1
                    sma = sma_max - 2 * step * (beta2 ** step) / bias_correction2
                    if sma >= 5:
                        r_t = math.sqrt(
                            ((sma - 4) * (sma - 2) * sma_max)
                            / ((sma_max - 4) * (sma_max - 2) * sma)
                        )
                        denom = (denom_var / bias_correction2).sqrt().add_(group["eps"])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * r_t)
                    else:
                        p.data.add_(exp_avg, alpha=-step_size)
                else:
                    denom = (denom_var / bias_correction2).sqrt().add_(group["eps"])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
