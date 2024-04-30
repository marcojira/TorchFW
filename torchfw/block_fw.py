from __future__ import annotations
import torch
from typing import Callable
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torchfw.utils import batch_golden_line_search, grid_line_search


def standard_step_size(step: int):
    return 2 / (step + 2)


class BlockFrankWolfe(Optimizer):
    name = "Block FW"

    def __init__(
        self,
        params: ParamsT,
        oracle: Callable,
        ls: bool | str = False,
        step_size_fn: Callable | None = None,
        ls_batch_size: int = 1000,
    ):
        self.oracle = oracle
        self.step_size_fn = step_size_fn
        self.ls = ls
        self.ls_batch_size = ls_batch_size

        self.curr_step = 0
        defaults = dict(differentiable=False)
        super().__init__(params, defaults)

    @_use_grad_for_differentiable
    def step(
        self,
        block: int,
        batch_train_loss: Callable | None = None,
    ):
        for group in self.param_groups:
            params = group["params"]
            for param in params:
                direction = torch.zeros_like(param)
                direction[:, block, :] = (
                    self.oracle(param.grad)[:, block, :] - param[:, block, :]
                )

                if self.ls:
                    if batch_train_loss is None:
                        raise ValueError(
                            "Need to pass `batch_train_loss` to step(), a function that takes B x d1 x ... d_n -> B (i.e. that computes the loss for each element of the batch and returns it as a tensor)"
                        )

                    if self.ls == "batch_golden":
                        step_size = batch_golden_line_search(
                            param, direction, batch_train_loss, 1, self.ls_batch_size
                        )
                    elif self.ls == "grid":
                        step_size = grid_line_search(
                            param, direction, batch_train_loss, 1, self.ls_batch_size
                        )
                    else:
                        raise ValueError(
                            f"Invalid `ls` argument of {self.ls}. Supported options are `batch_golden` and `grid`."
                        )
                else:
                    step_size = self.step_size_fn(self.curr_step)

                param += step_size * direction

        self.curr_step += 1
