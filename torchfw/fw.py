from __future__ import annotations
import torch

from typing import Callable
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torchfw.utils import golden_line_search, grid_line_search


def standard_step_size(step: int):
    return 2 / (step + 2)

class FrankWolfeBase(Optimizer):
    name = ""

    def __init__(self, params: ParamsT, oracle: Callable):
        self.oracle = oracle

        defaults = dict(differentiable=False)
        super().__init__(params, defaults)

    @_use_grad_for_differentiable
    def step(self, closure: Callable | None = None):
        for group in self.param_groups:
            params = group["params"]
            for param in params:
                self.param_step(param, closure=closure)

    def param_step(self, param: torch.Tensor, closure: Callable | None = None):
        pass


class FrankWolfe(FrankWolfeBase):
    name = "FW"

    def __init__(
        self,
        params: ParamsT,
        oracle: Callable,
        step_size_fn: Callable = standard_step_size,
        **kwargs
    ):
        self.step_size_fn = step_size_fn
        self.curr_step = 0

        defaults = dict(differentiable=False)
        super().__init__(params, oracle)

    @torch.no_grad()
    def param_step(self, param: torch.Tensor, **kwargs):
        direction = self.oracle(-param.grad) - param
        step_size = self.step_size_fn(self.curr_step)
        param += step_size * direction
        self.curr_step += 1

class FrankWolfeLS(FrankWolfeBase):
    name = "FW (LS)"

    def __init__(
        self,
        params: ParamsT,
        oracle: Callable,
        ls_tol: float = 1e-3,
        **kwargs
    ):
        self.ls_tol = ls_tol
        super().__init__(params, oracle)

    def param_step(self, param: torch.Tensor, closure: Callable):
        direction = self.oracle(-param.grad) - param

        if closure is None:
            raise ValueError(
                "Need to pass `closure` to step(), a function B x d1 x ... d_n -> B (i.e. a function that given a batch of elements of the same size as `params`, returns the loss for each element of that batch)"
            )

        step_size = golden_line_search(
            param, direction, closure, 1, tol=self.ls_tol
        )
        print(step_size)
        param += step_size * direction
