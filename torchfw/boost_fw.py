from __future__ import annotations
import torch
from typing import Callable
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torchfw.utils import golden_line_search, grid_line_search
from torchfw.fw import FrankWolfeBase


class BoostedFrankWolfe(FrankWolfeBase):
    name = "Boosted FW"

    def __init__(
        self,
        params: ParamsT,
        oracle: Callable,
        ls_tol: float = 1e-3,
        max_alignment_steps: int = 10,
        step_size_fn: Callable | None = None,
    ):
        self.ls_tol = ls_tol
        self.max_alignment_steps = max_alignment_steps
        self.step_size_fn = step_size_fn
        self.curr_step = 0

        super().__init__(params, oracle)

    @torch.no_grad()
    def param_step(self, param: torch.Tensor, closure: Callable):
        dir = torch.zeros_like(param).to(param.device)
        Lambda = 0

        for k in range(self.max_alignment_steps):
            rk = -param.grad - dir
            vk = self.oracle(rk)

            d_norm = torch.norm(dir)
            prev = -dir / d_norm
            new = vk - param

            prev_inner = (prev * rk).sum()
            new_inner = (new * rk).sum()

            if d_norm > 0 and prev_inner > new_inner:
                uk = prev
                last_uk = "prev"
            else:
                uk = new
                last_uk = "new"

            lambd = (rk * uk).sum() / (torch.norm(uk) ** 2)
            new_dir = dir + lambd * uk

            if torch.norm(new_dir - dir) < 1e-6:
                dir = new_dir
                break
            else:
                dir = new_dir
                if last_uk == "new":
                    Lambda = Lambda + lambd
                else:
                    Lambda = Lambda * (1 - lambd / d_norm)

        final_dir = dir / Lambda

        if self.step_size_fn is not None:
            gamma = self.step_size_fn(self.curr_step)
        else:
            gamma = golden_line_search(
                param,
                final_dir,
                closure,
                1,
                self.ls_tol,
            )
        param += gamma * final_dir
        self.curr_step += 1
