from __future__ import annotations
import torch
from typing import Callable
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torchfw.utils import golden_line_search, grid_line_search
from torchfw.fw import FrankWolfeBase


class AwayFrankWolfe(FrankWolfeBase):
    name = "Away FW"

    def __init__(
        self,
        params: ParamsT,
        oracle: Callable,
        ls_tol: float = 1e-3,
        step_size_fn: Callable | None = None,
        **kwargs
    ):
        self.ls_tol = ls_tol
        self.atoms = []
        self.alphas = []
        self.step_size_fn = step_size_fn
        self.curr_step = 0
        super().__init__(params, oracle)

    @torch.no_grad()
    def param_step(self, param: torch.Tensor, closure: Callable):
        away_step = False

        if len(self.atoms) == 0:
            self.atoms.append(param)
            self.alphas.append(1)

        fw_atom = self.oracle(-param.grad)
        fw_dir = fw_atom - param
        fw_inner = (-param.grad * fw_dir).sum()

        batch_atoms = torch.stack(self.atoms, dim=0)
        away_idx = (
            (param.grad * batch_atoms)
            .sum(dim=tuple(range(1, batch_atoms.ndim)))
            .argmax()
        )
        away_dir = param - self.atoms[away_idx]

        if fw_inner > (-param.grad * away_dir).sum():
            dir = fw_dir
            max_gamma = 1
        else:
            away_step = True
            dir = away_dir
            max_gamma = self.alphas[away_idx] / (1 - self.alphas[away_idx])

        if self.step_size_fn is not None:
            gamma = min(self.step_size_fn(self.curr_step), max_gamma)
        else:
            gamma = golden_line_search(
                param, dir, closure, max_gamma, tol=self.ls_tol
            )

        param += gamma * dir

        if away_step:
            if (max_gamma - gamma) < 1e-4:
                self.atoms.pop(away_idx)
                self.alphas.pop(away_idx)
            else:
                self.alphas = [(1 + gamma) * alpha for alpha in self.alphas]
                self.alphas[away_idx] = (
                    self.alphas[away_idx] - gamma
                )  # Already did (1 + gamma) * alpha
        else:
            if gamma == max_gamma:
                self.atoms = []
                self.alphas = []

            self.atoms.append(fw_atom)
            self.alphas = [(1 - gamma) * alpha for alpha in self.alphas]
            self.alphas.append(gamma)

        self.curr_step += 1
