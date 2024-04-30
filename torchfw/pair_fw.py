from __future__ import annotations
import torch
from typing import Callable
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, _use_grad_for_differentiable
from torchfw.utils import golden_line_search, grid_line_search
from torchfw.fw import FrankWolfeBase


class PairwiseFrankWolfe(FrankWolfeBase):
    name = "Pairwise FW"

    def __init__(
        self,
        params: ParamsT,
        oracle: Callable,
        ls_tol: float = 1e-3,
        **kwargs
    ):
        self.ls_tol = ls_tol
        self.atoms = []
        self.alphas = []

        super().__init__(params, oracle)

    @torch.no_grad()
    def param_step(self, param: torch.Tensor, closure: Callable):
        with torch.no_grad():
            if len(self.atoms) == 0:
                self.atoms.append(param)
                self.alphas.append(1)

            batch_atoms = torch.stack(self.atoms, dim=0)
            away_idx = (
                (param.grad * batch_atoms)
                .sum(dim=tuple(range(1, batch_atoms.ndim)))
                .argmax()
            )

            new_atom = self.oracle(-param.grad)
            pairwise_direction = new_atom - self.atoms[away_idx]
            max_step_size = self.alphas[away_idx]

            gamma = golden_line_search(
                param, pairwise_direction, closure, max_step_size, tol=self.ls_tol
            )

            param += gamma * pairwise_direction
            self.atoms.append(new_atom)
            self.alphas[away_idx] = self.alphas[away_idx] - gamma
            self.alphas.append(gamma)

            if self.alphas[away_idx] < 1e-4:
                self.atoms.pop(away_idx)
                self.alphas.pop(away_idx)
