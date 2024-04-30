import torch
from typing import Callable
import numpy as np

def grid_line_search(
    X: torch.Tensor,
    direction: torch.Tensor,
    loss_fn: Callable,
    max_step_size: float = 1,
    num_steps: int = 1000,
):
    steps = torch.linspace(0, max_step_size, num_steps, dtype=X.dtype, device=X.device)
    losses = [loss_fn(X + steps[step] * direction) for step in range(num_steps)]
    losses = torch.Tensor(losses)
    return steps[losses.argmin()].item()


def golden_line_search(
    X: torch.Tensor,
    direction: torch.Tensor,
    loss_fn: Callable,
    max_gamma: float = 1,
    tol=1e-6
):
    golden_ratio = (np.sqrt(5) + 1) / 2
    left, right = 0, max_gamma

    while abs(right - left) > tol:
        first = right - (right - left)/golden_ratio
        second = left + (right - left)/golden_ratio

        # Look at left interval
        if loss_fn(X + first*direction) < loss_fn(X + second*direction):
            right = second
        # Look at right interval
        else:
            left = first

    # Midpoint of interval
    return (right + left) / 2
