import torch
from typing import Callable


def grid_line_search(
    X: torch.Tensor,
    direction: torch.Tensor,
    loss_fn: Callable,
    max_step_size: float = 1,
    batch_size: int = 1000,
):
    X = X.detach().clone()
    num_dims = [1] * (X.ndim - 1)
    steps = torch.linspace(0, max_step_size, batch_size, dtype=X.dtype).to(X.device)
    steps = steps.view(batch_size, *num_dims)

    batch_X = X.repeat(batch_size, *num_dims)
    batch_dir = direction.repeat(batch_size, *num_dims)
    batch = batch_X + steps * batch_dir

    batch_loss = loss_fn(batch)
    return steps[batch_loss.argmin()].item()


def batch_golden_line_search(
    X: torch.Tensor,
    direction: torch.Tensor,
    loss_fn: Callable,
    max_gamma: float = 1,
    batch_size: int = 32,
):
    X = X.detach().clone()
    curr_min_gamma, curr_max_gamma = 0, max_gamma
    prev_min, curr_min = None, float("inf")

    num_dims = [1] * (X.ndim - 1)
    batch_X = X.repeat(batch_size, *num_dims)
    batch_dir = direction.repeat(batch_size, *num_dims)

    while prev_min is None or prev_min - curr_min > 1e-8:
        steps = torch.linspace(
            curr_min_gamma, curr_max_gamma, batch_size, dtype=X.dtype
        ).to(X.device)
        steps = steps.view(batch_size, *num_dims)

        batch = batch_X + steps * batch_dir
        batch_loss = loss_fn(batch)

        prev_min = curr_min
        argmin, curr_min = batch_loss.argmin(), batch_loss.min()

        curr_min_gamma = steps[max(argmin - 1, 0)].item()
        curr_max_gamma = steps[min(argmin + 1, batch_size - 1)].item()

    argmin = max(argmin, 1)
    return steps[argmin].item()
