from __future__ import annotations

from collections.abc import Iterable

import torch


class SparseGaussianAdam(torch.optim.Optimizer):
    """LIC2's visibility-aware Adam update.

    The native optimizer updates only Gaussians with positive rasterizer
    radii, uses beta=(0.9, 0.999), and intentionally omits bias correction.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, object]],
        *,
        eps: float = 1e-15,
    ) -> None:
        if eps <= 0:
            raise ValueError("SparseGaussianAdam eps must be positive")
        super().__init__(params, {"lr": 1e-3, "betas": (0.9, 0.999), "eps": eps})
        self._visibility: torch.Tensor | None = None

    def set_visibility(self, visibility: torch.Tensor) -> None:
        if visibility.ndim != 1 or visibility.dtype != torch.bool:
            raise ValueError("SparseGaussianAdam visibility must be a boolean [N] tensor")
        self._visibility = visibility.detach()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim < 1:
                    raise ValueError("SparseGaussianAdam parameters must have a Gaussian row dimension")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                visibility = self._visibility
                if visibility is None:
                    visible = torch.ones(parameter.shape[0], dtype=torch.bool, device=parameter.device)
                else:
                    if visibility.shape != (parameter.shape[0],):
                        raise ValueError(
                            "SparseGaussianAdam visibility row count does not match parameter"
                        )
                    visible = visibility.to(device=parameter.device)
                row_mask = visible.reshape((visible.shape[0],) + (1,) * (parameter.ndim - 1))
                gradient = parameter.grad
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                next_exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
                next_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient.square()
                exp_avg.copy_(torch.where(row_mask, next_exp_avg, exp_avg))
                exp_avg_sq.copy_(torch.where(row_mask, next_exp_avg_sq, exp_avg_sq))
                update = exp_avg / (exp_avg_sq.sqrt() + eps)
                parameter.add_(torch.where(row_mask, -lr * update, torch.zeros_like(update)))
                state["step"] += 1
        return loss


__all__ = ["SparseGaussianAdam"]
