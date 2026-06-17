"""Positive-unlabeled loss helpers used by the system model wrappers."""

from __future__ import annotations

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - 运行环境可能未安装 torch
    class _TorchFallback:
        @staticmethod
        def sigmoid(x):
            return x

        @staticmethod
        def clamp(x, *args, **kwargs):
            return x

        @staticmethod
        def tensor(*args, **kwargs):
            return 0.0

    torch = _TorchFallback()


class PULoss:
    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), nnpu=True, gamma=1, beta=0):
        self.prior = float(prior)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.loss_func = loss
        self.nnpu = bool(nnpu)
        self.positive = 1
        self.unlabeled = -1

    def __call__(self, inp, target):
        inp = inp.view(-1)
        target = target.view(-1)
        if inp.shape != target.shape:
            raise ValueError("PU loss expects input and target to have the same shape.")

        inp = torch.clamp(inp, min=-50, max=50)
        positive = target == self.positive
        unlabeled = target == self.unlabeled

        if not positive.any() or not unlabeled.any():
            return torch.tensor(0.0, requires_grad=True, device=inp.device)

        positive_loss = self.loss_func(inp[positive]).mean()
        positive_negative_loss = self.loss_func(-inp[positive]).mean()
        negative_loss = self.loss_func(-inp[unlabeled]).mean()

        if self.nnpu:
            positive_risk = self.prior * positive_loss
            negative_risk = negative_loss - self.prior * positive_negative_loss
            if negative_risk < -self.beta:
                return positive_risk - self.beta - self.gamma * negative_risk
            return positive_risk + negative_risk

        return self.prior * positive_loss + negative_loss - self.prior * positive_negative_loss


def pu_loss(x, t, prior, loss=(lambda x: torch.sigmoid(-x)), nnpu=True):
    """Compatibility helper matching the original PU loss API."""

    return PULoss(prior=prior, loss=loss, nnpu=nnpu)(x, t)


class AdaptivePULoss:
    def __init__(
        self,
        prior,
        loss=(lambda x: torch.sigmoid(-x)),
        gamma=1,
        beta=0,
        adaptive_window=10,
        nnpu=True,
        adaptive_lambda=1.0,
        gamma_min=None,
        gamma_max=None,
    ):
        self.prior = float(prior)
        self.base_gamma = float(gamma)
        self.base_beta = float(beta)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.adaptive_lambda = float(adaptive_lambda)
        self.gamma_min = float(gamma_min) if gamma_min is not None else self.base_gamma
        self.gamma_max = float(gamma_max) if gamma_max is not None else max(self.base_gamma, 20.0)
        if self.gamma_max < self.gamma_min:
            self.gamma_max = self.gamma_min
        self.loss_func = loss
        self.positive = 1
        self.unlabeled = -1
        self.nnpu = bool(nnpu)
        self.adaptive_window = int(adaptive_window)
        self.negative_risk_history = []
        self.epoch_counter = 0

    def adaptive_parameters(self):
        if len(self.negative_risk_history) >= self.adaptive_window:
            risk_trend = np.mean(self.negative_risk_history[-self.adaptive_window:])
            raw_gamma = self.base_gamma * (1 + self.adaptive_lambda * max(0.0, -risk_trend))
            self.gamma = float(np.clip(raw_gamma, self.gamma_min, self.gamma_max))
            self.beta = self.base_beta

    def __call__(self, inp, target):
        inp = inp.view(-1)
        target = target.view(-1)
        if inp.shape != target.shape:
            raise ValueError("Adaptive PU loss expects input and target to have the same shape.")

        inp = torch.clamp(inp, min=-50, max=50)
        positive = target == self.positive
        unlabeled = target == self.unlabeled

        if not positive.any() or not unlabeled.any():
            return torch.tensor(0.0, requires_grad=True, device=inp.device)

        positive_loss = self.loss_func(inp[positive]).mean()
        positive_negative_loss = self.loss_func(-inp[positive]).mean()
        negative_loss = self.loss_func(-inp[unlabeled]).mean()
        positive_risk = self.prior * positive_loss
        negative_risk = negative_loss - self.prior * positive_negative_loss

        self.negative_risk_history.append(float(negative_risk.detach().cpu().item()))
        self.adaptive_parameters()

        if self.nnpu:
            if negative_risk < -self.beta:
                return positive_risk - self.beta - self.gamma * negative_risk
            return positive_risk + negative_risk

        return self.prior * positive_loss + negative_loss - self.prior * positive_negative_loss
