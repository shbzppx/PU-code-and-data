# -*- coding: utf-8 -*-
"""Auditable standard and adaptive nnPU losses for nnPU-CNN.

Both variants share the same stable risk estimator, global-mean positive
weight semantics, and train-only diagnostics. The adaptive variant alone
updates gamma at epoch boundaries; validation under ``torch.no_grad()`` never
mutates loss state.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

from pu_loss import AdaptivePULoss, PULoss


class _StableDiagnosticNNPUMixin:
    """Shared nnPU-CNN risk calculation and structured diagnostics."""

    algorithm_name = "nnPU-CNN"
    loss_type = "unknown"
    positive_weight_normalization = "global_positive_mean_one"

    def _initialize_diagnostics(self) -> None:
        if not hasattr(self, "base_gamma"):
            self.base_gamma = float(self.gamma)
        if not hasattr(self, "base_beta"):
            self.base_beta = float(self.beta)
        if not hasattr(self, "negative_risk_history"):
            self.negative_risk_history = []
        if not hasattr(self, "_epoch_negative_risks"):
            self._epoch_negative_risks = []
        if not hasattr(self, "epoch_counter"):
            self.epoch_counter = 0

        self.correction_history = []
        self._epoch_corrections = []
        self.epoch_diagnostics = []
        self.total_training_batch_count = 0
        self.correction_batch_count = 0
        self.gamma_history = []
        self.gamma_target_history = []

    def _positive_risk_terms(self, inp, positive, sample_weights):
        positive_terms = self.loss_func(inp[positive])
        positive_negative_terms = self.loss_func(-inp[positive])
        if sample_weights is None:
            return positive_terms.mean(), positive_negative_terms.mean()

        weights = sample_weights.view(-1).to(
            device=inp.device,
            dtype=inp.dtype,
        )[positive]
        weights = torch.where(
            torch.isfinite(weights) & (weights >= 0),
            weights,
            torch.ones_like(weights),
        )
        # Weights are normalized once over the complete fold-train positive
        # population. A batch mean preserves their global scale and the
        # relative distance decay between multiple positives in the batch.
        return (
            (positive_terms * weights).mean(),
            (positive_negative_terms * weights).mean(),
        )

    def __call__(self, inp, target, sample_weights=None):
        inp = torch.clamp(inp.view(-1), min=-50, max=50)
        target = target.view(-1)
        assert inp.shape == target.shape

        positive = target == self.positive
        unlabeled = target == self.unlabeled
        if not positive.any() or not unlabeled.any():
            return inp.sum() * 0.0

        positive_loss, positive_negative_loss = self._positive_risk_terms(
            inp,
            positive,
            sample_weights,
        )
        negative_loss = self.loss_func(-inp[unlabeled]).mean()
        positive_risk = self.prior * positive_loss
        negative_risk = negative_loss - self.prior * positive_negative_loss
        corrected = bool(float(negative_risk.detach().cpu()) < -float(self.beta))

        if torch.is_grad_enabled():
            risk_value = float(negative_risk.detach().cpu())
            self.negative_risk_history.append(risk_value)
            self._epoch_negative_risks.append(risk_value)
            self.correction_history.append(corrected)
            self._epoch_corrections.append(corrected)
            self.total_training_batch_count += 1
            if corrected:
                self.correction_batch_count += 1

        if self.nnpu and corrected:
            return positive_risk - self.beta - self.gamma * negative_risk
        if self.nnpu:
            return positive_risk + negative_risk
        return positive_risk + negative_risk

    def _finish_epoch(self, *, adapt_gamma: bool) -> float:
        risks = np.asarray(self._epoch_negative_risks, dtype=np.float64)
        corrections = np.asarray(self._epoch_corrections, dtype=bool)
        gamma_before = float(self.gamma)
        gamma_target = float(self.base_gamma)

        risk_mean = float(risks.mean()) if risks.size else None
        if adapt_gamma and risk_mean is not None:
            gamma_target = self.base_gamma * (
                1.0 + self.adaptive_lambda * max(0.0, -risk_mean)
            )
            gamma_target = float(
                np.clip(gamma_target, self.gamma_min, self.gamma_max)
            )
            self.gamma = float(
                self.gamma_ema * gamma_before
                + (1.0 - self.gamma_ema) * gamma_target
            )
            self.last_epoch_risk_mean = risk_mean
            self.last_gamma_target = gamma_target
            if self.verbose:
                print(
                    f"nnPU-CNN adaptive epoch {self.epoch_counter + 1}: "
                    f"neg_risk_mean={risk_mean:.6f}, "
                    f"correction_fraction={float(corrections.mean()) if corrections.size else 0.0:.4f}, "
                    f"gamma {gamma_before:.4f}->{self.gamma:.4f} "
                    f"(target={gamma_target:.4f})"
                )

        gamma_after = float(self.gamma)
        self.epoch_diagnostics.append(
            {
                "epoch": int(self.epoch_counter + 1),
                "training_batch_count": int(risks.size),
                "negative_risk_mean": risk_mean,
                "negative_risk_min": float(risks.min()) if risks.size else None,
                "negative_risk_max": float(risks.max()) if risks.size else None,
                "negative_fraction": float(np.mean(risks < 0.0)) if risks.size else 0.0,
                "correction_batch_count": int(corrections.sum()) if corrections.size else 0,
                "correction_fraction": (
                    float(corrections.mean()) if corrections.size else 0.0
                ),
                "gamma_before": gamma_before,
                "gamma_target": gamma_target,
                "gamma_after": gamma_after,
            }
        )
        self.gamma_target_history.append(gamma_target)
        self.gamma_history.append(gamma_after)
        self._epoch_negative_risks.clear()
        self._epoch_corrections.clear()
        self.epoch_counter += 1
        return gamma_after

    def diagnostics(self) -> dict:
        risks = np.asarray(self.negative_risk_history, dtype=np.float64)
        corrections = np.asarray(self.correction_history, dtype=bool)
        return {
            "algorithm_name": self.algorithm_name,
            "loss_type": self.loss_type,
            "configuration": {
                "prior": float(self.prior),
                "beta": float(self.beta),
                "base_gamma": float(self.base_gamma),
                "positive_weight_normalization": self.positive_weight_normalization,
                "adaptive_window": (
                    int(self.adaptive_window)
                    if hasattr(self, "adaptive_window")
                    else None
                ),
                "adaptive_lambda": (
                    float(self.adaptive_lambda)
                    if hasattr(self, "adaptive_lambda")
                    else None
                ),
                "gamma_min": (
                    float(self.gamma_min) if hasattr(self, "gamma_min") else None
                ),
                "gamma_max": (
                    float(self.gamma_max) if hasattr(self, "gamma_max") else None
                ),
                "gamma_ema": (
                    float(self.gamma_ema) if hasattr(self, "gamma_ema") else None
                ),
            },
            "summary": {
                "epochs_completed": int(self.epoch_counter),
                "total_training_batch_count": int(self.total_training_batch_count),
                "correction_batch_count": int(self.correction_batch_count),
                "correction_frequency": (
                    float(self.correction_batch_count / self.total_training_batch_count)
                    if self.total_training_batch_count
                    else 0.0
                ),
                "negative_risk_mean": float(risks.mean()) if risks.size else None,
                "negative_risk_min": float(risks.min()) if risks.size else None,
                "negative_risk_max": float(risks.max()) if risks.size else None,
                "negative_fraction": float(np.mean(risks < 0.0)) if risks.size else 0.0,
                "gamma_final": float(self.gamma),
            },
            "epochs": list(self.epoch_diagnostics),
            "trace": {
                "negative_risk": list(self.negative_risk_history),
                "corrected": [bool(value) for value in corrections.tolist()],
                "gamma": list(self.gamma_history),
                "gamma_target": list(self.gamma_target_history),
            },
        }

    def save_diagnostics(self, path: str, context: Optional[dict] = None) -> str:
        payload = self.diagnostics()
        if context:
            payload["context"] = dict(context)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path


class StableStandardPULoss(_StableDiagnosticNNPUMixin, PULoss):
    """Standard nnPU with fixed gamma and train-only diagnostics."""

    loss_type = "standard"

    def __init__(self, prior, gamma=1.0, beta=0.0, nnpu=True):
        super().__init__(prior=prior, gamma=gamma, beta=beta, nnpu=nnpu)
        self.base_gamma = float(gamma)
        self.base_beta = float(beta)
        self._initialize_diagnostics()

    def end_epoch(self):
        return self._finish_epoch(adapt_gamma=False)


class StableAdaptivePULoss(_StableDiagnosticNNPUMixin, AdaptivePULoss):
    """Adaptive nnPU with epoch-level gamma updates and diagnostics."""

    loss_type = "adaptive"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_diagnostics()

    def end_epoch(self):
        return self._finish_epoch(adapt_gamma=True)


# Aliases kept for explicit imports and older result readers.
NNPUStableStandardPULoss = StableStandardPULoss
NNPUStableAdaptivePULoss = StableAdaptivePULoss
