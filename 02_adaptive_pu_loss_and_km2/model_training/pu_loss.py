import torch
import numpy as np


class PULoss:
    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), nnpu=True, gamma=1, beta=0):
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.nnpu = nnpu
        self.positive = 1
        self.unlabeled = -1

    def __call__(self, inp, target, sample_weights=None):
        # 确保输入张量的形状匹配
        inp = inp.view(-1)  # 展平为一维
        target = target.view(-1)  # 展平为一维
        assert(inp.shape == target.shape)
        
        # 添加数值稳定性
        inp = torch.clamp(inp, min=-50, max=50)  # 防止数值溢出
        
        positive = (target == self.positive)
        unlabeled = (target == self.unlabeled)
        
        # 确保有正样本和未标记样本
        if not positive.any() or not unlabeled.any():
            return torch.tensor(0.0, requires_grad=True, device=inp.device)

        if sample_weights is None:
            pos_w = None
        else:
            sample_weights = sample_weights.view(-1).to(inp.device)
            pos_w = sample_weights[positive]
            pos_w = pos_w / (pos_w.sum() + 1e-12)

        if pos_w is None:
            positive_loss = self.loss_func(inp[positive]).mean()
            positive_negative_loss = self.loss_func(-inp[positive]).mean()
        else:
            positive_loss = (self.loss_func(inp[positive]) * pos_w).sum()
            positive_negative_loss = (self.loss_func(-inp[positive]) * pos_w).sum()
        negative_loss = self.loss_func(-inp[unlabeled]).mean()
        
        if self.nnpu:
            positive_risk = self.prior * positive_loss
            negative_risk = negative_loss - self.prior * positive_negative_loss
            
            if negative_risk < -self.beta:
                return positive_risk - self.beta - self.gamma * negative_risk
            else:
                return positive_risk + negative_risk
        else:
            return self.prior * positive_loss + negative_loss - self.prior * positive_negative_loss


def pu_loss(x, t, prior, loss=(lambda x: torch.sigmoid(-x)), nnpu=True):
    """wrapper of loss function for non-negative/unbiased PU learning

        .. math::
            \\begin{array}{lc}
            L_[\\pi E_1[l(f(x))]+\\max(E_X[l(-f(x))]-\\pi E_1[l(-f(x))], \\beta) & {\\rm if nnPU learning}\\\\
            L_[\\pi E_1[l(f(x))]+E_X[l(-f(x))]-\\pi E_1[l(-f(x))] & {\\rm otherwise}
            \\end{array}

    Args:
        x (~chainer.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, 1).
        t (~chainer.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, ).
        prior (float): Constant variable for class prior.
        loss (~chainer.function): loss function.
            The loss function should be non-increasing.
        nnpu (bool): Whether use non-negative PU learning or unbiased PU learning.
            In default setting, non-negative PU learning will be used.

    Returns:
        ~chainer.Variable: A variable object holding a scalar array of the
            PU loss.

    See:
        Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama.
        "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
        Advances in neural information processing systems. 2017.
        du Plessis, Marthinus Christoffel, Gang Niu, and Masashi Sugiyama.
        "Convex formulation for learning from positive and unlabeled data."
        Proceedings of The 30th International Conference on Machine Learning. 2015.
    """
    return PULoss(prior=prior, loss=loss, nnpu=nnpu)(x, t)


class AdaptivePULoss:
    """nnPU loss with epoch-level EMA gamma adaptation (not per-batch)."""

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
        gamma_ema=0.8,
        verbose=True,
    ):
        self.prior = prior
        self.base_gamma = float(gamma)
        self.base_beta = beta
        self.gamma = float(gamma)
        self.beta = beta
        self.adaptive_lambda = float(adaptive_lambda)
        self.gamma_min = float(gamma_min) if gamma_min is not None else float(gamma)
        # Tighten default ceiling: previously max(gamma, 20) was too aggressive.
        self.gamma_max = (
            float(gamma_max) if gamma_max is not None else max(float(gamma), 5.0)
        )
        if self.gamma_max < self.gamma_min:
            self.gamma_max = self.gamma_min
        self.gamma_ema = float(np.clip(float(gamma_ema), 0.0, 0.999))
        self.verbose = bool(verbose)
        self.loss_func = loss
        self.positive = 1
        self.unlabeled = -1
        self.nnpu = nnpu

        # Kept for diagnostics / RN wrappers; adaptation uses epoch means + EMA.
        self.adaptive_window = max(1, int(adaptive_window))
        self.negative_risk_history = []
        self._epoch_negative_risks = []
        self.epoch_counter = 0
        self.last_epoch_risk_mean = None
        self.last_gamma_target = None

    def snapshot_state(self):
        return {
            "negative_risk_history": list(self.negative_risk_history),
            "epoch_negative_risks": list(self._epoch_negative_risks),
            "gamma": float(self.gamma),
            "beta": float(self.beta),
            "epoch_counter": int(self.epoch_counter),
            "last_epoch_risk_mean": self.last_epoch_risk_mean,
            "last_gamma_target": self.last_gamma_target,
        }

    def restore_state(self, state):
        self.negative_risk_history[:] = list(state.get("negative_risk_history") or [])
        self._epoch_negative_risks[:] = list(state.get("epoch_negative_risks") or [])
        self.gamma = float(state.get("gamma", self.gamma))
        self.beta = float(state.get("beta", self.beta))
        self.epoch_counter = int(state.get("epoch_counter", self.epoch_counter))
        self.last_epoch_risk_mean = state.get("last_epoch_risk_mean")
        self.last_gamma_target = state.get("last_gamma_target")

    def end_epoch(self):
        """Update gamma once per epoch from the mean batch negative risk."""
        if self._epoch_negative_risks:
            risk_mean = float(np.mean(self._epoch_negative_risks))
            self.last_epoch_risk_mean = risk_mean
            raw_gamma = self.base_gamma * (
                1.0 + self.adaptive_lambda * max(0.0, -risk_mean)
            )
            raw_gamma = float(np.clip(raw_gamma, self.gamma_min, self.gamma_max))
            self.last_gamma_target = raw_gamma
            prev = float(self.gamma)
            self.gamma = float(
                self.gamma_ema * prev + (1.0 - self.gamma_ema) * raw_gamma
            )
            if self.verbose:
                print(
                    f"AdaptivePU epoch {self.epoch_counter + 1}: "
                    f"neg_risk_mean={risk_mean:.4f}, "
                    f"gamma {prev:.4f}→{self.gamma:.4f} (target={raw_gamma:.4f})"
                )
        self._epoch_negative_risks.clear()
        self.epoch_counter += 1
        return float(self.gamma)

    def __call__(self, inp, target, sample_weights=None):
        # 确保输入张量的形状匹配
        inp = inp.view(-1)  # 展平为一维
        target = target.view(-1)  # 展平为一维
        assert(inp.shape == target.shape)
        
        # 添加数值稳定性
        inp = torch.clamp(inp, min=-50, max=50)  # 防止数值溢出
        
        positive = (target == self.positive)
        unlabeled = (target == self.unlabeled)
        
        # 确保有正样本和未标记样本
        if not positive.any() or not unlabeled.any():
            return torch.tensor(0.0, requires_grad=True, device=inp.device)

        if sample_weights is None:
            pos_w = None
        else:
            sample_weights = sample_weights.view(-1).to(inp.device)
            pos_w = sample_weights[positive]
            pos_w = pos_w / (pos_w.sum() + 1e-12)

        if pos_w is None:
            positive_loss = self.loss_func(inp[positive]).mean()
            positive_negative_loss = self.loss_func(-inp[positive]).mean()
        else:
            positive_loss = (self.loss_func(inp[positive]) * pos_w).sum()
            positive_negative_loss = (self.loss_func(-inp[positive]) * pos_w).sum()
        negative_loss = self.loss_func(-inp[unlabeled]).mean()
        
        # 计算风险
        positive_risk = self.prior * positive_loss
        negative_risk = negative_loss - self.prior * positive_negative_loss
        
        risk_value = float(negative_risk.detach().cpu().numpy())
        self.negative_risk_history.append(risk_value)
        self._epoch_negative_risks.append(risk_value)
        
        # Gamma is held fixed within the epoch; end_epoch() adapts it.
        if self.nnpu:
            if negative_risk < -self.beta:
                return positive_risk - self.beta - self.gamma * negative_risk
            else:
                return positive_risk + negative_risk
        else:
            return self.prior * positive_loss + negative_loss - self.prior * positive_negative_loss
