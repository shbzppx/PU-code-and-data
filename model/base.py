import torch
import torch.nn as nn

class MyClassifier(nn.Module):
    def __init__(self, prior=0):
        super(MyClassifier, self).__init__()
        self.prior = prior
        self.loss = None

    def forward(self, x):
        raise NotImplementedError

    def compute_loss(self, x, t, loss_func):
        self.loss = None
        h = self.forward(x)
        self.loss = loss_func(h.view(-1), t)
        return self.loss

    def error(self, x, t):
        with torch.no_grad():
            h = self.forward(x)
            h = (h.view(-1) > 0).float() * 2 - 1
            if isinstance(t, torch.Tensor):
                t = t.view(-1)
            size = len(t)
            result = (h != t).sum().float() / size
            return min(result.item(), 1.0)

    def compute_prediction_summary(self, x, t):
        with torch.no_grad():
            h = torch.sign(self.forward(x)).view(-1)
            if isinstance(t, torch.Tensor):
                t = t.data
            n_p = (t == 1).sum()
            n_n = (t == -1).sum()
            t_p = ((h == 1) & (t == 1)).sum()
            t_n = ((h == -1) & (t == -1)).sum()
            f_p = n_n - t_n
            f_n = n_p - t_p
        return int(t_p), int(t_n), int(f_p), int(f_n) 