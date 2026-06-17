from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn


class Trainer:
    """Minimal PyTorch trainer used by the compatibility bridge."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        epochs=20,
        lr=0.001,
        device="cpu",
        image_size=None,
        save_path=None,
        optimizer_name="AdamW",
        scheduler_enabled=False,
        scheduler_type="CosineAnnealingLR",
        lr_step_size=10,
        lr_gamma=0.1,
        early_stopping_patience=15,
    ):
        del image_size
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.save_path = self._resolve_checkpoint_path(save_path)
        self.optimizer_name = str(optimizer_name or "AdamW")
        self.scheduler_enabled = bool(scheduler_enabled)
        self.scheduler_type = str(scheduler_type or "CosineAnnealingLR")
        self.lr_step_size = int(lr_step_size)
        self.lr_gamma = float(lr_gamma)
        self.early_stopping_patience = int(early_stopping_patience)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self._stop_requested = False
        self.best_state_dict = None
        self.best_metric = float("-inf")
        self.best_epoch = -1

    def _resolve_checkpoint_path(self, save_path):
        if not save_path:
            return None
        path = Path(save_path)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{self.model.__class__.__name__}_best.pt"

    def stop(self):
        self._stop_requested = True

    def _build_optimizer(self):
        name = self.optimizer_name.lower()
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        if name == "rmsprop":
            return torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def _build_scheduler(self, optimizer):
        if not self.scheduler_enabled:
            return None
        name = self.scheduler_type.lower()
        if name == "steplr":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(self.lr_step_size, 1),
                gamma=self.lr_gamma,
            )
        if name == "reducelronplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.lr_gamma,
                patience=3,
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.epochs, 1))

    @staticmethod
    def _batch_accuracy(logits, labels):
        predictions = torch.argmax(logits, dim=1)
        return float((predictions == labels).float().mean().item())

    def _run_epoch(self, loader, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        criterion = nn.CrossEntropyLoss()
        loss_total = 0.0
        accuracy_total = 0.0
        batch_count = 0

        for batch in loader:
            if self._stop_requested:
                break
            inputs, labels = batch
            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long).view(-1)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = self.model(inputs)
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(labels.shape[0])
            loss_total += float(loss.item()) * batch_size
            accuracy_total += self._batch_accuracy(outputs.detach(), labels) * batch_size
            batch_count += batch_size

        if batch_count == 0:
            return 0.0, 0.0
        return loss_total / batch_count, accuracy_total / batch_count

    def _evaluate(self, loader):
        if loader is None:
            return None, None
        with torch.no_grad():
            return self._run_epoch(loader, optimizer=None)

    def train(self):
        self.model.to(self.device)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        patience_counter = 0

        for epoch in range(self.epochs):
            if self._stop_requested:
                break

            train_loss, train_acc = self._run_epoch(self.train_loader, optimizer=optimizer)
            val_loss, val_acc = self._evaluate(self.val_loader)
            if val_acc is None:
                val_loss, val_acc = train_loss, train_acc

            self.history["train_loss"].append(float(train_loss))
            self.history["train_acc"].append(float(train_acc))
            self.history["val_loss"].append(float(val_loss))
            self.history["val_acc"].append(float(val_acc))

            monitor_value = float(val_acc if self.val_loader is not None else train_acc)
            if monitor_value > self.best_metric + 1e-6:
                self.best_metric = monitor_value
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if self.save_path is not None:
                    torch.save(
                        {
                            "model_state_dict": self.best_state_dict,
                            "history": self.history,
                            "best_metric": self.best_metric,
                            "best_epoch": self.best_epoch,
                        },
                        self.save_path,
                    )
            else:
                patience_counter += 1

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor_value)
                else:
                    scheduler.step()

            if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        return self.history
