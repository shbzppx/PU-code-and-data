"""Hyperparameter optimization helper used by model comparison.

The module keeps the original Optuna-facing API, but it no longer hard-depends
on the external ``optuna`` package at import time. When Optuna is available, it
uses a real study/pruner. When it is not installed, it falls back to a built-in
random-search loop so the comparison GUI can still open and run.
"""

from __future__ import annotations

import math
import random
import time
import traceback
from dataclasses import dataclass, field

from PyQt5.QtCore import QObject, pyqtSignal

try:  # pragma: no cover - optional dependency
    import optuna  # type: ignore
    from optuna.pruners import MedianPruner  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    optuna = None
    MedianPruner = None


@dataclass
class _FallbackTrial:
    number: int
    rng: random.Random
    params: dict = field(default_factory=dict)

    def suggest_int(self, name, low, high):
        value = self.rng.randint(int(low), int(high))
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, log=False):
        low = float(low)
        high = float(high)
        if log:
            low = max(low, 1e-12)
            high = max(high, low * 1.000001)
            value = math.exp(self.rng.uniform(math.log(low), math.log(high)))
        else:
            value = self.rng.uniform(low, high)
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        choices = list(choices)
        if not choices:
            raise ValueError(f"Categorical search space for {name} is empty.")
        value = self.rng.choice(choices)
        self.params[name] = value
        return value


@dataclass
class _FallbackStudy:
    direction: str
    best_params: dict = field(default_factory=dict)
    best_value: float = field(default_factory=lambda: float("-inf"))
    trials: list = field(default_factory=list)

    def record(self, trial_record, *, is_better: bool):
        self.trials.append(trial_record)
        if is_better:
            self.best_params = dict(trial_record.get("params") or {})
            self.best_value = float(trial_record.get("value", self.best_value))


class OptunaOptimizer(QObject):
    """Optimize a model wrapper with Optuna or a built-in fallback search."""

    trial_completed = pyqtSignal(int, dict, float)
    optimization_completed = pyqtSignal(dict, float)

    def __init__(
        self,
        model_wrapper,
        train_data,
        val_data,
        metric="val_acc",
        direction="maximize",
        score_fn=None,
        train_on_trial=True,
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.train_data = train_data
        self.val_data = val_data
        self.metric = metric
        self.direction = direction
        self.score_fn = score_fn
        self.train_on_trial = bool(train_on_trial)
        self.study = None
        self._trial_records = []
        self._rng = random.Random(42)

    def _suggest_params(self, trial):
        params = {}
        for param_name, space_def in self.model_wrapper.get_param_space().items():
            space_type = space_def[0]
            if space_type == "int":
                params[param_name] = trial.suggest_int(param_name, space_def[1], space_def[2])
            elif space_type == "float":
                params[param_name] = trial.suggest_float(param_name, space_def[1], space_def[2])
            elif space_type == "loguniform":
                params[param_name] = trial.suggest_float(param_name, space_def[1], space_def[2], log=True)
            elif space_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, space_def[1])
        return params

    def objective(self, trial):
        params = self._suggest_params(trial)
        try:
            history = {}
            default_score = 0.0
            if self.train_on_trial:
                history = self.model_wrapper.train(self.train_data, self.val_data, **params)
                default_score = float(history.get(self.metric, 0.0))
            evaluation = {}
            if self.score_fn is not None:
                try:
                    evaluation = dict(self.score_fn(self.model_wrapper, params, history, trial=trial) or {})
                except TypeError:
                    evaluation = dict(self.score_fn(self.model_wrapper, params, history) or {})

            score = float(evaluation.get("score", default_score))
            metrics = dict(evaluation.get("metrics", {}))
            trial_record = {
                "trial": trial.number,
                "params": params,
                "value": score,
            }
            trial_record.update(metrics)
            self._trial_records.append(trial_record)
            self.trial_completed.emit(trial.number, params, score)
            return score
        except Exception as exc:
            failure_value = 0.0 if self.direction == "maximize" else float("inf")
            error_traceback = traceback.format_exc()
            self._trial_records.append(
                {
                    "trial": trial.number,
                    "params": params,
                    "value": failure_value,
                    "status": "failed",
                    "error": str(exc),
                    "error_traceback": error_traceback,
                }
            )
            print(f"Trial {trial.number} failed: {exc}")
            print(error_traceback)
            return failure_value

    def optimize(self, n_trials=50, timeout=None):
        self._trial_records = []
        if optuna is not None:
            self.study = optuna.create_study(
                direction=self.direction,
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3) if MedianPruner is not None else None,
            )
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
            )

            best_params = dict(self.study.best_params)
            best_score = float(self.study.best_value)
            self.optimization_completed.emit(best_params, best_score)
            return best_params, best_score

        self.study = _FallbackStudy(direction=self.direction)
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        total_trials = max(int(n_trials or 0), 0)
        best_record = None
        for trial_number in range(total_trials):
            if deadline is not None and time.monotonic() >= deadline:
                break

            trial = _FallbackTrial(trial_number, self._rng)
            score = self.objective(trial)
            trial_record = self._trial_records[-1]
            self.study.record(
                trial_record,
                is_better=self._is_better_trial(trial_record, best_record),
            )
            if trial_record.get("status") != "failed":
                best_record = trial_record if self._is_better_score(score, best_record) else best_record

        if best_record is None and self._trial_records:
            best_record = self._trial_records[0]

        best_params = dict((best_record or {}).get("params") or {})
        best_score = float((best_record or {}).get("value", 0.0))
        self.optimization_completed.emit(best_params, best_score)
        return best_params, best_score

    def get_optimization_history(self):
        return list(self._trial_records)

    def _is_better_score(self, score, best_record):
        if best_record is None:
            return True
        best_value = float(best_record.get("value", 0.0))
        if self.direction == "maximize":
            return float(score) > best_value
        return float(score) < best_value

    def _is_better_trial(self, trial_record, best_record):
        if trial_record.get("status") == "failed":
            return best_record is None
        if best_record is None or best_record.get("status") == "failed":
            return True
        return self._is_better_score(trial_record.get("value", 0.0), best_record)
