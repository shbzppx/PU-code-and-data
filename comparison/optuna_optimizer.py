"""Optuna-based hyperparameter optimization."""

from __future__ import annotations

import optuna
from optuna.pruners import MedianPruner
from PyQt5.QtCore import QObject, pyqtSignal


class OptunaOptimizer(QObject):
    """Optimize a model wrapper with Optuna."""

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
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.train_data = train_data
        self.val_data = val_data
        self.metric = metric
        self.direction = direction
        self.score_fn = score_fn
        self.study = None
        self._trial_records = []

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
            history = self.model_wrapper.train(self.train_data, self.val_data, **params)
            default_score = float(history.get(self.metric, 0.0))
            evaluation = {}
            if self.score_fn is not None:
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
            self._trial_records.append(
                {
                    "trial": trial.number,
                    "params": params,
                    "value": failure_value,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"Trial {trial.number} failed: {exc}")
            return failure_value

    def optimize(self, n_trials=50, timeout=None):
        self._trial_records = []
        self.study = optuna.create_study(
            direction=self.direction,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        best_params = self.study.best_params
        best_score = self.study.best_value
        self.optimization_completed.emit(best_params, best_score)
        return best_params, best_score

    def get_optimization_history(self):
        return list(self._trial_records)
