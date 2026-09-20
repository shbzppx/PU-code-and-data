"""Positive/unlabeled batch construction for PU neural models."""

from __future__ import annotations

import math
from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class RNPositiveUnlabeledBatchSampler(Sampler[List[int]]):
    """Yield fixed-size batches containing positive and unlabeled examples.

    ``positives_per_batch`` is independent of ``batch_size``. The sampler uses
    enough batches to cover every positive at least once and to reach the
    requested unique-unlabeled coverage. Scarce positives are then cycled;
    unlabeled examples are selected without replacement up to the coverage
    target and cycled only to fill the remaining fixed-size batch slots.

    The defaults retain the original RN behavior: one positive per batch and
    one optimizer step per positive. nnPU-CNN passes explicit multi-positive
    and unlabeled-coverage settings through the training loader.
    """

    def __init__(
        self,
        labels: Sequence[float],
        batch_size: int,
        seed: int = 0,
        positives_per_batch: int = 1,
        unlabeled_coverage: float | None = None,
    ) -> None:
        labels_array = np.asarray(labels).reshape(-1)
        self.positive_indices = np.flatnonzero(labels_array == 1).astype(np.int64)
        self.unlabeled_indices = np.flatnonzero(labels_array == -1).astype(np.int64)
        if not len(self.positive_indices) or not len(self.unlabeled_indices):
            raise ValueError("PU batches require both positive and unlabeled samples.")
        self.batch_size = max(2, int(batch_size))
        self.positives_per_batch = int(positives_per_batch)
        if self.positives_per_batch < 1:
            raise ValueError("positives_per_batch must be at least 1.")
        if self.positives_per_batch >= self.batch_size:
            raise ValueError(
                "positives_per_batch must be smaller than batch_size so every "
                "PU batch also contains unlabeled examples."
            )
        self.unlabeled_per_batch = self.batch_size - self.positives_per_batch

        batches_for_positive_coverage = int(
            math.ceil(len(self.positive_indices) / self.positives_per_batch)
        )
        if unlabeled_coverage is None:
            # Original RN protocol: one pass over positives determines the epoch.
            self.unlabeled_coverage_target = None
            target_unlabeled_count = min(
                len(self.unlabeled_indices),
                batches_for_positive_coverage * self.unlabeled_per_batch,
            )
            batches_for_unlabeled_coverage = 0
        else:
            coverage = float(unlabeled_coverage)
            if not np.isfinite(coverage) or not 0.0 < coverage <= 1.0:
                raise ValueError("unlabeled_coverage must be in the interval (0, 1].")
            self.unlabeled_coverage_target = coverage
            target_unlabeled_count = max(
                1,
                int(math.ceil(len(self.unlabeled_indices) * coverage)),
            )
            batches_for_unlabeled_coverage = int(
                math.ceil(target_unlabeled_count / self.unlabeled_per_batch)
            )

        self.num_batches = max(
            1,
            batches_for_positive_coverage,
            batches_for_unlabeled_coverage,
        )
        self.target_unlabeled_count = int(target_unlabeled_count)
        self.seed = int(seed)
        self.pass_index = 0

    @staticmethod
    def _cycled_draw(
        indices: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        chunks = []
        remaining = int(count)
        while remaining > 0:
            shuffled = rng.permutation(indices)
            take = min(remaining, len(shuffled))
            chunks.append(shuffled[:take])
            remaining -= take
        return np.concatenate(chunks).astype(np.int64, copy=False)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.pass_index)
        self.pass_index += 1
        positives = self._cycled_draw(
            self.positive_indices,
            self.num_batches * self.positives_per_batch,
            rng,
        )
        unlabeled_subset = rng.permutation(self.unlabeled_indices)[
            : self.target_unlabeled_count
        ]
        unlabeled = self._cycled_draw(
            unlabeled_subset,
            self.num_batches * self.unlabeled_per_batch,
            rng,
        )

        for batch_index in range(self.num_batches):
            positive_start = batch_index * self.positives_per_batch
            unlabeled_start = batch_index * self.unlabeled_per_batch
            batch = np.concatenate(
                [
                    positives[
                        positive_start : positive_start + self.positives_per_batch
                    ],
                    unlabeled[
                        unlabeled_start : unlabeled_start + self.unlabeled_per_batch
                    ],
                ]
            )
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches

    def diagnostics(self) -> dict:
        positive_draws = self.num_batches * self.positives_per_batch
        unlabeled_draws = self.num_batches * self.unlabeled_per_batch
        return {
            "sampling_protocol": (
                "positive_and_target_unlabeled_coverage"
                if self.unlabeled_coverage_target is not None
                else "legacy_positive_coverage"
            ),
            "positive_count": int(len(self.positive_indices)),
            "unlabeled_count": int(len(self.unlabeled_indices)),
            "batch_size": int(self.batch_size),
            "positives_per_batch": int(self.positives_per_batch),
            "unlabeled_per_batch": int(self.unlabeled_per_batch),
            "batches_per_pass": int(self.num_batches),
            "positive_coverage_rate": 1.0,
            "positive_draw_count": int(positive_draws),
            "positive_repeat_draw_count": int(
                max(0, positive_draws - len(self.positive_indices))
            ),
            "unlabeled_coverage_target": self.unlabeled_coverage_target,
            "unlabeled_unique_count_per_pass": int(self.target_unlabeled_count),
            "unlabeled_coverage_rate": float(
                self.target_unlabeled_count / len(self.unlabeled_indices)
            ),
            "unlabeled_draw_count": int(unlabeled_draws),
            "unlabeled_repeat_draw_count": int(
                max(0, unlabeled_draws - self.target_unlabeled_count)
            ),
            "passes_iterated": int(self.pass_index),
        }


# Neutral alias for new callers; the original name remains for compatibility.
PositiveUnlabeledBatchSampler = RNPositiveUnlabeledBatchSampler
