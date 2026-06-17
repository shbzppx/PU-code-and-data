"""Mineral-point detection evaluation utilities."""

from __future__ import annotations

import h5py
import numpy as np

from .analysis_utils import DETECTION_THRESHOLD
from .coordinate_utils import CoordinateConverter


class MineralEvaluator:
    """Evaluate how well model probabilities recover known mineral points."""

    def __init__(
        self,
        h5_path,
        buffer_radius=500,
        prediction_patch_size=None,
        prediction_batch_size=128,
        detection_threshold=DETECTION_THRESHOLD,
        region_radius=1,
    ):
        self.h5_path = h5_path
        self.buffer_radius = buffer_radius
        self.prediction_patch_size = prediction_patch_size
        self.prediction_batch_size = prediction_batch_size
        self.detection_threshold = float(detection_threshold)
        self.region_radius = int(region_radius)
        self.metadata = self._load_metadata()
        self.converter = CoordinateConverter(self.metadata)

    def evaluate(self, model_wrapper, mineral_points_df, threshold=None):
        if mineral_points_df is None or len(mineral_points_df) == 0:
            return None

        probability_map = self.generate_probability_map_for_wrapper(model_wrapper)
        return self.evaluate_probability_map(probability_map, mineral_points_df, threshold=threshold)

    def evaluate_probability_map(self, probability_map, mineral_points_df, threshold=None):
        if mineral_points_df is None or len(mineral_points_df) == 0:
            return None

        active_threshold = float(self.detection_threshold if threshold is None else threshold)
        results = []
        for _, row in mineral_points_df.iterrows():
            pixel_x, pixel_y = self.converter.geo_to_pixel(float(row["x"]), float(row["y"]))
            probability = self._extract_region_probability(probability_map, pixel_x, pixel_y)
            detected = probability >= active_threshold
            results.append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "probability": probability,
                    "detected": detected,
                }
            )

        detected_count = sum(1 for item in results if item["detected"])
        detection_rate = detected_count / len(results)
        avg_probability = float(np.mean([item["probability"] for item in results]))
        return {
            "detection_rate": detection_rate,
            "avg_detection_rate": detection_rate,
            "detection_ratio": detection_rate,
            "mineral_count": len(results),
            "detected_count": detected_count,
            "avg_probability": avg_probability,
            "threshold": active_threshold,
            "details": results,
        }

    def generate_probability_map_for_wrapper(self, model_wrapper):
        samples = self._load_samples()
        probabilities = model_wrapper.predict_proba(samples)
        if probabilities.ndim == 1:
            positive_scores = probabilities
        else:
            class_index = 1 if probabilities.shape[1] > 1 else 0
            positive_scores = probabilities[:, class_index]
        return self._reshape_probability_map(positive_scores)

    def generate_probability_map_for_torch_model(self, model, device, config):
        import torch

        samples = self._load_samples(config=config)
        predictions = []
        positive_class_index = 1
        normalization_stats = getattr(model, "normalization_stats", None) or config.get("normalization_stats")

        model.eval()
        with torch.no_grad():
            for start in range(0, len(samples), self.prediction_batch_size):
                batch = torch.as_tensor(
                    samples[start:start + self.prediction_batch_size],
                    dtype=torch.float32,
                ).to(device)
                if normalization_stats:
                    mean_tensor = torch.as_tensor(normalization_stats["mean"], dtype=torch.float32, device=device).view(-1, 1, 1)
                    std_tensor = torch.as_tensor(normalization_stats["std"], dtype=torch.float32, device=device).view(-1, 1, 1)
                    batch = (batch - mean_tensor) / std_tensor
                outputs = model(batch)
                probabilities = torch.softmax(outputs, dim=1)
                class_index = min(positive_class_index, probabilities.shape[1] - 1)
                predictions.extend(probabilities[:, class_index].cpu().numpy().tolist())

        return self._reshape_probability_map(predictions)

    def _load_metadata(self):
        with h5py.File(self.h5_path, "r") as handle:
            if "metadata" not in handle:
                raise KeyError("H5 文件缺少 metadata 数据集，无法进行矿点评估。")
            return dict(handle["metadata"].attrs)

    def _load_samples(self, config=None):
        from ..feature.patch_creator import PatchCreator

        config = config or {}
        with h5py.File(self.h5_path, "r") as handle:
            if "samples" in handle:
                return handle["samples"][:]
            if "fused_features" not in handle:
                raise KeyError("H5 文件既没有 samples，也没有 fused_features 数据集。")

        patch_creator = PatchCreator(self.h5_path)
        try:
            patch_size = int(
                config.get("prediction_patch_size")
                or self.prediction_patch_size
                or self.metadata.get("patch_size", 11)
            )
            samples, _ = patch_creator.generate_patches(
                patch_size,
                (1, 1),
                enable_padding=True,
            )
            return samples
        finally:
            patch_creator.source_h5_file.close()

    def _reshape_probability_map(self, positive_scores):
        expected_size = int(self.metadata["ny"]) * int(self.metadata["nx"])
        if len(positive_scores) != expected_size:
            raise ValueError(
                f"预测图尺寸不匹配: 预测样本数 {len(positive_scores)}，网格尺寸 {expected_size}。"
            )

        return np.asarray(positive_scores, dtype=np.float32).reshape(
            int(self.metadata["ny"]),
            int(self.metadata["nx"]),
        )

    def _extract_region_probability(self, predictions, pixel_x, pixel_y):
        height, width = predictions.shape
        radius = self.region_radius
        y1, y2 = max(0, pixel_y - radius), min(height, pixel_y + radius + 1)
        x1, x2 = max(0, pixel_x - radius), min(width, pixel_x + radius + 1)
        region = predictions[y1:y2, x1:x2]
        return float(region.mean()) if region.size > 0 else 0.0
