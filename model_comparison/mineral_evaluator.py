"""Mineral-point detection evaluation utilities."""

from __future__ import annotations

import h5py
import numpy as np

from .analysis_utils import DETECTION_THRESHOLD
from .coordinate_utils import CoordinateConverter, infer_grid_metadata_from_coordinates
from .metric_protocol import THRESHOLD_RULE


class MineralEvaluator:
    """Evaluate how well model probabilities recover known mineral points."""

    def __init__(
        self,
        h5_path,
        buffer_radius=500,
        prediction_patch_size=None,
        prediction_patch_stride=None,
        prediction_batch_size=128,
        detection_threshold=DETECTION_THRESHOLD,
        region_radius=1,
    ):
        self.h5_path = h5_path
        self.buffer_radius = buffer_radius
        self.prediction_patch_size = prediction_patch_size
        self.prediction_patch_stride = prediction_patch_stride
        self.prediction_batch_size = prediction_batch_size
        self.detection_threshold = float(detection_threshold)
        self.region_radius = int(region_radius)
        self.coordinates = self._load_coordinates()
        self.metadata = self._load_metadata(self.coordinates)
        self.converter = CoordinateConverter(self.metadata)
        self.last_prediction_positions = None

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
            detected = probability > active_threshold
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
            "threshold_rule": THRESHOLD_RULE,
            "details": results,
        }

    def generate_probability_map_for_wrapper(self, model_wrapper):
        samples, sample_positions = self._load_samples_with_positions()
        probabilities = model_wrapper.predict_proba(samples)
        if probabilities.ndim == 1:
            positive_scores = probabilities
        else:
            class_index = 1 if probabilities.shape[1] > 1 else 0
            positive_scores = probabilities[:, class_index]
        return self._scores_to_probability_map(positive_scores, sample_positions=sample_positions)

    def generate_probability_map_for_torch_model(self, model, device, config):
        import torch

        samples, sample_positions = self._load_samples_with_positions(config=config)
        predictions = []
        positive_class_index = 1
        normalization_stats = getattr(model, "normalization_stats", None) or config.get("normalization_stats")

        model.eval()
        with torch.no_grad():
            for start in range(0, len(samples), self.prediction_batch_size):
                batch = torch.as_tensor(
                    samples[start : start + self.prediction_batch_size],
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

        return self._scores_to_probability_map(predictions, sample_positions=sample_positions)

    def _load_coordinates(self):
        with h5py.File(self.h5_path, "r") as handle:
            if "coordinates" in handle:
                return np.asarray(handle["coordinates"][:], dtype=np.float64)
        return None

    def _load_metadata(self, coordinates=None):
        with h5py.File(self.h5_path, "r") as handle:
            if "metadata" in handle:
                return dict(handle["metadata"].attrs)
            if "coordinates" in handle:
                inferred = infer_grid_metadata_from_coordinates(handle["coordinates"][:])
                if inferred:
                    return inferred
        raise KeyError("H5 file is missing metadata or coordinates required for mineral evaluation.")

    def _load_samples(self, config=None):
        samples, _ = self._load_samples_with_positions(config=config)
        return samples

    def _load_samples_with_positions(self, config=None):
        from ..feature.patch_creator import PatchCreator
        from ..cnn.data_loader import _patch_indices_to_geo

        config = config or {}
        with h5py.File(self.h5_path, "r") as handle:
            if "samples" in handle:
                samples = np.asarray(handle["samples"][:], dtype=np.float32)
                if samples.ndim == 2:
                    samples = samples[:, :, None, None]
                elif samples.ndim == 3:
                    samples = samples[:, None, :, :]
                positions = None
                if "coordinates" in handle:
                    coords = np.asarray(handle["coordinates"][:], dtype=np.float64)
                    if len(coords) == len(samples):
                        positions = _patch_indices_to_geo(coords, self.metadata)
                return samples, positions

            if "vectors" in handle and "coordinates" in handle:
                vectors = np.asarray(handle["vectors"][:], dtype=np.float32)
                patch_size = int(
                    config.get("prediction_patch_size")
                    or self.prediction_patch_size
                    or self.metadata.get("patch_size", 11)
                )
                patch_stride = int(
                    config.get("prediction_patch_stride")
                    or config.get("patch_stride")
                    or self.prediction_patch_stride
                    or self.metadata.get("patch_stride", 1)
                    or 1
                )
                reflect_padding = bool(
                    config.get(
                        "reflect_padding",
                        config.get(
                            "use_reflect_padding",
                            self.metadata.get("reflect_padding", self.metadata.get("prediction_reflect_padding", True)),
                        ),
                    )
                )
                if patch_size > 1:
                    patch_creator = PatchCreator(self.h5_path)
                    try:
                        samples, coords = patch_creator.generate_patches(
                            patch_size,
                            (patch_stride, patch_stride),
                            enable_padding=reflect_padding,
                            padding_mode="reflect",
                        )
                        position_metadata = dict(self.metadata)
                        position_metadata["window_width"] = int(patch_size)
                        position_metadata["window_height"] = int(patch_size)
                        position_metadata["patch_stride"] = int(patch_stride)
                        position_metadata["coordinates_are_centers"] = bool(reflect_padding)
                        position_metadata["coordinate_order"] = "row_col"
                        return samples, _patch_indices_to_geo(coords, position_metadata)
                    finally:
                        patch_creator.close()
                if vectors.ndim == 1:
                    vectors = vectors.reshape(-1, 1)
                coords = np.asarray(handle["coordinates"][:], dtype=np.float64)
                positions = _patch_indices_to_geo(coords, self.metadata) if len(coords) == len(vectors) else None
                if vectors.ndim == 2:
                    return vectors[:, :, None, None], positions
                if vectors.ndim == 3:
                    return vectors[:, None, :, :], positions
                if vectors.ndim == 4:
                    return vectors, positions
                raise ValueError(f"Unsupported vectors shape for H5 file: {vectors.shape}")

            if "fused_features" not in handle:
                raise KeyError("H5 file does not contain samples, vectors, or fused_features.")

        patch_creator = PatchCreator(self.h5_path)
        try:
            patch_size = int(
                config.get("prediction_patch_size")
                or self.prediction_patch_size
                or self.metadata.get("patch_size", 11)
            )
            patch_stride = int(
                config.get("prediction_patch_stride")
                or config.get("patch_stride")
                or self.prediction_patch_stride
                or self.metadata.get("patch_stride", patch_size)
                or patch_size
            )
            reflect_padding = bool(
                config.get(
                    "reflect_padding",
                    config.get(
                        "use_reflect_padding",
                        self.metadata.get("reflect_padding", self.metadata.get("prediction_reflect_padding", False)),
                    ),
                )
            )
            samples, coords = patch_creator.generate_patches(
                patch_size,
                (patch_stride, patch_stride),
                enable_padding=reflect_padding,
                padding_mode="reflect",
            )
            position_metadata = dict(self.metadata)
            position_metadata["window_width"] = int(patch_size)
            position_metadata["window_height"] = int(patch_size)
            position_metadata["patch_stride"] = int(patch_stride)
            position_metadata["coordinates_are_centers"] = bool(reflect_padding)
            position_metadata["coordinate_order"] = "row_col"
            return samples, _patch_indices_to_geo(coords, position_metadata)
        finally:
            patch_creator.close()

    def _scores_to_probability_map(self, positive_scores, sample_positions=None):
        positive_scores = np.asarray(positive_scores, dtype=np.float32).reshape(-1)
        positions = sample_positions
        if positions is None and self.coordinates is not None and len(self.coordinates) == len(positive_scores):
            positions = self.coordinates

        if positions is not None and len(positions) == len(positive_scores):
            positions = np.asarray(positions, dtype=np.float64)
            probability_map = np.zeros((int(self.metadata["ny"]), int(self.metadata["nx"])), dtype=np.float32)
            counts = np.zeros_like(probability_map, dtype=np.int32)

            x_values = np.asarray(positions[:, 0], dtype=np.float64)
            y_values = np.asarray(positions[:, 1], dtype=np.float64)
            x_span = max(float(self.converter.x_max - self.converter.x_min), 1e-12)
            y_span = max(float(self.converter.y_max - self.converter.y_min), 1e-12)

            pixel_x = np.rint((x_values - self.converter.x_min) / x_span * max(self.converter.nx - 1, 1)).astype(np.int64)
            pixel_y = np.rint((self.converter.y_max - y_values) / y_span * max(self.converter.ny - 1, 1)).astype(np.int64)
            pixel_x = np.clip(pixel_x, 0, self.converter.nx - 1)
            pixel_y = np.clip(pixel_y, 0, self.converter.ny - 1)

            np.add.at(probability_map, (pixel_y, pixel_x), positive_scores)
            np.add.at(counts, (pixel_y, pixel_x), 1)
            valid_mask = counts > 0
            probability_map[valid_mask] /= counts[valid_mask]
            self.last_prediction_positions = positions[:, :2].astype(np.float64, copy=False)
            return probability_map

        probability_map = self._reshape_probability_map(positive_scores)
        self.last_prediction_positions = None
        return probability_map

    def _reshape_probability_map(self, positive_scores):
        expected_size = int(self.metadata["ny"]) * int(self.metadata["nx"])
        if len(positive_scores) != expected_size:
            raise ValueError(
                f"Prediction size mismatch: got {len(positive_scores)} samples, expected {expected_size}."
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
