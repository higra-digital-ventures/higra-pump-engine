"""Surrogate model training pipeline with experiment tracking.

Manages the lifecycle of surrogate models: data collection,
training, evaluation, and comparison with previous versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hpe.ai.surrogate.dataset import SurrogateDataset, generate_dataset
from hpe.ai.surrogate.model import SurrogateMetrics, SurrogateModel
from hpe.ai.surrogate.predictor import SurrogatePredictor


@dataclass
class TrainingResult:
    """Result of a training or retraining cycle."""

    new_metrics: SurrogateMetrics
    old_metrics: Optional[SurrogateMetrics]
    improved: bool  # True if new model is better
    accepted: bool  # True if new model was accepted
    n_total_samples: int


def retrain_surrogate(
    predictor: SurrogatePredictor,
    n_new_samples: int = 100,
    seed: int = None,
    min_improvement: float = 0.01,
) -> TrainingResult:
    """Retrain surrogate with additional samples.

    Generates new training data, combines with existing data,
    trains a new model, and accepts it only if R^2 improves.

    Args:
        predictor: Existing SurrogatePredictor to improve.
        n_new_samples: Number of new samples to generate.
        seed: Random seed for new data.
        min_improvement: Minimum R^2 improvement to accept new model.

    Returns:
        TrainingResult with comparison metrics.
    """
    old_metrics = predictor.metrics

    # Generate new data
    new_seed = seed if seed is not None else (42 + n_new_samples)
    new_dataset = generate_dataset(predictor.problem, n_new_samples, new_seed)

    # Combine with existing data if available
    if predictor.dataset is not None:
        combined_X = np.vstack([predictor.dataset.X, new_dataset.X])
        combined_y = np.vstack([predictor.dataset.y, new_dataset.y])
        combined = SurrogateDataset(
            X=combined_X,
            y=combined_y,
            feature_names=new_dataset.feature_names,
            target_names=new_dataset.target_names,
            n_feasible=predictor.dataset.n_feasible + new_dataset.n_feasible,
        )
    else:
        combined = new_dataset

    # Train new model
    new_model = SurrogateModel()
    new_metrics = new_model.train(combined)

    # Compare
    improved = False
    if old_metrics is not None:
        improved = new_metrics.mean_r2 > old_metrics.mean_r2 + min_improvement
    else:
        improved = True  # First training always accepted

    # Accept if improved
    accepted = improved
    if accepted:
        predictor.model = new_model
        predictor.dataset = combined
        predictor.metrics = new_metrics

    return TrainingResult(
        new_metrics=new_metrics,
        old_metrics=old_metrics,
        improved=improved,
        accepted=accepted,
        n_total_samples=len(combined.X),
    )


def incremental_train(
    predictor: SurrogatePredictor,
    new_X: np.ndarray,
    new_y: np.ndarray,
) -> SurrogateMetrics:
    """Add new data points and retrain.

    Used when new CFD or test bench data becomes available.

    Args:
        predictor: Existing predictor.
        new_X: New feature vectors (n_new, n_vars).
        new_y: New target vectors (n_new, n_obj).

    Returns:
        Updated SurrogateMetrics.
    """
    if predictor.dataset is None:
        raise RuntimeError("Predictor has no existing dataset. Use build() first.")

    combined_X = np.vstack([predictor.dataset.X, new_X])
    combined_y = np.vstack([predictor.dataset.y, new_y])

    predictor.dataset = SurrogateDataset(
        X=combined_X,
        y=combined_y,
        feature_names=predictor.dataset.feature_names,
        target_names=predictor.dataset.target_names,
        n_feasible=predictor.dataset.n_feasible + len(new_X),
    )

    metrics = predictor.model.train(predictor.dataset)
    predictor.metrics = metrics
    return metrics
