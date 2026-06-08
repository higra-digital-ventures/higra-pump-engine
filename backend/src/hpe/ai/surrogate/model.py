"""Surrogate model — fast performance prediction without physics evaluation.

Uses scikit-learn RandomForest as a baseline model. Can be upgraded
to PyTorch neural networks when more data is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor

from hpe.ai.surrogate.dataset import SurrogateDataset


@dataclass
class SurrogateMetrics:
    """Training metrics for surrogate model."""

    r2_scores: dict[str, float]  # R² per objective
    mean_r2: float
    n_train: int
    n_test: int


class SurrogateModel:
    """Random Forest surrogate model for pump performance prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
        )
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self._is_trained = False

    def train(self, dataset: SurrogateDataset) -> SurrogateMetrics:
        """Train the surrogate model on a dataset.

        Args:
            dataset: SurrogateDataset with features and targets.

        Returns:
            SurrogateMetrics with R² scores.
        """
        self.feature_names = dataset.feature_names
        self.target_names = dataset.target_names

        # Filter feasible samples (remove penalty values)
        mask = dataset.y[:, 0] > 0  # efficiency > 0 means feasible
        X = dataset.X[mask]
        y = dataset.y[mask]

        if len(X) < 10:
            raise ValueError(f"Too few feasible samples ({len(X)}). Need at least 10.")

        # Train
        self.model.fit(X, y)
        self._is_trained = True

        # Evaluate with cross-validation
        n_cv = min(5, len(X))
        r2_scores = {}
        for i, name in enumerate(self.target_names):
            rf = RandomForestRegressor(
                n_estimators=100, random_state=42,
            )
            scores = cross_val_score(rf, X, y[:, i], cv=n_cv, scoring="r2")
            r2_scores[name] = float(np.mean(scores))

        return SurrogateMetrics(
            r2_scores=r2_scores,
            mean_r2=float(np.mean(list(r2_scores.values()))),
            n_train=len(X),
            n_test=len(X) // n_cv,
        )

    def predict(self, design_vector: list[float]) -> dict[str, float]:
        """Predict objectives for a design vector.

        Args:
            design_vector: List of design variable values.

        Returns:
            Dict of {objective_name: predicted_value}.

        Raises:
            RuntimeError: If model is not trained.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.array([design_vector])
        y_pred = self.model.predict(X)[0]

        return {
            name: float(y_pred[i])
            for i, name in enumerate(self.target_names)
        }

    def predict_batch(self, design_vectors: np.ndarray) -> np.ndarray:
        """Predict objectives for a batch of design vectors.

        Args:
            design_vectors: (n_samples, n_variables) array.

        Returns:
            (n_samples, n_objectives) array of predictions.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(design_vectors)

    @property
    def is_trained(self) -> bool:
        return self._is_trained
