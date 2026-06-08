"""Experiment tracking for surrogate model training.

Wraps MLflow to track training runs, model versions, and metrics.
Falls back to local logging if MLflow is not available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hpe.ai.surrogate.model import SurrogateMetrics


@dataclass
class ExperimentRecord:
    """Record of a single training experiment."""

    experiment_id: str
    run_id: str
    metrics: dict[str, float]
    params: dict[str, Any]
    model_path: Optional[str] = None


def log_training_run(
    metrics: SurrogateMetrics,
    params: dict[str, Any],
    experiment_name: str = "hpe-surrogate",
    use_mlflow: bool = False,
) -> ExperimentRecord:
    """Log a training run to MLflow or local file.

    Args:
        metrics: SurrogateMetrics from model training.
        params: Training parameters (n_samples, seed, etc.).
        experiment_name: MLflow experiment name.
        use_mlflow: Whether to use MLflow (requires server running).

    Returns:
        ExperimentRecord with IDs.
    """
    metrics_dict = {
        "mean_r2": metrics.mean_r2,
        "n_train": metrics.n_train,
        **{f"r2_{k}": v for k, v in metrics.r2_scores.items()},
    }

    if use_mlflow:
        return _log_to_mlflow(metrics_dict, params, experiment_name)
    else:
        return _log_to_file(metrics_dict, params, experiment_name)


def _log_to_mlflow(
    metrics: dict[str, float],
    params: dict[str, Any],
    experiment_name: str,
) -> ExperimentRecord:
    """Log to MLflow server."""
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)

            return ExperimentRecord(
                experiment_id=run.info.experiment_id,
                run_id=run.info.run_id,
                metrics=metrics,
                params=params,
            )
    except Exception:
        # Fall back to file logging
        return _log_to_file(metrics, params, experiment_name)


def _log_to_file(
    metrics: dict[str, float],
    params: dict[str, Any],
    experiment_name: str,
) -> ExperimentRecord:
    """Log to local JSON file."""
    import uuid
    from datetime import datetime

    run_id = str(uuid.uuid4())[:8]
    exp_id = experiment_name

    log_dir = Path("output") / "experiments" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "params": {k: str(v) for k, v in params.items()},
    }

    log_file = log_dir / f"run_{run_id}.json"
    log_file.write_text(json.dumps(record, indent=2))

    return ExperimentRecord(
        experiment_id=exp_id,
        run_id=run_id,
        metrics=metrics,
        params=params,
        model_path=str(log_file),
    )
