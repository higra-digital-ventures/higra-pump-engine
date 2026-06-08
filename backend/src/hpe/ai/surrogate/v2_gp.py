"""Surrogate model v2 — Gaussian Process pump performance predictor.

Provides native uncertainty quantification (posterior std) compared to v1 XGBoost.
Suitable for active learning loops and Bayesian optimization acquisition functions.

Why GP for v2 (vs v1 XGBoost)
--------------------------------
- Native posterior uncertainty — P(y|x) not just point estimate
- Better extrapolation behaviour in sparse regions
- Foundation for Expected Improvement / UCB acquisition (Fase 3 BO loop)
- Interpretable length-scales per feature (kernel hyperparameter)

Computational trade-off
------------------------
GP exact inference is O(n^3). With n=2931 training points this is ~30s.
Strategy: subsample 500 points for GP training (covering the Ns/D2 space
well enough for pump sizing). Full dataset used for StandardScaler fit.
Alternatively, set USE_FULL_DATASET=True to train on all points (slow).

Acceptance criterion (same as v1)
------------------------------------
  RMSE ≤ 8% (relative) on 20% holdout test set for eta_total.

Usage
-----
    from hpe.ai.surrogate.v2_gp import SurrogateV2GP, SurrogateInput

    model = SurrogateV2GP()
    result = model.train("dataset/bancada_features.parquet")
    pred, sigma = model.predict_with_uncertainty(
        SurrogateInput(ns=35.0, d2_mm=320.0, q_m3h=200.0, h_m=45.0, n_rpm=1750)
    )
    print(pred.eta_total, sigma)

    model.save("models/surrogate_v2_gp.pkl")
"""

from __future__ import annotations

import logging
import time
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# scikit-learn GP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MLflow (optional — graceful degradation if unavailable)
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature / Target definitions (same columns as v1 for compatibility)
# ---------------------------------------------------------------------------

PRIMARY_FEATURES = [
    "feat_ns",
    "feat_nq",
    "feat_d2_m",
    "feat_u2",
    "feat_psi",
    "feat_phi",
    "feat_re",
    "feat_q_star",
    "feat_h_star",
    "feat_nstages",
]

RAW_FEATURES = ["q_m3s", "h_stage_m", "n_rpm", "d2_mm"]

ALL_FEATURES = PRIMARY_FEATURES + RAW_FEATURES

TARGETS = {
    "eta_total": "Pump total efficiency [%]",
    "eta_hid":   "Hydraulic efficiency [%]",
    "p_kw":      "Shaft power [kW]",
}

G = 9.80665  # m/s²

# GP subsample size — 500 covers Ns/D2 space well while keeping fit under 60s
GP_SUBSAMPLE = 500
USE_FULL_DATASET = False   # set True to train on all ~2931 rows (slow, ~60-120s)


# ---------------------------------------------------------------------------
# Shared input/output dataclasses (compatible with v1)
# ---------------------------------------------------------------------------

@dataclass
class SurrogateInput:
    """Minimal inputs for a v2 GP prediction.

    Parameters match v1_xgboost.SurrogateInput for drop-in compatibility.
    """
    ns: float       # European specific speed n·√Q / H^0.75 [rpm, m³/s, m]
    d2_mm: float    # Impeller outlet diameter [mm]
    q_m3h: float    # Flow rate [m³/h]
    h_m: float      # Total head per stage [m]
    n_rpm: float    # Rotational speed [rpm]
    n_stages: int = 1

    def to_feature_dict(self) -> dict[str, float]:
        """Compute all model features from raw inputs (same logic as v1)."""
        q = self.q_m3h / 3600.0
        d2 = self.d2_mm / 1000.0
        h = self.h_m / self.n_stages
        n = self.n_rpm
        eps = 1e-9

        u2 = np.pi * d2 * n / 60.0
        nq = self.ns / 51.65
        psi = G * h / (u2**2 + eps)
        phi = q / (u2 * np.pi / 4 * d2**2 + eps)
        re = u2 * d2 / 1e-6

        return {
            "feat_ns":      self.ns,
            "feat_nq":      nq,
            "feat_d2_m":    d2,
            "feat_u2":      u2,
            "feat_psi":     psi,
            "feat_phi":     phi,
            "feat_re":      re,
            "feat_q_star":  1.0,
            "feat_h_star":  1.0,
            "feat_nstages": float(self.n_stages),
            "q_m3s":        q,
            "h_stage_m":    h,
            "n_rpm":        n,
            "d2_mm":        self.d2_mm,
        }


@dataclass
class SurrogateOutput:
    """Prediction output compatible with v1_xgboost.SurrogateOutput."""
    eta_total: float
    eta_hid: float
    p_kw: float
    eta_total_std: float = 0.0
    eta_hid_std: float = 0.0
    p_kw_std: float = 0.0
    latency_ms: float = 0.0


@dataclass
class EvalMetrics:
    """Per-target evaluation metrics."""
    target: str
    rmse: float
    rmse_pct: float
    mae: float
    r2: float
    n_test: int
    passes_criterion: bool  # rmse_pct ≤ 8%


@dataclass
class TrainingResult:
    """Summary of a v2 GP training run."""
    mlflow_run_id: str
    metrics: list[EvalMetrics]
    kernel_params: dict[str, Any]
    train_rows: int
    test_rows: int
    training_time_s: float
    model_path: str
    gp_subsample: int


# ---------------------------------------------------------------------------
# SurrogateV2GP
# ---------------------------------------------------------------------------

class SurrogateV2GP:
    """Gaussian Process multi-output surrogate for centrifugal pump performance.

    Trains one GPR per target variable (eta_total, eta_hid, p_kw).
    Input normalization via StandardScaler (required for GP length-scale
    hyperparameter optimisation to be well-conditioned).

    Kernel
    ------
    ``ConstantKernel(1.0) * RBF([1.0]*n_features) + WhiteKernel(0.01)``
    - ConstantKernel: output scale (signal variance)
    - RBF: separate length-scale per feature (Automatic Relevance Determination)
    - WhiteKernel: noise level (handles pump-to-pump variability)

    Uncertainty
    -----------
    ``predict_with_uncertainty()`` returns (SurrogateOutput, sigma_pct) where
    sigma_pct is the GP posterior std normalised by the target mean — a direct
    measure of prediction confidence in percentage points.
    """

    VERSION = "2.0.0"
    EXPERIMENT_NAME = "hpe-surrogate-v2-gp"

    def __init__(
        self,
        n_restarts_optimizer: int = 3,
        subsample: int = GP_SUBSAMPLE,
        use_full_dataset: bool = USE_FULL_DATASET,
    ):
        self.n_restarts_optimizer = n_restarts_optimizer
        self.subsample = subsample
        self.use_full_dataset = use_full_dataset

        self.models: dict[str, GaussianProcessRegressor] = {}
        self.scaler = StandardScaler()
        self.feature_names: list[str] = ALL_FEATURES
        self._target_means: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, features_path: str, test_size: float = 0.20) -> TrainingResult:
        """Train GP surrogate on bancada feature parquet.

        Parameters
        ----------
        features_path : str
            Path to ``bancada_features.parquet`` (output of bancada_etl).
        test_size : float
            Fraction of data held out for final evaluation (default 0.20).

        Returns
        -------
        TrainingResult
            MLflow run ID, metrics, kernel params, model path.
        """
        t_start = time.perf_counter()

        df = pd.read_parquet(features_path)
        log.info("v2_gp train: loaded %d rows from %s", len(df), features_path)

        # Validate columns
        missing = [c for c in ALL_FEATURES + list(TARGETS) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        X_all = df[ALL_FEATURES].astype(float)
        y_all = {t: df[t].astype(float) for t in TARGETS}

        # Train/test split
        nq_bucket = pd.cut(df["feat_ns"].clip(5, 200), bins=5, labels=False).fillna(0)
        X_train_full, X_test, idx_train, idx_test = train_test_split(
            X_all, X_all.index,
            test_size=test_size,
            random_state=42,
            stratify=nq_bucket,
        )
        y_train_full = {t: y_all[t].loc[idx_train] for t in TARGETS}
        y_test = {t: y_all[t].loc[idx_test] for t in TARGETS}

        # Fit scaler on full training set
        self.scaler.fit(X_train_full)

        # Subsample for GP fit (O(n^3) complexity)
        if self.use_full_dataset:
            X_train = X_train_full
            y_train = y_train_full
            log.info("v2_gp train: using full dataset (%d rows)", len(X_train))
        else:
            n_sub = min(self.subsample, len(X_train_full))
            rng = np.random.default_rng(42)
            sub_idx = rng.choice(len(X_train_full), size=n_sub, replace=False)
            X_train = X_train_full.iloc[sub_idx]
            y_train = {t: y_train_full[t].iloc[sub_idx] for t in TARGETS}
            log.info(
                "v2_gp train: subsampled %d/%d rows for GP fit",
                n_sub, len(X_train_full),
            )

        X_train_sc = self.scaler.transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        n_features = X_train_sc.shape[1]

        # Build kernel: ConstantKernel * RBF(ARD) + WhiteKernel
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(
                length_scale=np.ones(n_features),
                length_scale_bounds=(1e-2, 1e2),
            )
            + WhiteKernel(
                noise_level=0.01,
                noise_level_bounds=(1e-4, 1e0),
            )
        )

        mlflow_run_id = ""
        all_metrics: list[EvalMetrics] = []
        kernel_params: dict[str, Any] = {}

        if _MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.EXPERIMENT_NAME)
            ctx = mlflow.start_run()
        else:
            ctx = _NullContext()

        with ctx as run:
            if _MLFLOW_AVAILABLE and run is not None:
                mlflow_run_id = run.info.run_id
                mlflow.log_params({
                    "n_features": n_features,
                    "train_rows_full": len(X_train_full),
                    "train_rows_gp": len(X_train),
                    "test_rows": len(X_test),
                    "n_restarts_optimizer": self.n_restarts_optimizer,
                    "use_full_dataset": self.use_full_dataset,
                })

            for target in TARGETS:
                log.info("v2_gp train: fitting GP for target='%s'...", target)
                t_target = time.perf_counter()

                gp = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(kernel.theta),
                    n_restarts_optimizer=self.n_restarts_optimizer,
                    normalize_y=True,
                    random_state=42,
                )

                y_tr = y_train[target].values
                self._target_means[target] = float(np.mean(np.abs(y_tr)))

                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    gp.fit(X_train_sc, y_tr)

                self.models[target] = gp

                # Test set evaluation
                y_pred = gp.predict(X_test_sc)
                metrics = self._compute_metrics(
                    target, y_test[target].values, y_pred
                )
                all_metrics.append(metrics)

                # Store fitted kernel params
                kernel_params[target] = {
                    "log_marginal_likelihood": round(
                        float(gp.log_marginal_likelihood(gp.kernel_.theta)), 4
                    ),
                    "kernel": str(gp.kernel_),
                }

                elapsed_target = time.perf_counter() - t_target
                log.info(
                    "v2_gp train: %s — RMSE=%.2f (%.1f%%), R²=%.3f %s [%.1fs]",
                    target, metrics.rmse, metrics.rmse_pct, metrics.r2,
                    "OK" if metrics.passes_criterion else "FAIL",
                    elapsed_target,
                )

                if _MLFLOW_AVAILABLE and run is not None:
                    mlflow.log_metrics({
                        f"{target}_rmse":     metrics.rmse,
                        f"{target}_rmse_pct": metrics.rmse_pct,
                        f"{target}_mae":      metrics.mae,
                        f"{target}_r2":       metrics.r2,
                    })

            # Save model artifact
            models_dir = Path(features_path).parent.parent / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = str(models_dir / "surrogate_v2_gp.pkl")
            self.save(model_path)

            if _MLFLOW_AVAILABLE and run is not None:
                mlflow.log_artifact(model_path, artifact_path="model")
                mlflow.set_tag("version", self.VERSION)
                mlflow.set_tag("criterion_passed",
                               str(all(m.passes_criterion for m in all_metrics)))
                mlflow.set_tag("gp_subsample", str(len(X_train)))

        training_time = time.perf_counter() - t_start
        log.info(
            "v2_gp train: completed in %.1f s, MLflow run_id=%s",
            training_time, mlflow_run_id,
        )

        return TrainingResult(
            mlflow_run_id=mlflow_run_id,
            metrics=all_metrics,
            kernel_params=kernel_params,
            train_rows=len(X_train),
            test_rows=len(X_test),
            training_time_s=round(training_time, 2),
            model_path=model_path,
            gp_subsample=len(X_train),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, inp: SurrogateInput) -> SurrogateOutput:
        """Point prediction — returns SurrogateOutput without uncertainty.

        Equivalent interface to v1_xgboost.SurrogateV1.predict().
        """
        out, _ = self.predict_with_uncertainty(inp)
        return out

    def predict_with_uncertainty(
        self, inp: SurrogateInput
    ) -> tuple[SurrogateOutput, float]:
        """Predict with GP posterior uncertainty.

        Parameters
        ----------
        inp : SurrogateInput
            Pump operating conditions and basic geometry.

        Returns
        -------
        (SurrogateOutput, sigma_pct) where sigma_pct is the mean normalised
        posterior std across targets expressed as a percentage.
        A sigma_pct < 5% indicates the point is well-covered by training data.
        """
        if not self.models:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        t0 = time.perf_counter()
        feat = inp.to_feature_dict()
        X_raw = pd.DataFrame([feat])[ALL_FEATURES].astype(float)
        X_sc = self.scaler.transform(X_raw)

        preds: dict[str, float] = {}
        stds: dict[str, float] = {}

        for target, gp in self.models.items():
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                y_mean, y_std = gp.predict(X_sc, return_std=True)
            preds[target] = float(y_mean[0])
            # Normalise std by target mean to get relative uncertainty
            t_mean = self._target_means.get(target, 1.0) or 1.0
            stds[target] = float(y_std[0] / t_mean * 100.0)

        latency_ms = (time.perf_counter() - t0) * 1000
        sigma_pct_mean = float(np.mean(list(stds.values())))

        output = SurrogateOutput(
            eta_total=round(preds.get("eta_total", 0.0), 2),
            eta_hid=round(preds.get("eta_hid", 0.0), 2),
            p_kw=round(preds.get("p_kw", 0.0), 2),
            eta_total_std=round(stds.get("eta_total", 0.0), 3),
            eta_hid_std=round(stds.get("eta_hid", 0.0), 3),
            p_kw_std=round(stds.get("p_kw", 0.0), 3),
            latency_ms=round(latency_ms, 2),
        )
        return output, sigma_pct_mean

    def predict_batch(self, inputs: list[SurrogateInput]) -> list[SurrogateOutput]:
        """Batch prediction without uncertainty (faster for optimisation loops)."""
        if not self.models:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        t0 = time.perf_counter()
        rows = [inp.to_feature_dict() for inp in inputs]
        X_raw = pd.DataFrame(rows)[ALL_FEATURES].astype(float)
        X_sc = self.scaler.transform(X_raw)
        preds: dict[str, np.ndarray] = {}
        for target, gp in self.models.items():
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                preds[target] = gp.predict(X_sc)
        latency_ms = (time.perf_counter() - t0) * 1000 / len(inputs)
        return [
            SurrogateOutput(
                eta_total=round(float(preds["eta_total"][i]), 2),
                eta_hid=round(float(preds.get("eta_hid", np.zeros(len(inputs)))[i]), 2),
                p_kw=round(float(preds.get("p_kw", np.zeros(len(inputs)))[i]), 2),
                latency_ms=round(latency_ms, 2),
            )
            for i in range(len(inputs))
        ]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_df: pd.DataFrame) -> list[EvalMetrics]:
        """Evaluate model on an external test set."""
        if not self.models:
            raise RuntimeError("Model not trained.")
        X_sc = self.scaler.transform(test_df[ALL_FEATURES].astype(float))
        metrics = []
        for target, gp in self.models.items():
            if target not in test_df.columns:
                continue
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                y_pred = gp.predict(X_sc)
            metrics.append(
                self._compute_metrics(target, test_df[target].values, y_pred)
            )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise GP bundle to pickle."""
        bundle = {
            "version": self.VERSION,
            "models": self.models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target_means": self._target_means,
            "subsample": self.subsample,
        }
        joblib.dump(bundle, path)
        log.info("v2_gp save: bundle saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved GP bundle."""
        bundle = joblib.load(path)
        self.models = bundle.get("models", {})
        self.scaler = bundle.get("scaler", StandardScaler())
        self.feature_names = bundle.get("feature_names", ALL_FEATURES)
        self._target_means = bundle.get("target_means", {})
        self.subsample = bundle.get("subsample", GP_SUBSAMPLE)
        log.info("v2_gp load: GP v%s loaded from %s", self.VERSION, path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        target: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> EvalMetrics:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        mean_val = float(np.mean(np.abs(y_true)))
        rmse_pct = (rmse / mean_val * 100) if mean_val > 0 else 999.0
        return EvalMetrics(
            target=target,
            rmse=round(rmse, 3),
            rmse_pct=round(rmse_pct, 2),
            mae=round(mae, 3),
            r2=round(r2, 4),
            n_test=len(y_true),
            passes_criterion=rmse_pct <= 8.0,
        )


# ---------------------------------------------------------------------------
# Null context manager — used when MLflow is unavailable
# ---------------------------------------------------------------------------

class _NullContext:
    """No-op context manager substituting mlflow.start_run()."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train HPE Surrogate v2 GP")
    parser.add_argument(
        "--features",
        default=str(Path(__file__).resolve().parents[5] / "dataset" / "bancada_features.parquet"),
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--full", action="store_true", help="Use full dataset (slow)")
    args = parser.parse_args()

    model = SurrogateV2GP(use_full_dataset=args.full)
    result = model.train(args.features, test_size=args.test_size)

    print("\n=== SurrogateV2GP Training Results ===")
    print(f"MLflow run  : {result.mlflow_run_id}")
    print(f"GP rows     : {result.train_rows} (subsampled) / test: {result.test_rows}")
    print(f"Training    : {result.training_time_s:.1f}s")
    print(f"Model saved : {result.model_path}")
    print("\n--- Metrics ---")
    for m in result.metrics:
        status = "PASS" if m.passes_criterion else "FAIL (criterion: RMSE ≤ 8%)"
        print(f"  {m.target:12s}  RMSE={m.rmse:.2f} ({m.rmse_pct:.1f}%)  R²={m.r2:.3f}  {status}")

    # Smoke test with uncertainty
    test_input = SurrogateInput(ns=35.0, d2_mm=320.0, q_m3h=200.0, h_m=45.0, n_rpm=1750)
    pred, sigma = model.predict_with_uncertainty(test_input)
    print("\n--- Smoke Test (Ns=35, D2=320mm) ---")
    print(f"  η_total = {pred.eta_total:.1f}% ± {pred.eta_total_std:.2f}%")
    print(f"  η_hid   = {pred.eta_hid:.1f}% ± {pred.eta_hid_std:.2f}%")
    print(f"  P_shaft = {pred.p_kw:.1f} kW  (σ = {sigma:.2f}% mean)")
    print(f"  Latency = {pred.latency_ms:.2f} ms")
