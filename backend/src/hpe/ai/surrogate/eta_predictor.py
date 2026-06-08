"""Dedicated η (efficiency) surrogate predictor (#30).

Physics-informed features derived from dimensionless pump similarity parameters:
    - Nq  (specific speed)                         — captures impeller type
    - D2/D1 (diameter ratio)                       — flow coefficient shape
    - b2/D2 (relative outlet width)                — head coefficient
    - Z  (blade count)                             — slip / loss trade-off
    - β2 (outlet blade angle, deg)                 — head coefficient
    - 1/(D2·n) proxy: u2=π·D2·n/60 (tip speed m/s) — Reynolds effect

The predictor trains on synthetic data (from the physics model) and can be
updated with real bench-test measurements via ``update(X_new, y_new)``.

Model: Gradient Boosting (scikit-learn) — fast, accurate, no GPU required.
       Features ~10 dimensionless variables; GBM typically reaches R²>0.97.

Persistence: pickled to ``~/.hpe/surrogate_eta.pkl`` (or custom path).
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path.home() / ".hpe" / "surrogate_eta.pkl"


def _physics_features(
    nq: float,
    d2: float,
    d1: float,
    b2: float,
    z: int,
    beta2: float,
    rpm: float,
) -> NDArray[np.float64]:
    """Compute physics-informed feature vector for η prediction.

    All inputs are dimensional (SI + degrees), but features are
    dimensionless ratios + log-transforms for better model conditioning.
    """
    d1_d2  = d1 / max(d2, 1e-6)
    b2_d2  = b2 / max(d2, 1e-6)
    u2     = math.pi * d2 * rpm / 60.0          # tip speed [m/s]
    phi    = math.radians(beta2)                 # outlet blade angle [rad]
    phi_t  = math.tan(phi)                       # tan(β2)

    # Logarithmic transforms for variables spanning orders of magnitude
    log_nq = math.log(max(nq, 1.0))
    log_u2 = math.log(max(u2, 1.0))

    return np.array([
        log_nq,
        d1_d2,
        b2_d2,
        float(z) / 10.0,     # normalise
        phi_t,
        log_u2,
        nq / 100.0,          # linear nq for interaction
        d1_d2 * phi_t,       # interaction feature
        b2_d2 * float(z),    # interaction feature
    ], dtype=np.float64)


class EtaSurrogate:
    """Gradient Boosting surrogate for hydraulic efficiency prediction.

    Usage::

        surr = EtaSurrogate()
        metrics = surr.train(n_samples=800, flow_rate=0.05, head=30, rpm=1750)
        eta = surr.predict(nq=35.0, d2=0.18, d1=0.09, b2=0.016, z=7, beta2=22.5, rpm=1750)
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or _DEFAULT_MODEL_PATH
        self._model: Optional[object] = None
        self._scaler: Optional[object] = None
        self.metrics: dict[str, float] = {}

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        n_samples: int = 800,
        flow_rate: float = 0.05,
        head: float = 30.0,
        rpm: float = 1750.0,
        seed: int = 42,
        save: bool = True,
    ) -> dict[str, float]:
        """Generate synthetic dataset from physics model, then train.

        The synthetic dataset perturbs (flow_rate, head, rpm) within
        physically plausible ranges and records the resulting sizing outputs.

        Args:
            n_samples: Number of latin-hypercube samples.
            flow_rate: Reference design flow rate [m³/s].
            head: Reference design head [m].
            rpm: Reference design speed [rpm].
            seed: Random seed for reproducibility.
            save: If True, persist model to ``self.model_path``.

        Returns:
            dict with ``r2_train``, ``r2_cv``, ``mae_cv``, ``n_train``.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from hpe.core.models import OperatingPoint
        from hpe.sizing import run_sizing

        rng = np.random.default_rng(seed)

        # Latin-hypercube sampling over (Q_ratio, H_ratio, N_ratio)
        q_ratios = rng.uniform(0.40, 1.60, n_samples)
        h_ratios = rng.uniform(0.25, 2.25, n_samples)
        n_ratios = rng.uniform(0.60, 1.40, n_samples)

        X_list: list[NDArray[np.float64]] = []
        y_list: list[float] = []

        skipped = 0
        for qr, hr, nr in zip(q_ratios, h_ratios, n_ratios):
            q_i = flow_rate * float(qr)
            h_i = head * float(hr)
            n_i = rpm * float(nr)
            try:
                r = run_sizing(OperatingPoint(flow_rate=q_i, head=h_i, rpm=n_i))
                feats = _physics_features(
                    nq=r.specific_speed_nq,
                    d2=r.impeller_d2,
                    d1=r.impeller_d1,
                    b2=r.impeller_b2,
                    z=r.blade_count,
                    beta2=r.beta2,
                    rpm=n_i,
                )
                X_list.append(feats)
                y_list.append(r.estimated_efficiency)
            except Exception:
                skipped += 1

        if len(X_list) < 20:
            raise RuntimeError(
                f"Insufficient valid samples ({len(X_list)}). "
                f"Got {skipped} failures out of {n_samples} attempts."
            )

        X = np.vstack(X_list)
        y = np.array(y_list)

        log.info("EtaSurrogate: training on %d samples (%d skipped)", len(X), skipped)

        # Pipeline: scaler + GBM
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gbm", GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=seed,
            )),
        ])
        pipe.fit(X, y)

        # Cross-validation R²
        cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2")
        cv_mae = -cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")

        self._model = pipe
        self._scaler = None  # already inside pipeline

        self.metrics = {
            "r2_train": float(pipe.score(X, y)),
            "r2_cv": float(np.mean(cv_r2)),
            "mae_cv": float(np.mean(cv_mae)),
            "n_train": len(X),
        }

        log.info(
            "EtaSurrogate: R²_train=%.3f R²_cv=%.3f MAE_cv=%.4f",
            self.metrics["r2_train"],
            self.metrics["r2_cv"],
            self.metrics["mae_cv"],
        )

        if save:
            self.save()

        return self.metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        nq: float,
        d2: float,
        d1: float,
        b2: float,
        z: int,
        beta2: float,
        rpm: float,
    ) -> float:
        """Predict hydraulic efficiency η for a given impeller design.

        Args:
            nq: Specific speed [—].
            d2: Outlet diameter [m].
            d1: Inlet diameter [m].
            b2: Outlet width [m].
            z: Blade count [—].
            beta2: Outlet blade angle [deg].
            rpm: Rotational speed [rpm].

        Returns:
            Predicted efficiency η ∈ (0, 1).
        """
        if self._model is None:
            raise RuntimeError("EtaSurrogate not trained. Call train() or load() first.")

        feats = _physics_features(nq=nq, d2=d2, d1=d1, b2=b2, z=z, beta2=beta2, rpm=rpm)
        eta = float(self._model.predict(feats.reshape(1, -1))[0])  # type: ignore[union-attr]
        return max(0.0, min(1.0, eta))

    def predict_from_sizing(self, sizing_result: object, rpm: float) -> float:
        """Convenience wrapper: predict directly from a SizingResult."""
        r = sizing_result
        return self.predict(
            nq=r.specific_speed_nq,  # type: ignore[attr-defined]
            d2=r.impeller_d2,  # type: ignore[attr-defined]
            d1=r.impeller_d1,  # type: ignore[attr-defined]
            b2=r.impeller_b2,  # type: ignore[attr-defined]
            z=r.blade_count,  # type: ignore[attr-defined]
            beta2=r.beta2,  # type: ignore[attr-defined]
            rpm=rpm,
        )

    # ── Online update ─────────────────────────────────────────────────────────

    def update(
        self,
        X_new: NDArray[np.float64],
        y_new: NDArray[np.float64],
        save: bool = True,
    ) -> None:
        """Partial-update with new bench-test data points.

        Re-trains the GBM on augmented dataset (warm-start not supported
        by GBM, so we keep a buffer).  Intended for continual learning with
        real bancada data.

        Args:
            X_new: (n_new, n_features) array of raw physics features.
            y_new: (n_new,) array of measured η values.
            save: Persist updated model to disk.
        """
        if self._model is None:
            raise RuntimeError("Cannot update — model not initialised.")

        # For now, re-fit the existing pipeline on augmented data.
        # In production: keep a rolling buffer in Redis/Postgres.

        existing_X = getattr(self, "_X_buffer", np.empty((0, X_new.shape[1])))
        existing_y = getattr(self, "_y_buffer", np.empty(0))

        self._X_buffer = np.vstack([existing_X, X_new])
        self._y_buffer = np.concatenate([existing_y, y_new])

        self._model.fit(self._X_buffer, self._y_buffer)  # type: ignore[union-attr]
        log.info("EtaSurrogate: updated with %d new points (total %d)", len(y_new), len(self._y_buffer))

        if save:
            self.save()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the model to disk."""
        out = Path(path or self.model_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump({"model": self._model, "metrics": self.metrics}, f)
        log.info("EtaSurrogate: saved to %s", out)
        return out

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EtaSurrogate":
        """Load a previously saved EtaSurrogate from disk."""
        load_path = Path(path or _DEFAULT_MODEL_PATH)
        if not load_path.exists():
            raise FileNotFoundError(f"No surrogate model found at {load_path}")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        obj = cls(model_path=load_path)
        obj._model = data["model"]
        obj.metrics = data.get("metrics", {})
        log.info("EtaSurrogate: loaded from %s  (metrics=%s)", load_path, obj.metrics)
        return obj

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        status = f"trained R²_cv={self.metrics.get('r2_cv', '?'):.3f}" if self.is_trained else "untrained"
        return f"EtaSurrogate({status})"
