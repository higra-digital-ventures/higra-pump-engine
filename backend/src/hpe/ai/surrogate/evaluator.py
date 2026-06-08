"""SurrogateEvaluator — unified interface for all HPE surrogate versions.

Provides a single, version-agnostic API so that callers (API, CLI,
optimization loop) never depend on a specific surrogate implementation.

Surrogate roadmap
-----------------
v1  XGBoost (this file wraps v1_xgboost.py)          — Fase 1 (current)
v2  Gaussian Process / Random Forest (H-Q curve)      — Fase 3
v3  Graph Neural Network (PyTorch Geometric)           — Fase 4

Usage
-----
    from hpe.ai.surrogate.evaluator import SurrogateEvaluator, SurrogateInput

    ev = SurrogateEvaluator.load_default()
    result = ev.predict(SurrogateInput(ns=35, d2_mm=320, q_m3h=200, h_m=45, n_rpm=1750))
    print(result.eta_total, result.confidence)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default model path (override with HPE_SURROGATE_PATH env var)
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[5] / "models" / "surrogate_v1.pkl"
)


# ---------------------------------------------------------------------------
# Canonical input / output (v2.0 spec — matches document interface)
# ---------------------------------------------------------------------------

@dataclass
class SurrogateInput:
    """Canonical input for all surrogate versions.

    Parameters
    ----------
    Ns : float
        Dimensional specific speed n*sqrt(Q)/H^0.75 [rpm, m3/s, m].
    D2 : float
        Impeller outlet diameter [mm].
    b2 : float
        Impeller outlet width [mm].
    beta2 : float
        Blade outlet angle [deg].
    n : float
        Rotational speed [rpm].
    Q : float
        Flow rate [m3/s].
    H : float
        Total head [m].  Used to compute derived features.
    n_stages : int
        Number of stages (default 1).
    """
    Ns: float
    D2: float        # mm
    b2: float        # mm
    beta2: float     # deg
    n: float         # rpm
    Q: float         # m3/s
    H: float         # m
    n_stages: int = 1

    def to_v1_input(self):
        """Convert to v1_xgboost.SurrogateInput format."""
        from hpe.ai.surrogate.v1_xgboost import SurrogateInput as V1Input
        return V1Input(
            ns=self.Ns,
            d2_mm=self.D2,
            q_m3h=self.Q * 3600,
            h_m=self.H,
            n_rpm=self.n,
            n_stages=self.n_stages,
        )


@dataclass
class SurrogateOutput:
    """Canonical output for all surrogate versions.

    Parameters
    ----------
    eta_hid : float
        Predicted hydraulic efficiency [%].
    H : float
        Predicted actual head [m].  (same as input for v1; modelled in v2+)
    P_shaft : float
        Predicted shaft power [kW].
    confidence : float
        Model confidence score 0-1.
        v1: based on Ns proximity to training data.
        v2+: GP posterior std / ensemble spread.
    surrogate_version : str
        Version tag of the model that produced this prediction.
    latency_ms : float
        Inference wall-clock time [ms].
    """
    eta_hid: float
    H: float
    P_shaft: float
    confidence: float
    surrogate_version: str = "v1"
    latency_ms: float = 0.0

    @property
    def eta_total(self) -> float:
        """Alias — same as eta_hid in v1 (motor not modelled)."""
        return self.eta_hid


# ---------------------------------------------------------------------------
# EvalMetrics (re-exported for API consumers)
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """Evaluation metrics per target."""
    target: str
    rmse: float
    rmse_pct: float
    mae: float
    r2: float
    n_test: int
    passes_criterion: bool


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

class SurrogateEvaluator:
    """Version-agnostic surrogate evaluator.

    Wraps the active surrogate model and exposes a stable interface
    regardless of the underlying implementation (XGBoost, GP, GNN).

    Parameters
    ----------
    model_path : str, optional
        Path to the .pkl model bundle.  Defaults to HPE_SURROGATE_PATH
        env var or ``<repo>/models/surrogate_v1.pkl``.
    version : str
        Version tag, e.g. 'v1', 'v2'.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        version: str = "v1",
    ):
        self.model_path = model_path or os.getenv("HPE_SURROGATE_PATH", DEFAULT_MODEL_PATH)
        self.version = version
        self._model = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load_default(cls) -> "SurrogateEvaluator":
        """Load the default (currently active) surrogate model."""
        ev = cls()
        ev._load()
        return ev

    @classmethod
    def train_from_bancada(
        cls,
        features_path: Optional[str] = None,
        test_size: float = 0.20,
    ) -> "SurrogateEvaluator":
        """Train a fresh v1 surrogate from bancada features and return evaluator.

        Parameters
        ----------
        features_path : str, optional
            Path to bancada_features.parquet.  Auto-detected if None.
        test_size : float
            Holdout fraction for evaluation.

        Returns
        -------
        SurrogateEvaluator
            Ready-to-use evaluator with trained model.
        """
        if features_path is None:
            features_path = str(
                Path(DEFAULT_MODEL_PATH).parent.parent / "dataset" / "bancada_features.parquet"
            )
        from hpe.ai.surrogate.v1_xgboost import SurrogateV1
        v1 = SurrogateV1()
        result = v1.train(features_path, test_size=test_size)
        log.info("SurrogateEvaluator: trained v1 — %s", result.mlflow_run_id)

        ev = cls(model_path=result.model_path)
        ev._model = v1
        return ev

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def predict(self, inp: SurrogateInput) -> SurrogateOutput:
        """Predict pump performance for a single operating point.

        Parameters
        ----------
        inp : SurrogateInput
            Canonical input (version-agnostic).

        Returns
        -------
        SurrogateOutput
            Predicted eta_hid, H, P_shaft with confidence score.
        """
        self._ensure_loaded()

        v1_input = inp.to_v1_input()
        v1_out = self._model.predict(v1_input)

        confidence = self._estimate_confidence(inp)

        return SurrogateOutput(
            eta_hid=v1_out.eta_hid,
            H=inp.H,
            P_shaft=v1_out.p_kw,
            confidence=confidence,
            surrogate_version=self.version,
            latency_ms=v1_out.latency_ms,
        )

    def predict_batch(self, inputs: list[SurrogateInput]) -> list[SurrogateOutput]:
        """Batch prediction for multiple operating points."""
        self._ensure_loaded()
        v1_inputs = [inp.to_v1_input() for inp in inputs]
        v1_outputs = self._model.predict_batch(v1_inputs)
        return [
            SurrogateOutput(
                eta_hid=o.eta_hid,
                H=inp.H,
                P_shaft=o.p_kw,
                confidence=self._estimate_confidence(inp),
                surrogate_version=self.version,
                latency_ms=o.latency_ms,
            )
            for inp, o in zip(inputs, v1_outputs)
        ]

    def evaluate(self, test_df) -> list[EvalMetrics]:
        """Evaluate model on an external test DataFrame."""
        self._ensure_loaded()
        raw_metrics = self._model.evaluate(test_df)
        return [
            EvalMetrics(
                target=m.target,
                rmse=m.rmse,
                rmse_pct=m.rmse_pct,
                mae=m.mae,
                r2=m.r2,
                n_test=m.n_test,
                passes_criterion=m.passes_criterion,
            )
            for m in raw_metrics
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to path."""
        self._ensure_loaded()
        self._model.save(path)
        self.model_path = path

    def load(self, path: str) -> None:
        """Load model from path."""
        from hpe.ai.surrogate.v1_xgboost import SurrogateV1
        v1 = SurrogateV1()
        v1.load(path)
        self._model = v1
        self.model_path = path
        log.info("SurrogateEvaluator.load: loaded from %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Surrogate model not found: {self.model_path}\n"
                "Train it first: SurrogateEvaluator.train_from_bancada()"
            )
        self.load(self.model_path)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load()

    def _estimate_confidence(self, inp: SurrogateInput) -> float:
        """Estimate confidence based on training data coverage.

        Simple heuristic for v1: confidence decreases as Ns and D2 move
        away from the training distribution (Ns ~ 20-50, D2 ~ 200-400mm).

        v2+ will use GP posterior std or ensemble variance instead.
        """
        # Training range priors
        ns_center, ns_std = 35.0, 15.0
        d2_center, d2_std = 300.0, 80.0

        ns_score = 1.0 - min(abs(inp.Ns - ns_center) / (3 * ns_std), 1.0)
        d2_score = 1.0 - min(abs(inp.D2 - d2_center) / (3 * d2_std), 1.0)
        return round((ns_score + d2_score) / 2, 2)
