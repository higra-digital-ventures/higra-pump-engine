"""Ingestion + retrain orchestration — the HPE "data flywheel".

This module closes the loop described as the golden rule of HPE:

    every run (sizing / CFD) MUST be recorded, and the surrogate MUST be
    retrained as new high-quality data accumulates.

It ties together pieces that already existed but were disconnected:

* ``hpe.data.training_log``      — row-level insert into PostgreSQL.
* ``hpe.data.bancada_etl``       — feature engineering (``compute_features``).
* ``hpe.ai.surrogate.v1_xgboost``— the surrogate trainer.

Why this layer is needed
------------------------
1. The version-save auto-log in ``orchestrator/versions.py`` imported a
   non-existent module (``hpe.db.training_log``) with the wrong signature,
   so **no sizing run ever reached the training_log**. ``log_sizing_run``
   here is the correct, graceful replacement.
2. The retrain helpers called ``SurrogateV1().train()`` with **no
   ``features_path``** (a required argument) — so retrain always crashed.
   ``retrain_surrogate`` here calls it correctly.
3. The ETL only reads the read-only bench table (``hgr_lab_reg_teste``),
   so CFD rows logged into ``training_log`` never influenced the model.
   ``build_combined_features`` folds training_log rows into the feature
   matrix, finally closing the flywheel.

Everything degrades gracefully: if PostgreSQL / scikit-learn / xgboost /
mlflow are missing, functions log a warning and return ``None`` instead of
raising, so producers (API routes, Celery tasks) never crash on logging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# CFD data sources count toward the retrain trigger (bench/sizing are weaker).
CFD_FONTES = ("cfd_openfoam", "cfd_su2")

# Marker file storing how many CFD rows existed at the last retrain.
_DATASET_DIR = Path(__file__).resolve().parents[4] / "dataset"
_RETRAIN_MARKER = _DATASET_DIR / ".last_retrain.json"
_COMBINED_FEATURES = _DATASET_DIR / "combined_features.parquet"


# ---------------------------------------------------------------------------
# 1. Logging a run (graceful — never raises)
# ---------------------------------------------------------------------------

def log_run(entry: "Any") -> Optional[str]:
    """Insert a ``TrainingLogEntry`` into the training_log, never raising.

    Returns the new row UUID, or ``None`` if the DB is unavailable.
    """
    try:
        from hpe.data.training_log import insert_entry
        return insert_entry(entry)
    except Exception as exc:  # noqa: BLE001 — logging must never break the caller
        log.warning("ingest.log_run: skipped (training_log unavailable: %s)", exc)
        return None


def log_sizing_run(
    *,
    op: dict,
    sizing: dict,
    qualidade: float = 0.7,
    projeto_id: Optional[str] = None,
    modelo_bomba: Optional[str] = None,
    notas: Optional[str] = None,
) -> Optional[str]:
    """Record a 1D sizing result (correct replacement for the broken auto-log).

    Parameters
    ----------
    op : dict
        Operating point with ``flow_rate`` [m³/s], ``head`` [m], ``rpm``.
    sizing : dict
        Sizing result (``estimated_efficiency``, ``impeller_d2`` [m],
        ``estimated_power`` [W], ``specific_speed_nq``, ...).
    qualidade : float
        Confidence score; 1D sizing is weaker than CFD (default 0.7).
    """
    try:
        from hpe.data.training_log import TrainingLogEntry
    except Exception as exc:  # noqa: BLE001
        log.warning("ingest.log_sizing_run: training_log import failed (%s)", exc)
        return None

    q_m3s = float(op.get("flow_rate", 0.0) or 0.0)
    head = float(op.get("head", 0.0) or 0.0)
    rpm = float(op.get("rpm", op.get("speed", 0.0)) or 0.0)
    nq = float(sizing.get("specific_speed_nq", 0.0) or 0.0)

    entry = TrainingLogEntry(
        fonte="sizing_1d",
        ns=nq * 51.65,
        nq=nq or None,
        d2_mm=float(sizing.get("impeller_d2", 0.0) or 0.0) * 1000.0,
        b2_mm=(float(sizing.get("impeller_b2", 0.0) or 0.0) * 1000.0) or None,
        n_rpm=rpm,
        q_m3h=q_m3s * 3600.0,
        h_m=head,
        eta_total=float(sizing.get("estimated_efficiency", 0.0) or 0.0) * 100.0,
        p_shaft_kw=float(sizing.get("estimated_power", 0.0) or 0.0) / 1000.0,
        npsh_r_m=sizing.get("estimated_npsh_r"),
        beta1_deg=sizing.get("beta1"),
        beta2_deg=sizing.get("beta2"),
        z_palhetas=sizing.get("blade_count"),
        qualidade=qualidade,
        projeto_id=projeto_id,
        modelo_bomba=modelo_bomba,
        notas=notas or "auto-logged via ingest.log_sizing_run",
    )
    return log_run(entry)


# ---------------------------------------------------------------------------
# 2. Building features from training_log (closes the ETL gap)
# ---------------------------------------------------------------------------

def map_training_log_to_features(df: "Any") -> "Any":
    """Map raw training_log rows to the surrogate feature schema.

    Reuses ``bancada_etl.compute_features`` so the derived columns are
    identical to the bench feature matrix the model already trains on.

    Parameters
    ----------
    df : pandas.DataFrame
        training_log rows (columns ``q_m3h``, ``h_m``, ``n_rpm``, ``d2_mm``,
        ``p_shaft_kw``, ``eta_total``, ``n_estagios``/``n_stages``, ...).

    Returns
    -------
    pandas.DataFrame
        Columns = ``ALL_FEATURES + TARGETS`` expected by ``SurrogateV1.train``.
    """
    import pandas as pd
    from hpe.data.bancada_etl import compute_features, FEATURE_COLS, TARGET_COLS

    if df is None or len(df) == 0:
        return pd.DataFrame()

    src = df.copy()

    # Number of stages → head per stage (the bench features use per-stage head).
    if "n_estagios" in src.columns:
        n_stages = pd.to_numeric(src["n_estagios"], errors="coerce").fillna(1).clip(lower=1)
    elif "n_stages" in src.columns:
        n_stages = pd.to_numeric(src["n_stages"], errors="coerce").fillna(1).clip(lower=1)
    else:
        n_stages = pd.Series(1.0, index=src.index)

    # Model group label (used by compute_features for the q_star / h_star ratios).
    if "modelo_bomba" in src.columns:
        modelo = src["modelo_bomba"].fillna("cfd")
    elif "fonte" in src.columns:
        modelo = src["fonte"].fillna("cfd")
    else:
        modelo = pd.Series("cfd", index=src.index)

    eta_total = pd.to_numeric(src["eta_total"], errors="coerce")
    eta_hid = pd.to_numeric(src["eta_hid"], errors="coerce") if "eta_hid" in src.columns else eta_total
    p_kw = pd.to_numeric(src.get("p_shaft_kw", 0.0), errors="coerce").fillna(0.0)

    mapped = pd.DataFrame({
        "q_m3s": pd.to_numeric(src["q_m3h"], errors="coerce") / 3600.0,
        "h_stage_m": pd.to_numeric(src["h_m"], errors="coerce") / n_stages,
        "n_rpm": pd.to_numeric(src["n_rpm"], errors="coerce"),
        "d2_mm": pd.to_numeric(src["d2_mm"], errors="coerce"),
        "p_kw": p_kw,
        "n_stages": n_stages,
        "modelobomba": modelo,
        # Targets
        "eta_total": eta_total,
        "eta_hid": eta_hid.fillna(eta_total),
    })

    feat = compute_features(mapped)
    keep = [c for c in (FEATURE_COLS + TARGET_COLS) if c in feat.columns]
    return feat[keep]


def build_combined_features(
    out_path: Optional[str] = None,
    min_quality: float = 0.5,
    include_bancada: bool = True,
    training_df: "Any" = None,
) -> Optional[Path]:
    """Build the combined feature matrix (bench + training_log) for training.

    This is the missing pipe in the flywheel: it unions the bench-derived
    features (``bancada_features.parquet``) with features computed from the
    accumulated ``training_log`` rows, so CFD runs finally influence the model.

    Parameters
    ----------
    out_path : str, optional
        Where to write the combined parquet (default ``dataset/combined_features.parquet``).
    min_quality : float
        Minimum ``qualidade`` of training_log rows to include.
    include_bancada : bool
        Whether to include the existing bench feature matrix.
    training_df : pandas.DataFrame, optional
        Pre-loaded training_log rows (mainly for testing). If ``None``, the
        rows are pulled from the database via the feature store.

    Returns
    -------
    pathlib.Path | None
        Path of the written parquet, or ``None`` if no data was available.
    """
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        log.warning("ingest.build_combined_features: pandas unavailable (%s)", exc)
        return None

    frames: list[Any] = []

    # Bench features (already engineered + has norm_ columns train() ignores).
    if include_bancada:
        bancada_path = _DATASET_DIR / "bancada_features.parquet"
        if bancada_path.exists():
            try:
                frames.append(pd.read_parquet(bancada_path))
            except Exception as exc:  # noqa: BLE001
                log.warning("ingest: could not read bancada_features (%s)", exc)

    # training_log rows → features.
    if training_df is None:
        try:
            from hpe.data.feature_store import FeatureStore
            training_df = FeatureStore().export_training_log()
        except Exception as exc:  # noqa: BLE001
            log.warning("ingest: training_log export skipped (%s)", exc)
            training_df = None

    if training_df is not None and len(training_df) > 0:
        if "qualidade" in training_df.columns:
            training_df = training_df[training_df["qualidade"] >= min_quality]
        tl_feat = map_training_log_to_features(training_df)
        if len(tl_feat) > 0:
            frames.append(tl_feat)

    if not frames:
        log.warning("ingest.build_combined_features: no data to combine")
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Drop rows missing any target (can't train on them).
    from hpe.data.bancada_etl import TARGET_COLS
    present_targets = [c for c in TARGET_COLS if c in combined.columns]
    combined = combined.dropna(subset=present_targets)

    path = Path(out_path) if out_path else _COMBINED_FEATURES
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    log.info("ingest.build_combined_features: wrote %d rows → %s", len(combined), path)
    return path


# ---------------------------------------------------------------------------
# 3. Retrain (correct call) + count-based trigger
# ---------------------------------------------------------------------------

def retrain_surrogate(min_quality: float = 0.5) -> Optional[dict]:
    """Rebuild combined features and retrain the surrogate v1 (XGBoost).

    Fixes the previous broken helpers that called ``train()`` without the
    required ``features_path`` argument.

    Returns
    -------
    dict | None
        ``{"rmse_pct": ..., "n_samples": ..., "mlflow_run_id": ...}`` on
        success, or ``None`` if data / dependencies were unavailable.
    """
    features_path = build_combined_features(min_quality=min_quality)
    if features_path is None:
        log.warning("ingest.retrain_surrogate: no features available — skipping")
        return None

    try:
        from hpe.ai.surrogate.v1_xgboost import SurrogateV1
        result = SurrogateV1().train(str(features_path))
    except Exception as exc:  # noqa: BLE001
        log.warning("ingest.retrain_surrogate: training failed (%s)", exc)
        return None

    # Update the retrain marker so maybe_retrain() resets its counter.
    _write_retrain_marker(_count_cfd_rows())

    rmse_pct = None
    metrics = getattr(result, "metrics", None) or getattr(result, "r2_scores", None)
    if isinstance(metrics, dict):
        rmse_pct = metrics.get("eta_total_rmse_pct") or next(iter(metrics.values()), None)

    report = {
        "n_samples": getattr(result, "n_samples", None),
        "rmse_pct": rmse_pct,
        "mlflow_run_id": getattr(result, "mlflow_run_id", None),
        "model_path": getattr(result, "model_path", None),
    }
    log.info("ingest.retrain_surrogate: done %s", report)
    return report


def count_new_cfd_since_train() -> int:
    """How many CFD rows were added since the last retrain (0 if unknown)."""
    last = _read_retrain_marker()
    current = _count_cfd_rows()
    if current is None:
        return 0
    return max(0, current - last)


def maybe_retrain(min_new_cfd: int = 10, min_quality: float = 0.5) -> Optional[dict]:
    """Retrain only if enough new CFD rows accumulated since the last train.

    Safe to call after every run: it's a cheap COUNT query plus a threshold
    check; the expensive training only fires when the threshold is crossed.
    """
    new = count_new_cfd_since_train()
    if new < min_new_cfd:
        log.debug("ingest.maybe_retrain: %d/%d new CFD rows — not retraining", new, min_new_cfd)
        return None
    log.info("ingest.maybe_retrain: %d new CFD rows ≥ %d — retraining", new, min_new_cfd)
    return retrain_surrogate(min_quality=min_quality)


# ---------------------------------------------------------------------------
# Internal helpers (marker + counts) — all DB access is graceful
# ---------------------------------------------------------------------------

def _count_cfd_rows() -> Optional[int]:
    """Count CFD-source rows in training_log, or ``None`` if DB unavailable."""
    try:
        from hpe.data.training_log import _connect
        conn = _connect()
        try:
            with conn.cursor() as cur:
                placeholders = ", ".join(["%s"] * len(CFD_FONTES))
                cur.execute(
                    f"SELECT COUNT(*) FROM hpe.training_log WHERE fonte IN ({placeholders})",
                    list(CFD_FONTES),
                )
                return int(cur.fetchone()[0])
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        log.debug("ingest._count_cfd_rows: DB unavailable (%s)", exc)
        return None


def _read_retrain_marker() -> int:
    try:
        data = json.loads(_RETRAIN_MARKER.read_text())
        return int(data.get("cfd_rows", 0))
    except Exception:  # noqa: BLE001
        return 0


def _write_retrain_marker(cfd_rows: Optional[int]) -> None:
    if cfd_rows is None:
        return
    try:
        _RETRAIN_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _RETRAIN_MARKER.write_text(json.dumps({"cfd_rows": cfd_rows}))
    except Exception as exc:  # noqa: BLE001
        log.debug("ingest._write_retrain_marker: could not write marker (%s)", exc)
