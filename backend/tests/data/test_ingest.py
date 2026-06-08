"""Tests for the data-flywheel ingestion layer (hpe.data.ingest).

These run WITHOUT a live PostgreSQL or CFD solver: DB access is monkeypatched
and feature building is exercised with synthetic DataFrames.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hpe.data import ingest
from hpe.data.bancada_etl import FEATURE_COLS, TARGET_COLS


def _synthetic_training_log(n: int = 6) -> pd.DataFrame:
    """A minimal training_log-shaped frame (CFD source)."""
    return pd.DataFrame({
        "q_m3h": [180.0 + 10 * i for i in range(n)],
        "h_m": [30.0 + i for i in range(n)],
        "n_rpm": [1750.0] * n,
        "d2_mm": [320.0 + i for i in range(n)],
        "p_shaft_kw": [48.0 + i for i in range(n)],
        "eta_total": [78.0 + i for i in range(n)],
        "eta_hid": [82.0 + i for i in range(n)],
        "n_estagios": [1] * n,
        "modelo_bomba": [f"MB-{i % 2}" for i in range(n)],
        "fonte": ["cfd_openfoam"] * n,
        "qualidade": [0.9] * n,
    })


# ---------------------------------------------------------------------------
# log_run / log_sizing_run — graceful, never raises
# ---------------------------------------------------------------------------

def test_log_run_returns_none_when_db_unavailable(monkeypatch):
    def _boom(_entry):
        raise RuntimeError("no database")
    monkeypatch.setattr("hpe.data.training_log.insert_entry", _boom)
    assert ingest.log_run(object()) is None  # swallowed, not raised


def test_log_run_returns_row_id_on_success(monkeypatch):
    monkeypatch.setattr("hpe.data.training_log.insert_entry", lambda _e: "uuid-123")
    assert ingest.log_run(object()) == "uuid-123"


def test_log_sizing_run_maps_units_correctly(monkeypatch):
    captured = {}

    def _capture(entry):
        captured["entry"] = entry
        return "row-1"

    monkeypatch.setattr("hpe.data.training_log.insert_entry", _capture)

    row_id = ingest.log_sizing_run(
        op={"flow_rate": 0.05, "head": 30.0, "rpm": 1750},
        sizing={
            "specific_speed_nq": 35.0,
            "impeller_d2": 0.32,         # m
            "estimated_efficiency": 0.78,  # fraction
            "estimated_power": 48000.0,    # W
            "estimated_npsh_r": 3.2,
        },
    )
    assert row_id == "row-1"
    e = captured["entry"]
    assert e.fonte == "sizing_1d"
    assert e.q_m3h == pytest.approx(180.0)        # 0.05 * 3600
    assert e.d2_mm == pytest.approx(320.0)        # 0.32 * 1000
    assert e.eta_total == pytest.approx(78.0)     # 0.78 * 100
    assert e.p_shaft_kw == pytest.approx(48.0)    # 48000 / 1000


# ---------------------------------------------------------------------------
# Feature mapping (the ETL-gap closer) — pure pandas
# ---------------------------------------------------------------------------

def test_map_training_log_to_features_schema_and_rows():
    df = _synthetic_training_log(6)
    feat = ingest.map_training_log_to_features(df)

    assert len(feat) == 6
    # Every column the surrogate trainer expects must be present.
    for col in FEATURE_COLS + TARGET_COLS:
        assert col in feat.columns, f"missing feature column {col}"
    # Sanity on a derived feature: specific speed must be positive.
    assert (feat["feat_ns"] > 0).all()


def test_map_training_log_to_features_empty():
    assert ingest.map_training_log_to_features(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# build_combined_features — union without touching the DB or bancada parquet
# ---------------------------------------------------------------------------

def test_build_combined_features_from_injected_rows(tmp_path):
    out = tmp_path / "combined.parquet"
    path = ingest.build_combined_features(
        out_path=str(out),
        include_bancada=False,
        training_df=_synthetic_training_log(6),
    )
    assert path is not None and path.exists()
    written = pd.read_parquet(path)
    assert len(written) == 6
    for col in TARGET_COLS:
        assert col in written.columns


def test_build_combined_features_filters_low_quality(tmp_path):
    df = _synthetic_training_log(4)
    df.loc[:1, "qualidade"] = 0.1  # two rows below threshold
    path = ingest.build_combined_features(
        out_path=str(tmp_path / "c.parquet"),
        include_bancada=False,
        min_quality=0.5,
        training_df=df,
    )
    assert path is not None
    assert len(pd.read_parquet(path)) == 2


def test_build_combined_features_returns_none_without_data(tmp_path):
    assert ingest.build_combined_features(
        out_path=str(tmp_path / "x.parquet"),
        include_bancada=False,
        training_df=pd.DataFrame(),
    ) is None


# ---------------------------------------------------------------------------
# maybe_retrain — count-based threshold
# ---------------------------------------------------------------------------

def test_maybe_retrain_below_threshold_does_not_train(monkeypatch):
    monkeypatch.setattr(ingest, "count_new_cfd_since_train", lambda: 3)
    called = {"n": 0}
    monkeypatch.setattr(ingest, "retrain_surrogate", lambda **_: called.__setitem__("n", called["n"] + 1))
    assert ingest.maybe_retrain(min_new_cfd=10) is None
    assert called["n"] == 0


def test_maybe_retrain_at_threshold_triggers(monkeypatch):
    monkeypatch.setattr(ingest, "count_new_cfd_since_train", lambda: 12)
    monkeypatch.setattr(ingest, "retrain_surrogate", lambda **_: {"rmse_pct": 4.2})
    report = ingest.maybe_retrain(min_new_cfd=10)
    assert report == {"rmse_pct": 4.2}


def test_count_new_cfd_uses_marker(monkeypatch):
    monkeypatch.setattr(ingest, "_read_retrain_marker", lambda: 100)
    monkeypatch.setattr(ingest, "_count_cfd_rows", lambda: 137)
    assert ingest.count_new_cfd_since_train() == 37


def test_count_new_cfd_zero_when_db_unavailable(monkeypatch):
    monkeypatch.setattr(ingest, "_read_retrain_marker", lambda: 100)
    monkeypatch.setattr(ingest, "_count_cfd_rows", lambda: None)
    assert ingest.count_new_cfd_since_train() == 0
