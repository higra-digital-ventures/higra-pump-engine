"""ETL pipeline for HIGRA test bench data.

Reads from ``hgr_lab_reg_teste`` (higra_sigs database), computes
physics-informed features and saves normalised Parquet datasets.

Source table
------------
- host    : higra_sigs PostgreSQL
- table   : public.hgr_lab_reg_teste
- records : ~4 165 rows × 91 columns (bench test measurements)

Output
------
dataset/
  bancada_raw.parquet          Raw selected columns (SI units)
  bancada_features.parquet     Normalised feature matrix for ML
  etl_report.json              Data quality report

Usage
-----
    python -m hpe.data.bancada_etl            # uses DATABASE_SIGS_URL from .env
    python -m hpe.data.bancada_etl --dry-run  # just print report, no files

Notes
-----
- Idempotent: safe to re-run; output files are overwritten.
- All continuous features normalised with StandardScaler (stored alongside).
- Features prefixed ``feat_`` in the output parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import StandardScaler
import joblib

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.80665  # m/s²

# Source database (higra_sigs)
SIGS_DB_URL = os.getenv(
    "DATABASE_SIGS_URL",
    "postgresql://postgres:higra123@localhost:5432/higra_sigs",
)

SOURCE_TABLE = "hgr_lab_reg_teste"

# Output directory (relative to project root or absolute via env)
DATASET_DIR = Path(
    os.getenv("HPE_DATASET_DIR", str(Path(__file__).resolve().parents[5] / "dataset"))
)

# Columns to extract from source table
RAW_COLS = [
    # --- Operating point ---
    "id",
    "e3timestamp",
    "modelobomba",
    "tipoderotor",
    "qntdeestag",
    # Flow
    "vazm3h",        # m³/h  → will convert to m³/s
    "vazao",         # raw reading
    "vazlps",        # L/s
    # Head / pressure
    "pressaototal",  # mca (m)
    "pressao",       # raw kPa
    "pressaosuccao", # mca
    "pressaodescarga",
    # Speed
    "rotacao",       # rpm (set)
    "rotacaomedida", # rpm (measured)
    # Impeller geometry
    "diarotor",      # mm (varchar → parse)
    "diarotorinter", # mm (for multi-stage intermediate)
    # Performance targets
    "rendbomba",     # % total pump efficiency
    "rendhidroenerg",# % hydraulic efficiency
    "potmecancia",   # kW shaft power
    "potkw",         # kW electrical
    # Motor
    "tensao",
    "corrente",
    "frequencia",
    "cosfi",
    "rendmotor",     # % motor efficiency
    # Temperatures / vibration (secondary)
    "tempaxial",
    "tempradial",
    "vibaxial",
    "vibradial",
    # Approval / quality
    "aprovacao",
    "aprovcalc",
    "norma",
]

# Minimum non-null threshold for a row to be kept
REQUIRED_COLS = ["vazm3h", "pressaototal", "rotacao", "rendbomba", "diarotor"]

# Outlier bounds (physics-based)
BOUNDS: dict[str, tuple[float, float]] = {
    "q_m3s":     (0.0001, 10.0),      # m³/s
    "h_m":       (0.5,    350.0),     # m
    "n_rpm":     (500.0,  3600.0),    # rpm
    "eta_total": (5.0,    95.0),      # %
    "d2_mm":     (50.0,   800.0),     # mm
    "p_kw":      (0.1,    5000.0),    # kW
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _connect(url: str) -> psycopg2.extensions.connection:
    """Create psycopg2 connection from postgres URL."""
    import re
    m = re.match(
        r"postgresql://(?P<user>[^:]+):(?P<pwd>[^@]+)@(?P<host>[^:/]+)"
        r"(?::(?P<port>\d+))?/(?P<db>.+)",
        url,
    )
    if not m:
        raise ValueError(f"Cannot parse DATABASE_SIGS_URL: {url}")
    return psycopg2.connect(
        host=m.group("host"),
        port=int(m.group("port") or 5432),
        user=m.group("user"),
        password=m.group("pwd"),
        dbname=m.group("db"),
    )


def extract_raw(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Pull raw columns from source table."""
    cols_sql = ", ".join(RAW_COLS)
    query = f"SELECT {cols_sql} FROM {SOURCE_TABLE} ORDER BY id"
    log.info("extract_raw: executing query on %s", SOURCE_TABLE)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    log.info("extract_raw: %d rows extracted", len(df))
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _parse_diarotor(val: Any) -> float | None:
    """Parse diarotor column (varchar like '351', '360/125', '360') → float mm."""
    if val is None:
        return None
    s = str(val).strip()
    # Take first number before '/' for multi-stage
    part = s.split("/")[0].strip()
    try:
        return float(part)
    except ValueError:
        return None


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean, type-cast and filter raw dataframe.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe in SI units.
    report : dict
        Data quality metrics.
    """
    report: dict[str, Any] = {"source_rows": len(df)}
    df = df.copy()

    # --- Parse diarotor ---
    df["d2_mm"] = df["diarotor"].apply(_parse_diarotor)

    # --- Unit conversions ---
    df["q_m3s"]    = pd.to_numeric(df["vazm3h"], errors="coerce") / 3600.0
    df["h_m"]      = pd.to_numeric(df["pressaototal"], errors="coerce")
    df["n_rpm"]    = pd.to_numeric(df["rotacao"], errors="coerce")
    df["eta_total"] = pd.to_numeric(df["rendbomba"], errors="coerce")
    df["eta_hid"]  = pd.to_numeric(df["rendhidroenerg"], errors="coerce")
    df["p_kw"]     = pd.to_numeric(df["potmecancia"], errors="coerce")
    df["p_elec_kw"]= pd.to_numeric(df["potkw"], errors="coerce")
    df["n_stages"] = pd.to_numeric(df["qntdeestag"], errors="coerce").fillna(1).astype(int)

    # --- Single-stage head (divide by stages) ---
    df["h_stage_m"] = df["h_m"] / df["n_stages"].clip(lower=1)

    # --- Drop rows missing required fields ---
    before = len(df)
    df = df.dropna(subset=["q_m3s", "h_m", "n_rpm", "eta_total", "d2_mm"])
    report["dropped_missing_required"] = before - len(df)

    # --- Physics-based outlier removal ---
    report["dropped_outliers"] = {}
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        mask = (df[col] >= lo) & (df[col] <= hi)
        n_dropped = (~mask).sum()
        if n_dropped:
            report["dropped_outliers"][col] = int(n_dropped)
            df = df[mask]

    report["clean_rows"] = len(df)
    report["retention_pct"] = round(len(df) / report["source_rows"] * 100, 1)

    log.info(
        "clean: %d/%d rows retained (%.1f%%)",
        len(df), report["source_rows"], report["retention_pct"],
    )
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physics-informed derived features.

    All features prefixed ``feat_`` to distinguish from raw columns.

    Features
    --------
    feat_ns      : Specific speed (dimensional, rpm, m³/s, m)
    feat_nq      : Specific speed (European, = Ns / 51.65)
    feat_d2_m    : Impeller diameter [m]
    feat_u2      : Tip speed u2 = π · D2 · n / 60  [m/s]
    feat_psi     : Head coefficient  g·H / u2²  [-]
    feat_phi     : Flow coefficient  Q / (u2 · (D2/2)²·π)  [-]
    feat_re      : Rotor Reynolds  u2·D2 / ν  [-]  (ν=1e-6 m²/s water 20°C)
    feat_p_spec  : Specific power  P / (ρ·g·Q·H)  [-]  (= 1/η)
    feat_nstages : Number of stages
    feat_q_star  : Relative flow (Q / Q_mean per model group)
    feat_h_star  : Relative head (H / H_mean per model group)
    """
    df = df.copy()
    eps = 1e-9

    d2 = df["d2_mm"] / 1000.0          # m
    q  = df["q_m3s"]
    h  = df["h_stage_m"]               # per-stage head
    n  = df["n_rpm"]
    p  = df["p_kw"] * 1000.0           # W

    df["feat_d2_m"]  = d2
    df["feat_u2"]    = np.pi * d2 * n / 60.0

    u2 = df["feat_u2"].clip(lower=eps)

    df["feat_ns"]    = n * q**0.5 / (h + eps)**0.75
    df["feat_nq"]    = df["feat_ns"] / 51.65
    df["feat_psi"]   = G * h / (u2**2 + eps)
    df["feat_phi"]   = q / (u2 * (np.pi / 4) * d2**2 + eps)
    df["feat_re"]    = u2 * d2 / 1e-6
    df["feat_p_spec"]= p / (1000 * G * q * h + eps)
    df["feat_nstages"] = df["n_stages"].astype(float)

    # Relative Q and H (within model group)
    df["feat_q_star"] = 1.0
    df["feat_h_star"] = 1.0
    for model, grp in df.groupby("modelobomba"):
        q_mean = grp["q_m3s"].median()
        h_mean = grp["h_stage_m"].median()
        mask = df["modelobomba"] == model
        df.loc[mask, "feat_q_star"] = df.loc[mask, "q_m3s"] / (q_mean + eps)
        df.loc[mask, "feat_h_star"] = df.loc[mask, "h_stage_m"] / (h_mean + eps)

    return df


FEATURE_COLS = [
    "feat_ns", "feat_nq", "feat_d2_m", "feat_u2",
    "feat_psi", "feat_phi", "feat_re", "feat_p_spec",
    "feat_nstages", "feat_q_star", "feat_h_star",
    # Raw inputs (non-normalised but included for reference)
    "q_m3s", "h_stage_m", "n_rpm", "d2_mm",
]

TARGET_COLS = ["eta_total", "eta_hid", "p_kw"]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame, scaler_path: Path) -> pd.DataFrame:
    """Fit StandardScaler on FEATURE_COLS and add ``norm_`` columns."""
    feat_df = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_df)
    norm_df = pd.DataFrame(
        scaled,
        columns=[f"norm_{c}" for c in FEATURE_COLS],
        index=df.index,
    )
    joblib.dump(scaler, scaler_path)
    log.info("normalise: scaler saved to %s", scaler_path)
    return pd.concat([df, norm_df], axis=1)


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def build_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame, clean_meta: dict) -> dict:
    """Generate data quality report."""
    feat_cols_present = [c for c in FEATURE_COLS if c in df_clean.columns]

    # Distribution stats for key columns
    dist: dict[str, dict] = {}
    for col in TARGET_COLS + ["feat_ns", "feat_nq", "q_m3s", "h_stage_m"]:
        if col in df_clean.columns:
            s = df_clean[col].describe()
            dist[col] = {k: round(float(v), 4) for k, v in s.items()}

    # Null counts in raw
    null_pct = {
        col: round(df_raw[col].isna().mean() * 100, 1)
        for col in RAW_COLS if col in df_raw.columns
    }
    top_nulls = dict(sorted(null_pct.items(), key=lambda x: -x[1])[:15])

    # Nq distribution (impeller type classification by specific speed)
    # feat_ns = n*sqrt(Q)/H^0.75 with SI units → European Nq scale (10–300)
    # feat_nq = feat_ns / 51.65 → dimensionless omega_s (~0.1–6)
    # Use feat_ns for the bins (correct dimensional scale).
    nq_bins = pd.cut(
        df_clean["feat_ns"].clip(1, 300),
        bins=[0, 25, 60, 100, 150, 300],
        labels=["radial_hp", "radial", "mixed", "semi_axial", "axial"],
    ).value_counts().to_dict()
    nq_dist = {str(k): int(v) for k, v in nq_bins.items()}

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "source_table": SOURCE_TABLE,
        **clean_meta,
        "feature_columns": len(feat_cols_present),
        "target_columns": TARGET_COLS,
        "distributions": dist,
        "top_null_pct": top_nulls,
        "nq_distribution": nq_dist,
        "models_count": int(df_clean["modelobomba"].nunique()),
        "top_models": df_clean["modelobomba"].value_counts().head(10).to_dict(),
    }


# ---------------------------------------------------------------------------
# Main ETL pipeline
# ---------------------------------------------------------------------------

def run_etl(dry_run: bool = False) -> dict:
    """Execute full ETL pipeline.

    Parameters
    ----------
    dry_run : bool
        If True, compute everything but do not write files.

    Returns
    -------
    dict
        Quality report.
    """
    log.info("=== HPE Bancada ETL started ===")
    log.info("Source: %s @ %s", SOURCE_TABLE, SIGS_DB_URL.split("@")[-1])
    log.info("Output: %s", DATASET_DIR)

    # 1. Connect & extract
    conn = _connect(SIGS_DB_URL)
    try:
        df_raw = extract_raw(conn)
    finally:
        conn.close()

    # 2. Clean
    df_clean, clean_meta = clean(df_raw)

    # 3. Feature engineering
    df_feat = compute_features(df_clean)

    # 4. Normalise
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = DATASET_DIR / "scaler.joblib"
    df_final = normalise(df_feat, scaler_path)

    # 5. Build report
    report = build_report(df_raw, df_feat, clean_meta)

    if not dry_run:
        # Save raw (SI units, selected columns)
        raw_save_cols = (
            ["id", "e3timestamp", "modelobomba", "tipoderotor", "n_stages"]
            + ["q_m3s", "h_m", "h_stage_m", "n_rpm", "d2_mm",
               "eta_total", "eta_hid", "p_kw", "p_elec_kw"]
        )
        raw_save = df_final[[c for c in raw_save_cols if c in df_final.columns]]
        raw_path = DATASET_DIR / "bancada_raw.parquet"
        raw_save.to_parquet(raw_path, index=False)
        log.info("Saved raw: %s (%d rows)", raw_path, len(raw_save))

        # Save features (normalised)
        feat_save_cols = (
            ["id", "modelobomba"]
            + FEATURE_COLS
            + [f"norm_{c}" for c in FEATURE_COLS]
            + TARGET_COLS
        )
        feat_save = df_final[[c for c in feat_save_cols if c in df_final.columns]]
        feat_path = DATASET_DIR / "bancada_features.parquet"
        feat_save.to_parquet(feat_path, index=False)
        log.info("Saved features: %s (%d rows, %d cols)", feat_path, len(feat_save), len(feat_save.columns))

        # Save report
        report_path = DATASET_DIR / "etl_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info("Saved report: %s", report_path)

    # Print summary
    log.info("=== ETL Summary ===")
    log.info("Source rows  : %d", report["source_rows"])
    log.info("Clean rows   : %d", report["clean_rows"])
    log.info("Retention    : %.1f%%", report["retention_pct"])
    log.info("Models found : %d", report["models_count"])
    log.info("Nq dist      : %s", report["nq_distribution"])
    log.info("=== ETL complete ===")
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="HPE Bancada ETL")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
    args = parser.parse_args()

    report = run_etl(dry_run=args.dry_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
