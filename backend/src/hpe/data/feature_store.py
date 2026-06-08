"""Feature store — centralised access to HPE ML datasets.

Provides a single interface to read, append and refresh the Parquet
feature datasets produced by the ETL pipeline.

Datasets managed
----------------
bancada_features.parquet   Normalised features from test bench (ETL output)
bancada_raw.parquet        Raw SI-unit columns from test bench
training_log.parquet       Exported snapshot of hpe.training_log (optional)

Usage
-----
    from hpe.data.feature_store import FeatureStore

    fs = FeatureStore()
    df = fs.load_bancada()          # returns full feature DataFrame
    sample = fs.sample(n=500)       # random sample for quick experiments
    fs.refresh_from_db()            # re-runs ETL and overwrites parquet
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# Default dataset directory (override with HPE_DATASET_DIR env var)
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[5] / "dataset"

BANCADA_FEATURES = "bancada_features.parquet"
BANCADA_RAW      = "bancada_raw.parquet"
TRAINING_SNAPSHOT= "training_log_snapshot.parquet"


class FeatureStore:
    """Centralised access to HPE ML feature datasets.

    Parameters
    ----------
    dataset_dir : str or Path, optional
        Directory containing Parquet files.
        Defaults to HPE_DATASET_DIR env var or ``<repo>/dataset/``.
    """

    def __init__(self, dataset_dir: Optional[str | Path] = None):
        self.dataset_dir = Path(
            dataset_dir or os.getenv("HPE_DATASET_DIR", str(DEFAULT_DATASET_DIR))
        )
        self._cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_bancada(
        self,
        normalised: bool = True,
        refresh_cache: bool = False,
    ) -> pd.DataFrame:
        """Load test bench feature dataset.

        Parameters
        ----------
        normalised : bool
            If True, include ``norm_*`` columns (StandardScaler output).
        refresh_cache : bool
            Force reload from disk even if cached.

        Returns
        -------
        pd.DataFrame
            Feature matrix with physics features and targets.
        """
        key = f"bancada_{'norm' if normalised else 'raw'}"
        if key not in self._cache or refresh_cache:
            path = self.dataset_dir / BANCADA_FEATURES
            if not path.exists():
                raise FileNotFoundError(
                    f"Feature file not found: {path}\n"
                    "Run bancada_etl.py first to generate the dataset."
                )
            df = pd.read_parquet(path)
            if not normalised:
                df = df[[c for c in df.columns if not c.startswith("norm_")]]
            self._cache[key] = df
            log.info("feature_store: loaded %s (%d rows, %d cols)", BANCADA_FEATURES,
                     len(df), len(df.columns))
        return self._cache[key]

    def load_raw(self) -> pd.DataFrame:
        """Load raw (SI-unit) columns from test bench."""
        path = self.dataset_dir / BANCADA_RAW
        if not path.exists():
            raise FileNotFoundError(f"Raw file not found: {path}")
        return pd.read_parquet(path)

    def load_training_snapshot(self) -> pd.DataFrame:
        """Load snapshot of hpe.training_log exported as Parquet.

        Returns empty DataFrame if snapshot does not exist yet.
        """
        path = self.dataset_dir / TRAINING_SNAPSHOT
        if not path.exists():
            log.warning("feature_store: training_log snapshot not found — returning empty df")
            return pd.DataFrame()
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int = 500,
        random_state: int = 42,
        source: str = "bancada",
    ) -> pd.DataFrame:
        """Return a random sample from the feature store.

        Parameters
        ----------
        n : int
            Number of rows (capped at dataset size).
        random_state : int
            Reproducibility seed.
        source : str
            'bancada' | 'training_log'.
        """
        if source == "bancada":
            df = self.load_bancada()
        elif source == "training_log":
            df = self.load_training_snapshot()
        else:
            raise ValueError(f"Unknown source: {source!r}")

        n = min(n, len(df))
        return df.sample(n=n, random_state=random_state).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Info / stats
    # ------------------------------------------------------------------

    def info(self) -> dict:
        """Return metadata about available datasets."""
        result = {}
        for name, fname in [
            ("bancada_features", BANCADA_FEATURES),
            ("bancada_raw", BANCADA_RAW),
            ("training_snapshot", TRAINING_SNAPSHOT),
        ]:
            path = self.dataset_dir / fname
            if path.exists():
                size_kb = path.stat().st_size / 1024
                try:
                    df = pd.read_parquet(path, columns=["id"])
                    rows = len(df)
                except Exception:
                    rows = "?"
                result[name] = {"exists": True, "rows": rows, "size_kb": round(size_kb, 1)}
            else:
                result[name] = {"exists": False}
        return result

    # ------------------------------------------------------------------
    # ETL refresh
    # ------------------------------------------------------------------

    def refresh_from_db(self) -> dict:
        """Re-run the full ETL pipeline and overwrite Parquet files.

        Returns
        -------
        dict
            ETL quality report.
        """
        log.info("feature_store: triggering ETL refresh...")
        from hpe.data.bancada_etl import run_etl  # lazy import
        report = run_etl(dry_run=False)
        self._cache.clear()
        log.info("feature_store: cache cleared after ETL refresh")
        return report

    def export_training_log(self) -> pd.DataFrame:
        """Export hpe.training_log table to Parquet snapshot.

        Returns
        -------
        pd.DataFrame
            Exported rows.
        """
        from psycopg2.extras import RealDictCursor
        from hpe.data.training_log import _connect

        conn = _connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM hpe.training_log ORDER BY created_at")
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            log.warning("feature_store: training_log is empty")
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        path = self.dataset_dir / TRAINING_SNAPSHOT
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("feature_store: training_log snapshot saved (%d rows)", len(df))
        return df
