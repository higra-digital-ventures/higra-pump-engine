"""Design version tracking — persists each pipeline result for audit and retraining.

A ``DesignVersion`` captures the full context of a single design run:
operating point, sizing result, geometry summary, surrogate prediction,
and (once available) CFD performance.

Persistence strategy
--------------------
1. **PostgreSQL training_log** — if ``DATABASE_URL`` is set and the
   ``hpe.db`` module is available, the result is inserted as a
   ``training_log`` row so it feeds back into surrogate retraining.
2. **Local JSON file** (``dataset/design_versions.json``) — always written
   as a human-readable fallback; useful when the database is offline or
   during local development.

Usage
-----
    from hpe.orchestrator.versions import DesignVersion, save_version

    version = DesignVersion.from_sizing(op_dict, sizing_dict,
        geometry_summary=geo_dict.get("params", {}),
        surrogate_prediction=surrogate_dict,
    )
    version_id = save_version(version)
    print("Saved design version:", version_id)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Default local file path for version storage fallback
_DEFAULT_VERSIONS_FILE = str(
    Path(__file__).resolve().parents[5] / "dataset" / "design_versions.json"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DesignVersion:
    """Immutable snapshot of a complete design run.

    Attributes
    ----------
    id : str
        UUID assigned at creation.
    project_id : str | None
        Optional project grouping identifier.
    version_number : int
        Monotonically increasing within a project (or 1 if unknown).
    created_at : str
        ISO-8601 UTC timestamp.
    operating_point : dict
        Serialised OperatingPoint (Q, H, n, fluid, …).
    sizing_result : dict
        Serialised SizingResult from ``run_sizing_task``.
    geometry_summary : dict
        Compact geometry params dict (D2_mm, b2_mm, …).
    surrogate_prediction : dict | None
        Output of ``run_surrogate_task``, or ``None`` if not run.
    cfd_result : dict | None
        Output of ``run_cfd_task``, or ``None`` if not run.
    notes : str
        Free-text notes.
    tags : list[str]
        Arbitrary searchable tags (e.g. ``["validation", "fase-1"]``).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: Optional[str] = None
    version_number: int = 1
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    operating_point: dict = field(default_factory=dict)
    sizing_result: dict = field(default_factory=dict)
    geometry_summary: dict = field(default_factory=dict)
    surrogate_prediction: Optional[dict] = None
    cfd_result: Optional[dict] = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    def to_json(self) -> str:
        """Return a JSON string representation of this version."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_sizing(
        cls,
        op_dict: dict,
        sizing_dict: dict,
        *,
        geometry_summary: Optional[dict] = None,
        surrogate_prediction: Optional[dict] = None,
        cfd_result: Optional[dict] = None,
        project_id: Optional[str] = None,
        notes: str = "",
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "DesignVersion":
        """Construct a DesignVersion from pipeline outputs.

        Parameters
        ----------
        op_dict : dict
            Operating point dict passed to the pipeline.
        sizing_dict : dict
            Output of ``run_sizing_task``.
        geometry_summary : dict, optional
            ``geo_dict.get("params", {})`` from geometry stage.
        surrogate_prediction : dict, optional
            Output of ``run_surrogate_task``.
        cfd_result : dict, optional
            Output of ``run_cfd_task``.
        """
        return cls(
            project_id=project_id,
            operating_point=dict(op_dict),
            sizing_result=dict(sizing_dict),
            geometry_summary=dict(geometry_summary or {}),
            surrogate_prediction=dict(surrogate_prediction) if surrogate_prediction else None,
            cfd_result=dict(cfd_result) if cfd_result else None,
            notes=notes,
            tags=list(tags or []),
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "DesignVersion":
        """Reconstruct a DesignVersion from a previously serialised dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def D2_mm(self) -> Optional[float]:
        """Impeller outlet diameter in mm (from sizing_result)."""
        d2 = self.sizing_result.get("impeller_d2")
        return round(d2 * 1000, 1) if d2 is not None else None

    @property
    def eta(self) -> Optional[float]:
        """Estimated total efficiency (surrogate if available, else sizing)."""
        if self.surrogate_prediction and self.surrogate_prediction.get("eta_total") is not None:
            return self.surrogate_prediction["eta_total"]
        return self.sizing_result.get("estimated_efficiency")

    def __repr__(self) -> str:
        return (
            f"DesignVersion(id={self.id!r}, D2={self.D2_mm}mm, "
            f"eta={self.eta}, created_at={self.created_at!r})"
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_version(
    version: DesignVersion,
    db_url: Optional[str] = None,
    versions_file: Optional[str] = None,
) -> str:
    """Persist a ``DesignVersion`` and return its ID.

    Tries the following backends in order:
    1. PostgreSQL ``training_log`` (via ``hpe.db.training_log``).
    2. Local JSON append file (``dataset/design_versions.json``).

    Parameters
    ----------
    version : DesignVersion
        The design version to persist.
    db_url : str, optional
        Override ``DATABASE_URL`` env var.
    versions_file : str, optional
        Override the default local JSON file path.

    Returns
    -------
    str
        The version's UUID (``version.id``).
    """
    saved_to: list[str] = []

    # ── Attempt 1: PostgreSQL training_log ────────────────────────────
    _try_save_to_db(version, db_url, saved_to)

    # ── Attempt 2: Local JSON file ────────────────────────────────────
    _try_save_to_json(version, versions_file or _DEFAULT_VERSIONS_FILE, saved_to)

    if saved_to:
        log.info("save_version: id=%s saved to [%s]", version.id, ", ".join(saved_to))
    else:
        log.warning("save_version: id=%s — all persistence backends failed", version.id)

    return version.id


def _try_save_to_db(version: DesignVersion, db_url: Optional[str], saved: list[str]) -> None:
    """Insert version into training_log if database is accessible."""
    url = db_url or os.getenv("DATABASE_URL") or os.getenv("HPE_DATABASE_URL")
    if not url:
        log.debug("save_version: DATABASE_URL not set — skipping DB persistence")
        return

    # NOTE: previously this imported ``hpe.db.training_log`` (a module that
    # does not exist) with a mismatched signature, so it silently failed on
    # every save. It now uses the central ingestion layer, which maps the
    # version to a TrainingLogEntry and inserts gracefully.
    from hpe.data.ingest import log_sizing_run

    row_id = log_sizing_run(
        op=version.operating_point,
        sizing=version.sizing_result,
        projeto_id=version.project_id,
        notas=version.notes,
    )
    if row_id:
        saved.append("postgresql:training_log")
    else:
        log.debug("save_version: training_log insert skipped (DB unavailable)")


def _try_save_to_json(version: DesignVersion, file_path: str, saved: list[str]) -> None:
    """Append version to a local JSON lines file."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing records
        records: list[dict] = []
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    records = json.load(fh)
                if not isinstance(records, list):
                    records = []
            except (json.JSONDecodeError, OSError):
                records = []

        records.append(version.to_dict())

        with path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)

        saved.append(f"json:{file_path}")
        log.debug("save_version: appended to %s (%d total records)", file_path, len(records))
    except Exception as exc:
        log.warning("save_version: JSON write failed (%s)", exc)


def load_versions(
    versions_file: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 100,
) -> list[DesignVersion]:
    """Load DesignVersion records from the local JSON file.

    Parameters
    ----------
    versions_file : str, optional
        Override default file path.
    project_id : str, optional
        Filter by project.
    limit : int
        Maximum number of records to return (most recent first).

    Returns
    -------
    list[DesignVersion]
    """
    file_path = versions_file or _DEFAULT_VERSIONS_FILE
    path = Path(file_path)

    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as fh:
            records: list[dict] = json.load(fh)
    except Exception as exc:
        log.warning("load_versions: cannot read %s — %s", file_path, exc)
        return []

    versions = []
    for record in reversed(records):  # most recent first
        try:
            v = DesignVersion.from_dict(record)
            if project_id is None or v.project_id == project_id:
                versions.append(v)
                if len(versions) >= limit:
                    break
        except Exception:
            continue

    return versions
