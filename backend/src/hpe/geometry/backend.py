"""Pluggable geometry backends — classic (parametric) vs free (implicit).

The two design paradigms (see :class:`hpe.core.enums.DesignMode`) share the
whole HPE spine — intent, sizing, physics, CFD, surrogate — and differ in
exactly **one** stage: how the geometry is produced. This module is that seam.

* :class:`ParametricBackend` wraps the existing parametric pipeline
  (``hpe.geometry.parametric.run_geometry`` + ``export``/``storage``). It is a
  thin adapter — no behaviour changes.
* :class:`ImplicitBackend` is the placeholder for the free/generative
  (SDF/voxel) backend. It advertises its planned capabilities but raises
  ``NotImplementedError`` until the implicit kernel lands.

A registry maps a :class:`DesignMode` (or backend name) to a backend instance,
so callers (API routes, the CEM facade, the CFD pipeline) can stay agnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from hpe.core.enums import DesignMode


# ---------------------------------------------------------------------------
# Capability + artifact contracts
# ---------------------------------------------------------------------------

@dataclass
class GeometryCapabilities:
    """What a backend can produce — used by the UI to enable/disable features."""

    supports_step: bool
    supports_stl: bool
    supports_voxel: bool
    supports_internal_channels: bool
    mesh_strategy: str  # "body_fitted" | "cut_cell"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeometryArtifact:
    """Backend-agnostic geometry result that flows into the DesignState."""

    backend: str
    params: dict = field(default_factory=dict)
    meridional: dict = field(default_factory=dict)
    blade: dict = field(default_factory=dict)
    cad_available: bool = False
    step_path: Optional[str] = None
    step_url: Optional[str] = None
    stl_url: Optional[str] = None
    mesh_strategy: str = "body_fitted"
    warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GeometryBackend(Protocol):
    """Common interface every geometry backend implements."""

    name: str
    capabilities: GeometryCapabilities

    def generate(
        self,
        sizing: Any,
        *,
        output_dir: Optional[str] = None,
        export: bool = False,
        run_id: Optional[str] = None,
    ) -> GeometryArtifact:
        """Produce geometry for the given sizing result."""
        ...


# ---------------------------------------------------------------------------
# Parametric backend (wraps the existing pipeline)
# ---------------------------------------------------------------------------

class ParametricBackend:
    """Classic B-rep geometry via CadQuery/OCCT (existing pipeline)."""

    name = "parametric"
    capabilities = GeometryCapabilities(
        supports_step=True,
        supports_stl=True,
        supports_voxel=False,
        supports_internal_channels=False,
        mesh_strategy="body_fitted",
    )

    def generate(
        self,
        sizing: Any,
        *,
        output_dir: Optional[str] = None,
        export: bool = False,
        run_id: Optional[str] = None,
    ) -> GeometryArtifact:
        from hpe.geometry.parametric import run_geometry

        geo = run_geometry(sizing)
        d = geo.to_dict()

        artifact = GeometryArtifact(
            backend=self.name,
            params=d.get("params", {}),
            meridional=d.get("meridional", {}),
            blade=d.get("blade", {}),
            cad_available=bool(d.get("cad_available", False)),
            step_path=d.get("step_path"),
            mesh_strategy=self.capabilities.mesh_strategy,
            warnings=list(d.get("warnings", [])),
        )

        # Optional 3D export + upload — both are no-ops when deps are missing.
        if export and output_dir:
            self._export(sizing, output_dir, run_id, artifact)
        return artifact

    @staticmethod
    def _export(sizing: Any, output_dir: str, run_id: Optional[str], artifact: GeometryArtifact) -> None:
        import uuid
        from pathlib import Path
        try:
            from hpe.geometry.export import export_runner_3d
            from hpe.geometry.storage import upload_geometry_files

            cad = export_runner_3d(sizing, output_dir=Path(output_dir))
            if getattr(cad, "available", False):
                artifact.cad_available = True
                if cad.step_path:
                    artifact.step_path = str(cad.step_path)
                upload = upload_geometry_files(
                    run_id=run_id or uuid.uuid4().hex,
                    step_path=cad.step_path,
                    stl_path=cad.stl_path,
                )
                if getattr(upload, "available", False):
                    artifact.step_url = upload.step_url
                    artifact.stl_url = upload.stl_url
        except Exception as exc:  # noqa: BLE001 — export is best-effort
            artifact.warnings.append(f"3D export/upload skipped: {exc}")


# ---------------------------------------------------------------------------
# Implicit backend (free/generative — placeholder)
# ---------------------------------------------------------------------------

class ImplicitBackend:
    """Free/generative geometry via implicit SDF/voxel fields (planned).

    Advertises its planned capabilities so the UI and CFD pipeline can already
    branch on them, but generation is not implemented yet (see roadmap M3+).
    """

    name = "implicit"
    capabilities = GeometryCapabilities(
        supports_step=False,
        supports_stl=True,
        supports_voxel=True,
        supports_internal_channels=True,
        mesh_strategy="cut_cell",
    )

    def generate(
        self,
        sizing: Any,
        *,
        output_dir: Optional[str] = None,
        export: bool = False,
        run_id: Optional[str] = None,
    ) -> GeometryArtifact:
        raise NotImplementedError(
            "Implicit (free/generative) geometry backend is not implemented yet "
            "(roadmap M3). Use DesignMode.CLASSIC for now."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, GeometryBackend] = {}

# DesignMode → backend name.
_MODE_TO_BACKEND = {
    DesignMode.CLASSIC: "parametric",
    DesignMode.FREE: "implicit",
}


def register_backend(backend: GeometryBackend) -> None:
    """Register (or override) a geometry backend by its ``name``."""
    _BACKENDS[backend.name] = backend


def get_backend(mode_or_name: Any) -> GeometryBackend:
    """Resolve a backend from a :class:`DesignMode`, its value, or a name.

    Raises
    ------
    KeyError
        If no backend matches.
    """
    if isinstance(mode_or_name, DesignMode):
        name = _MODE_TO_BACKEND[mode_or_name]
    elif mode_or_name in _MODE_TO_BACKEND.values():
        name = mode_or_name
    else:
        # Accept the string value of a DesignMode ("classic" / "free").
        try:
            name = _MODE_TO_BACKEND[DesignMode(mode_or_name)]
        except ValueError:
            name = str(mode_or_name)
    if name not in _BACKENDS:
        raise KeyError(f"No geometry backend registered for {mode_or_name!r} (resolved {name!r})")
    return _BACKENDS[name]


def list_backends() -> list[dict]:
    """List registered backends and their capabilities (for the UI)."""
    mode_by_name = {v: k.value for k, v in _MODE_TO_BACKEND.items()}
    return [
        {
            "name": b.name,
            "mode": mode_by_name.get(b.name),
            "capabilities": b.capabilities.to_dict(),
        }
        for b in _BACKENDS.values()
    ]


# Register the built-in backends on import.
register_backend(ParametricBackend())
register_backend(ImplicitBackend())
