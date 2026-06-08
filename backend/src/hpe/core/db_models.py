"""SQLAlchemy ORM models for HPE data persistence.

These models store project data, sizing results, simulation runs,
and performance metrics — forming the foundation for the AI module's
training data and the platform's project history.

Tables:
    projects          — Top-level project container
    operating_points  — Design specifications (Q, H, RPM)
    sizing_results    — 1D meanline sizing outputs
    geometry_versions — Parametric geometry snapshots
    simulation_runs   — CFD or physics evaluation runs
    performance_data  — Extracted performance metrics per run
    curve_points      — H-Q, eta-Q curve data points
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Project(Base):
    """Top-level project — groups all design iterations for one machine."""

    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    machine_type = Column(String(50), nullable=False)  # MachineType enum value
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    operating_points = relationship("OperatingPointRecord", back_populates="project")
    sizing_results = relationship("SizingResultRecord", back_populates="project")
    geometry_versions = relationship("GeometryVersion", back_populates="project")


class OperatingPointRecord(Base):
    """Design operating point specification."""

    __tablename__ = "operating_points"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    label = Column(String(100), nullable=True)  # e.g. "design", "part-load", "overload"

    flow_rate = Column(Float, nullable=False)  # Q [m3/s]
    head = Column(Float, nullable=False)  # H [m]
    rpm = Column(Float, nullable=False)  # n [rev/min]
    fluid_type = Column(String(50), default="water")
    fluid_density = Column(Float, default=998.2)  # rho [kg/m3]
    fluid_viscosity = Column(Float, default=1.003e-3)  # mu [Pa.s]

    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="operating_points")


class SizingResultRecord(Base):
    """Stored 1D meanline sizing result."""

    __tablename__ = "sizing_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    operating_point_id = Column(String(36), ForeignKey("operating_points.id"), nullable=True)

    # Specific speed
    specific_speed_nq = Column(Float, nullable=False)
    impeller_type = Column(String(50), nullable=True)

    # Geometry
    impeller_d2 = Column(Float, nullable=False)  # [m]
    impeller_d1 = Column(Float, nullable=False)
    impeller_b2 = Column(Float, nullable=False)
    blade_count = Column(Integer, nullable=False)
    beta1 = Column(Float, nullable=False)  # [deg]
    beta2 = Column(Float, nullable=False)

    # Performance estimates
    estimated_efficiency = Column(Float, nullable=False)
    estimated_power = Column(Float, nullable=False)  # [W]
    estimated_npsh_r = Column(Float, nullable=False)  # [m]
    sigma = Column(Float, nullable=False)

    # Full data (velocity triangles, meridional profile, warnings)
    velocity_triangles = Column(JSON, nullable=True)
    meridional_profile = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="sizing_results")
    simulation_runs = relationship("SimulationRun", back_populates="sizing_result")


class GeometryVersion(Base):
    """Snapshot of a parametric geometry configuration."""

    __tablename__ = "geometry_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    sizing_result_id = Column(String(36), ForeignKey("sizing_results.id"), nullable=True)

    version = Column(Integer, default=1)
    parameters = Column(JSON, nullable=False)  # RunnerGeometryParams as dict
    file_path_step = Column(String(500), nullable=True)  # Path to STEP file in MinIO
    file_path_stl = Column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="geometry_versions")


class SimulationRun(Base):
    """Record of a simulation (physics or CFD) execution."""

    __tablename__ = "simulation_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sizing_result_id = Column(String(36), ForeignKey("sizing_results.id"), nullable=False)
    geometry_version_id = Column(String(36), ForeignKey("geometry_versions.id"), nullable=True)

    run_type = Column(String(50), nullable=False)  # "physics_1d", "openfoam", "su2"
    status = Column(String(20), default="pending")  # SimulationStatus enum value
    solver_config = Column(JSON, nullable=True)  # Solver settings

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    wall_time_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    sizing_result = relationship("SizingResultRecord", back_populates="simulation_runs")
    performance_data = relationship("PerformanceDataPoint", back_populates="simulation_run")


class PerformanceDataPoint(Base):
    """Performance metrics at a single operating point from a simulation run.

    Multiple data points per run enable storing full H-Q curves.
    This is the primary training data source for AI surrogate models.
    """

    __tablename__ = "performance_data"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    simulation_run_id = Column(String(36), ForeignKey("simulation_runs.id"), nullable=False)

    # Operating condition
    flow_rate = Column(Float, nullable=False)  # Q [m3/s]

    # Performance metrics
    head = Column(Float, nullable=False)  # H [m]
    hydraulic_efficiency = Column(Float, nullable=True)
    volumetric_efficiency = Column(Float, nullable=True)
    mechanical_efficiency = Column(Float, nullable=True)
    total_efficiency = Column(Float, nullable=True)
    power = Column(Float, nullable=True)  # P [W]
    torque = Column(Float, nullable=True)  # T [N.m]
    npsh_required = Column(Float, nullable=True)  # NPSHr [m]
    min_pressure_coefficient = Column(Float, nullable=True)  # Cp_min

    # Source label
    source = Column(String(50), default="physics_1d")  # "physics_1d", "cfd", "test_bench"

    created_at = Column(DateTime, default=datetime.utcnow)

    simulation_run = relationship("SimulationRun", back_populates="performance_data")


class TestBenchRecord(Base):
    """Real test bench measurement data for validation and AI training.

    Maps to HIGRA's sigs.teste_bancada table structure.
    Used to validate sizing/CFD predictions against actual performance.
    """

    __tablename__ = "test_bench_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=True)

    # Machine identification
    machine_model = Column(String(100), nullable=True)
    serial_number = Column(String(100), nullable=True)

    # Operating condition
    flow_rate = Column(Float, nullable=True)  # Q [m3/s]
    head = Column(Float, nullable=True)  # H [m]
    rpm = Column(Float, nullable=True)  # n [rev/min]

    # Measured performance
    measured_efficiency = Column(Float, nullable=True)
    measured_power = Column(Float, nullable=True)  # P [W]
    measured_npsh = Column(Float, nullable=True)  # NPSHr [m]
    measured_vibration = Column(Float, nullable=True)  # [mm/s]

    # Raw data reference
    raw_data = Column(JSON, nullable=True)  # All 91 columns from teste_bancada
    test_date = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
