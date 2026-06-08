"""Helpers to persist domain objects to the database.

Converts between in-memory dataclasses (SizingResult, PerformanceMetrics)
and SQLAlchemy ORM records for storage and later AI training.
"""

from __future__ import annotations

from hpe.core.db_models import (
    OperatingPointRecord,
    PerformanceDataPoint,
    SizingResultRecord,
)
from hpe.core.models import OperatingPoint, PerformanceMetrics, SizingResult


def sizing_result_to_record(
    sizing: SizingResult,
    project_id: str,
    operating_point_id: str | None = None,
) -> SizingResultRecord:
    """Convert a SizingResult dataclass to a database record."""
    return SizingResultRecord(
        project_id=project_id,
        operating_point_id=operating_point_id,
        specific_speed_nq=sizing.specific_speed_nq,
        impeller_type=sizing.meridional_profile.get("impeller_type"),
        impeller_d2=sizing.impeller_d2,
        impeller_d1=sizing.impeller_d1,
        impeller_b2=sizing.impeller_b2,
        blade_count=sizing.blade_count,
        beta1=sizing.beta1,
        beta2=sizing.beta2,
        estimated_efficiency=sizing.estimated_efficiency,
        estimated_power=sizing.estimated_power,
        estimated_npsh_r=sizing.estimated_npsh_r,
        sigma=sizing.sigma,
        velocity_triangles=sizing.velocity_triangles,
        meridional_profile=sizing.meridional_profile,
        warnings=sizing.warnings,
    )


def operating_point_to_record(
    op: OperatingPoint,
    project_id: str,
    label: str = "design",
) -> OperatingPointRecord:
    """Convert an OperatingPoint dataclass to a database record."""
    return OperatingPointRecord(
        project_id=project_id,
        label=label,
        flow_rate=op.flow_rate,
        head=op.head,
        rpm=op.rpm,
        fluid_type=op.fluid.value,
        fluid_density=op.fluid_density,
        fluid_viscosity=op.fluid_viscosity,
    )


def performance_to_data_point(
    perf: PerformanceMetrics,
    simulation_run_id: str,
    flow_rate: float,
    source: str = "physics_1d",
) -> PerformanceDataPoint:
    """Convert PerformanceMetrics to a database data point."""
    return PerformanceDataPoint(
        simulation_run_id=simulation_run_id,
        flow_rate=flow_rate,
        head=perf.head,
        hydraulic_efficiency=perf.hydraulic_efficiency,
        volumetric_efficiency=perf.volumetric_efficiency,
        mechanical_efficiency=perf.mechanical_efficiency,
        total_efficiency=perf.total_efficiency,
        power=perf.power,
        torque=perf.torque,
        npsh_required=perf.npsh_required,
        min_pressure_coefficient=perf.min_pressure_coefficient,
        source=source,
    )
