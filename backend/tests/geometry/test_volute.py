"""Tests for volute geometry generation."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("cadquery", reason="3D volute generation requires CadQuery")

from hpe.core.enums import GeometryFormat
from hpe.core.models import OperatingPoint, SizingResult
from hpe.geometry.volute.cross_section import (
    circular_section,
    rectangular_section,
    trapezoidal_section,
)
from hpe.geometry.volute.models import VoluteParams, VoluteSizing
from hpe.geometry.volute.sizing import size_volute
from hpe.geometry.volute.volute_3d import generate_volute, generate_volute_from_sizing
from hpe.sizing import run_sizing


@pytest.fixture
def volute_params() -> VoluteParams:
    return VoluteParams(
        d2=0.250,
        b2=0.020,
        flow_rate=0.05,
        cu2=14.0,
    )


@pytest.fixture
def sizing_result() -> SizingResult:
    op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
    return run_sizing(op)


class TestVoluteSizing:
    def test_area_increases(self, volute_params: VoluteParams) -> None:
        """Area should increase from 0 to 360 degrees."""
        sizing = size_volute(volute_params)
        # Area at end should be > area at start
        assert sizing.areas[-1] > sizing.areas[1]

    def test_correct_station_count(self, volute_params: VoluteParams) -> None:
        sizing = size_volute(volute_params)
        assert len(sizing.theta_stations) == volute_params.n_stations + 1

    def test_r3_positive(self, volute_params: VoluteParams) -> None:
        sizing = size_volute(volute_params)
        assert sizing.r3 > volute_params.d2 / 2

    def test_discharge_area_positive(self, volute_params: VoluteParams) -> None:
        sizing = size_volute(volute_params)
        assert sizing.discharge_area > 0

    def test_all_areas_non_negative(self, volute_params: VoluteParams) -> None:
        sizing = size_volute(volute_params)
        for a in sizing.areas:
            assert a >= 0

    def test_radii_increase(self, volute_params: VoluteParams) -> None:
        """Outer radius should generally increase."""
        sizing = size_volute(volute_params)
        assert sizing.radii[-1] > sizing.radii[1]


class TestCrossSection:
    def test_circular_area(self) -> None:
        points = circular_section(0.001, 0.15)
        assert len(points) > 3

    def test_circular_zero_area(self) -> None:
        points = circular_section(0.0, 0.15)
        assert len(points) == 1  # Single point

    def test_trapezoidal(self) -> None:
        points = trapezoidal_section(0.001, 0.02, 0.15)
        assert len(points) == 5  # 4 corners + close

    def test_rectangular(self) -> None:
        points = rectangular_section(0.001, 0.02, 0.15)
        assert len(points) == 5


class TestVolute3D:
    def test_creates_solid(self, volute_params: VoluteParams) -> None:
        volute = generate_volute(volute_params)
        assert volute is not None
        assert volute.val().Volume() > 0

    def test_from_sizing_result(self, sizing_result: SizingResult) -> None:
        volute = generate_volute_from_sizing(sizing_result)
        assert volute is not None
        assert volute.val().Volume() > 0

    def test_export_step(self, volute_params: VoluteParams) -> None:
        from hpe.geometry.runner.export import export_runner

        volute = generate_volute(volute_params)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_runner(volute, Path(tmpdir) / "volute.step", GeometryFormat.STEP)
            assert path.exists()
            assert path.stat().st_size > 100
