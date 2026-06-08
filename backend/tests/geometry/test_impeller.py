"""Tests for 3D impeller generation and export."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("cadquery", reason="3D impeller generation requires CadQuery")

from hpe.core.enums import GeometryFormat
from hpe.core.models import OperatingPoint, SizingResult
from hpe.geometry.models import RunnerGeometryParams
from hpe.geometry.runner.export import export_runner
from hpe.geometry.runner.impeller import generate_runner, generate_runner_from_sizing


class TestGenerateRunner:
    def test_creates_solid(self, runner_params: RunnerGeometryParams) -> None:
        """Should produce a valid CadQuery solid."""
        runner = generate_runner(runner_params)
        assert runner is not None
        # Should have a valid shape
        val = runner.val()
        assert val is not None

    def test_solid_has_volume(self, runner_params: RunnerGeometryParams) -> None:
        """Solid should have positive volume."""
        runner = generate_runner(runner_params)
        # Get the volume of the solid (in mm^3 since CadQuery uses mm)
        vol = runner.val().Volume()
        assert vol > 0

    def test_from_sizing_result(self, sizing_result: SizingResult) -> None:
        """Should work from a SizingResult."""
        runner = generate_runner_from_sizing(sizing_result)
        assert runner is not None
        assert runner.val().Volume() > 0


class TestExportRunner:
    def test_export_step(self, runner_params: RunnerGeometryParams) -> None:
        """Should export a valid STEP file."""
        runner = generate_runner(runner_params)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_runner(
                runner,
                Path(tmpdir) / "impeller.step",
                fmt=GeometryFormat.STEP,
            )
            assert path.exists()
            assert path.stat().st_size > 100  # Non-trivial file

    def test_export_stl(self, runner_params: RunnerGeometryParams) -> None:
        """Should export a valid STL file."""
        runner = generate_runner(runner_params)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_runner(
                runner,
                Path(tmpdir) / "impeller.stl",
                fmt=GeometryFormat.STL,
            )
            assert path.exists()
            assert path.stat().st_size > 100

    def test_export_fixes_extension(self, runner_params: RunnerGeometryParams) -> None:
        """Should fix file extension if wrong."""
        runner = generate_runner(runner_params)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_runner(
                runner,
                Path(tmpdir) / "impeller.wrong",
                fmt=GeometryFormat.STEP,
            )
            assert path.suffix == ".step"
            assert path.exists()


class TestEndToEnd:
    def test_sizing_to_geometry_to_export(self) -> None:
        """Full pipeline: OperatingPoint -> Sizing -> Geometry -> STEP file."""
        from hpe.sizing import run_sizing

        op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
        sizing = run_sizing(op)
        runner = generate_runner_from_sizing(sizing)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_runner(
                runner,
                Path(tmpdir) / "centrifugal_pump.step",
            )
            assert path.exists()
            assert path.stat().st_size > 1000  # Reasonable STEP file
