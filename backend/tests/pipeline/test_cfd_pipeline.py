"""Tests for CFD pipeline orchestrator."""

import tempfile

import pytest

from hpe.core.models import OperatingPoint
from hpe.pipeline.cfd_pipeline import run_cfd_pipeline
from hpe.sizing import run_sizing


class TestCFDPipeline:
    def test_generates_case_without_solver(self) -> None:
        """Pipeline should generate case even without OpenFOAM."""
        pytest.importorskip("cadquery", reason="geometry export step requires CadQuery")
        op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
        sizing = run_sizing(op)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cfd_pipeline(
                sizing, tmpdir,
                run_solver=False,
                export_geometry=True,
            )

            # Case directory should exist
            assert result.case_dir.exists()
            assert (result.case_dir / "system").is_dir()
            assert (result.case_dir / "run.sh").exists()

            # Geometry should be exported
            assert result.step_file is not None
            assert result.step_file.exists()

            # Solver should not have run
            assert result.ran_simulation is False

    def test_case_has_no_template_placeholders(self) -> None:
        """All template variables should be substituted."""
        op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
        sizing = run_sizing(op)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cfd_pipeline(sizing, tmpdir, run_solver=False)

            # Check all files for unresolved placeholders
            for f in result.case_dir.rglob("*"):
                if f.is_file() and f.suffix not in [".stl", ".step", ".sh"]:
                    content = f.read_text()
                    assert "{{" not in content, f"Placeholder found in {f}"
