"""Tests for OpenFOAM case builder."""

import tempfile
from pathlib import Path

import pytest

from hpe.cfd.openfoam.boundary_conditions import calc_bc_values
from hpe.cfd.openfoam.case_builder import build_case
from hpe.core.models import SizingResult


class TestCalcBCValues:
    def test_inlet_velocity_positive(self) -> None:
        bc = calc_bc_values(0.05, 1750, 0.12, 0.04)
        assert bc.u_inlet > 0

    def test_omega_positive(self) -> None:
        bc = calc_bc_values(0.05, 1750, 0.12, 0.04)
        assert bc.omega_rotor > 0

    def test_k_positive(self) -> None:
        bc = calc_bc_values(0.05, 1750, 0.12, 0.04)
        assert bc.k_init > 0

    def test_nu_correct(self) -> None:
        bc = calc_bc_values(0.05, 1750, 0.12, 0.04)
        expected_nu = 1.003e-3 / 998.2
        assert bc.nu == pytest.approx(expected_nu, rel=1e-3)


class TestBuildCase:
    def test_creates_directory_structure(self, sizing_result: SizingResult) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = build_case(
                sizing_result, Path(tmpdir) / "fake.stl",
                Path(tmpdir) / "case",
            )
            assert (case_dir / "0").is_dir()
            assert (case_dir / "constant").is_dir()
            assert (case_dir / "system").is_dir()

    def test_creates_run_script(self, sizing_result: SizingResult) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = build_case(
                sizing_result, Path(tmpdir) / "fake.stl",
                Path(tmpdir) / "case",
            )
            run_sh = case_dir / "run.sh"
            assert run_sh.exists()
            content = run_sh.read_text()
            assert "blockMesh" in content
            assert "simpleFoam" in content

    def test_creates_blockmesh_dict(self, sizing_result: SizingResult) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = build_case(
                sizing_result, Path(tmpdir) / "fake.stl",
                Path(tmpdir) / "case",
            )
            bmd = case_dir / "system" / "blockMeshDict"
            assert bmd.exists()
            assert "vertices" in bmd.read_text()

    def test_creates_snappy_dict(self, sizing_result: SizingResult) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = build_case(
                sizing_result, Path(tmpdir) / "fake.stl",
                Path(tmpdir) / "case",
            )
            shd = case_dir / "system" / "snappyHexMeshDict"
            assert shd.exists()

    def test_creates_mrf(self, sizing_result: SizingResult) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = build_case(
                sizing_result, Path(tmpdir) / "fake.stl",
                Path(tmpdir) / "case",
            )
            mrf = case_dir / "constant" / "MRFProperties"
            assert mrf.exists()
            content = mrf.read_text()
            assert "omega" in content
            assert "cellZone" in content

    def test_bc_files_have_values(self, sizing_result: SizingResult) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = build_case(
                sizing_result, Path(tmpdir) / "fake.stl",
                Path(tmpdir) / "case",
            )
            u_file = case_dir / "0" / "U"
            if u_file.exists():
                content = u_file.read_text()
                # Should have actual numeric values, not template placeholders
                assert "{{" not in content
