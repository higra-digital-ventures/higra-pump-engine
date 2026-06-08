"""Tests for distributor guide vane generation."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("cadquery", reason="distributor guide-vane generation requires CadQuery")

from hpe.core.enums import GeometryFormat
from hpe.core.models import OperatingPoint, SizingResult
from hpe.geometry.distributor.guide_vanes import (
    DistributorParams,
    generate_distributor,
    generate_distributor_from_sizing,
)
from hpe.sizing import run_sizing


@pytest.fixture
def distributor_params() -> DistributorParams:
    return DistributorParams(d2=0.250, b2=0.020)


@pytest.fixture
def sizing_result() -> SizingResult:
    op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
    return run_sizing(op)


class TestDistributorParams:
    def test_r_inner_greater_than_r2(self, distributor_params: DistributorParams) -> None:
        assert distributor_params.r_inner > distributor_params.d2 / 2

    def test_r_outer_greater_than_inner(self, distributor_params: DistributorParams) -> None:
        assert distributor_params.r_outer > distributor_params.r_inner

    def test_from_sizing(self, sizing_result: SizingResult) -> None:
        params = DistributorParams.from_sizing_result(sizing_result)
        assert params.d2 == sizing_result.impeller_d2


class TestDistributor3D:
    def test_creates_solid(self, distributor_params: DistributorParams) -> None:
        dist = generate_distributor(distributor_params)
        assert dist is not None
        assert dist.val().Volume() > 0

    def test_from_sizing_result(self, sizing_result: SizingResult) -> None:
        dist = generate_distributor_from_sizing(sizing_result)
        assert dist is not None
        assert dist.val().Volume() > 0

    def test_export_step(self, distributor_params: DistributorParams) -> None:
        from hpe.geometry.runner.export import export_runner

        dist = generate_distributor(distributor_params)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_runner(dist, Path(tmpdir) / "distributor.step", GeometryFormat.STEP)
            assert path.exists()
            assert path.stat().st_size > 100

    def test_different_vane_count(self) -> None:
        params_11 = DistributorParams(d2=0.250, b2=0.020, n_vanes=11)
        params_7 = DistributorParams(d2=0.250, b2=0.020, n_vanes=7)
        dist_11 = generate_distributor(params_11)
        dist_7 = generate_distributor(params_7)
        # 11 vanes should have more volume than 7
        assert dist_11.val().Volume() > dist_7.val().Volume()
