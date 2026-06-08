"""End-to-end integration tests.

Tests the complete pipeline: sizing → geometry → physics → optimization → AI
in a single flow, verifying that all modules work together correctly.
"""

import tempfile
from pathlib import Path

import pytest

from hpe.core.enums import GeometryFormat
from hpe.core.models import OperatingPoint


class TestFullPipeline:
    """Complete pipeline: OperatingPoint → Sizing → Geometry → Physics → Export."""

    def test_centrifugal_pump_pipeline(self) -> None:
        pytest.importorskip("cadquery", reason="3D runner export requires CadQuery")
        from hpe.geometry.runner import generate_runner_from_sizing
        from hpe.geometry.runner.export import export_runner
        from hpe.physics.curves import generate_curves
        from hpe.physics.performance import evaluate_design_point
        from hpe.physics.stability import analyze_stability
        from hpe.sizing import run_sizing

        # 1. Sizing
        op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
        sizing = run_sizing(op)
        assert sizing.impeller_d2 > 0
        assert sizing.estimated_efficiency > 0.5

        # 2. Geometry → STEP
        runner = generate_runner_from_sizing(sizing)
        assert runner.val().Volume() > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_runner(runner, Path(tmpdir) / "pump.step", GeometryFormat.STEP)
            assert path.exists()
            assert path.stat().st_size > 1000

        # 3. Physics — design point
        perf = evaluate_design_point(sizing)
        assert perf.head > 0
        assert 0 < perf.total_efficiency < 1
        assert perf.power > 0

        # 4. Performance curves
        curves = generate_curves(sizing, n_points=10)
        assert len(curves.flow_rates) == 10
        assert curves.heads[0] > curves.heads[-1]  # Falling H-Q

        # 5. Stability
        stability = analyze_stability(sizing)
        assert stability.bep_flow > 0
        assert stability.bep_efficiency > 0.5

    def test_volute_and_distributor(self) -> None:
        pytest.importorskip("cadquery", reason="3D volute/distributor export requires CadQuery")
        from hpe.geometry.distributor import generate_distributor_from_sizing
        from hpe.geometry.volute import generate_volute_from_sizing
        from hpe.sizing import run_sizing

        op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
        sizing = run_sizing(op)

        volute = generate_volute_from_sizing(sizing)
        assert volute.val().Volume() > 0

        dist = generate_distributor_from_sizing(sizing)
        assert dist.val().Volume() > 0

    def test_optimization_pipeline(self) -> None:
        from hpe.optimization import run_optimization
        from hpe.optimization.problem import OptimizationProblem

        problem = OptimizationProblem.default(0.05, 30.0, 1750)
        result = run_optimization(problem, method="nsga2", pop_size=10, n_gen=5, seed=42)

        assert len(result.pareto_front) > 0
        assert result.best_efficiency is not None
        best_eta = result.best_efficiency["objectives"]["efficiency"]
        assert best_eta > 0.3

    def test_ai_pipeline(self) -> None:
        from hpe.ai.anomaly.validators import validate_geometry, validate_performance
        from hpe.ai.assistant.interpreter import interpret_sizing
        from hpe.ai.assistant.recommender import recommend_improvements
        from hpe.ai.surrogate.predictor import SurrogatePredictor
        from hpe.optimization.problem import OptimizationProblem
        from hpe.physics.euler import get_design_flow_rate
        from hpe.physics.performance import evaluate_performance
        from hpe.sizing import run_sizing

        # Sizing
        op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
        sizing = run_sizing(op)

        # Validate
        geo_valid = validate_geometry(sizing)
        assert geo_valid.valid

        perf = evaluate_performance(sizing, get_design_flow_rate(sizing))
        perf_valid = validate_performance(perf)
        assert perf_valid.valid

        # Interpret
        text = interpret_sizing(sizing)
        assert len(text) > 50

        # Recommend
        recs = recommend_improvements(sizing, perf)
        assert isinstance(recs, list)

        # Surrogate
        problem = OptimizationProblem.default(0.05, 30.0, 1750)
        predictor = SurrogatePredictor(problem)
        metrics = predictor.build(n_samples=30, seed=42)
        assert predictor.is_ready
        pred = predictor.predict([25.0, 1.0, 1.0, 7])
        assert pred["efficiency"] > 0

    def test_cfd_case_generation(self) -> None:
        pytest.importorskip("cadquery", reason="CFD case geometry export requires CadQuery")
        from hpe.pipeline import run_cfd_pipeline
        from hpe.sizing import run_sizing

        op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
        sizing = run_sizing(op)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cfd_pipeline(sizing, tmpdir, run_solver=False)
            assert result.case_dir.exists()
            assert (result.case_dir / "system" / "blockMeshDict").exists()
            assert (result.case_dir / "run.sh").exists()
            assert result.step_file is not None

    def test_francis_turbine_sizing(self) -> None:
        from hpe.sizing.francis import size_francis

        result = size_francis(2.0, 100.0, 600)
        assert result.d1 > 0
        assert result.estimated_power > 1e6  # >1 MW
        assert result.blade_count >= 9
