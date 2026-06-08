"""Tests for OpenFOAM result parser."""

import tempfile


from hpe.postprocess.openfoam_parser import parse_forces, parse_solver_log


class TestParseSolverLog:
    def test_empty_log(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("")
            f.flush()
            data = parse_solver_log(f.name)
            assert data.converged is False
            assert len(data.p_residuals) == 0

    def test_parses_residuals(self) -> None:
        log_content = """
Time = 1
smoothSolver:  Solving for Ux, Initial residual = 1.0, Final residual = 0.1, No Iterations 5
smoothSolver:  Solving for Uy, Initial residual = 0.8, Final residual = 0.05, No Iterations 5
GAMG:  Solving for p, Initial residual = 0.5, Final residual = 0.01, No Iterations 10

Time = 2
smoothSolver:  Solving for Ux, Initial residual = 0.01, Final residual = 0.001, No Iterations 5
GAMG:  Solving for p, Initial residual = 0.001, Final residual = 0.0001, No Iterations 10

Time = 3
smoothSolver:  Solving for Ux, Initial residual = 1e-05, Final residual = 1e-06, No Iterations 5
GAMG:  Solving for p, Initial residual = 1e-05, Final residual = 1e-06, No Iterations 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            f.flush()
            data = parse_solver_log(f.name)
            assert len(data.p_residuals) == 3
            assert data.p_residuals[0] == 0.5
            assert data.p_residuals[-1] == 1e-5
            assert data.converged is True  # Final p < 1e-4

    def test_nonexistent_file(self) -> None:
        data = parse_solver_log("/nonexistent/log.txt")
        assert data.converged is False


class TestParseForces:
    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = parse_forces(tmpdir)
            assert len(data.force_x) == 0

    def test_nonexistent_dir(self) -> None:
        data = parse_forces("/nonexistent/forces1")
        assert len(data.force_x) == 0
