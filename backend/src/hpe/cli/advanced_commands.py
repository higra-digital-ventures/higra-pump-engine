"""CLI commands para Fases 17-20 — melhorias #46-48.

Subcomandos novos para o `hpe` CLI:
  hpe benchmarks [--method meanline]
  hpe report --project NAME [--out report.pdf]
  hpe adjoint --Q 0.05 --H 30 --n 1750 [--max-iter 5]

Estes comandos usam os módulos do backend diretamente sem precisar
da API rodar.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def cmd_benchmarks(args: argparse.Namespace) -> int:
    """`hpe benchmarks` — validar HPE contra benchmarks experimentais."""
    from hpe.validation.benchmarks import list_benchmarks, run_all_benchmarks
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    print("=" * 64)
    print(f"HPE Benchmarks — method: {args.method}")
    print("=" * 64)

    print("\nDisponíveis:")
    for b in list_benchmarks():
        print(f"  • {b.name}: {b.description}")
        print(f"    rpm={b.rpm}, D₂={b.D2*1000:.0f}mm, Z={b.n_blades}, n_pts={b.n_points}")

    print("\nExecutando validação...")

    def builder(Q_bep: float, H_bep: float, rpm: float):
        op = OperatingPoint(flow_rate=Q_bep, head=H_bep, rpm=rpm)
        sizing = run_sizing(op)
        eta_bep = float(getattr(sizing, "estimated_efficiency", 0.80))

        def H_fn(Q):
            f = Q / Q_bep if Q_bep > 0 else 1
            return H_bep * max(0.3, 1.25 - 0.05 * f - 0.20 * f * f)

        def eta_fn(Q):
            f = Q / Q_bep if Q_bep > 0 else 1
            return eta_bep * (1 - 0.6 * (f - 1) ** 2)

        def P_fn(Q):
            return 998.2 * 9.81 * Q * H_fn(Q) / max(eta_fn(Q), 0.1)

        return H_fn, eta_fn, P_fn

    results = run_all_benchmarks(builder)
    print()
    print(f"{'Benchmark':<25} {'N':>4} {'MAPE H':>10} {'MAPE η':>10} {'Status':>10}")
    print("-" * 64)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.benchmark:<25} {r.n_points:>4} "
              f"{r.mape_head:>9.2f}% {r.mape_efficiency:>9.2f}% {status:>10}")

    n_pass = sum(1 for r in results if r.passed)
    print(f"\n{n_pass}/{len(results)} passaram")
    return 0 if n_pass == len(results) else 1


def cmd_report(args: argparse.Namespace) -> int:
    """`hpe report` — gerar relatório técnico."""
    from hpe.reports.generator import generate_report, ReportContext
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    print(f"Gerando sizing para Q={args.Q}, H={args.H}, n={args.n}…")
    op = OperatingPoint(flow_rate=args.Q, head=args.H, rpm=args.n)
    sizing = run_sizing(op)

    sizing_dict = {
        "Q": args.Q, "H": args.H, "n": args.n,
        "specific_speed_nq": float(getattr(sizing, "specific_speed_nq", 0)),
        "impeller_d2": float(getattr(sizing, "impeller_d2", 0)),
        "impeller_d1": float(getattr(sizing, "impeller_d1", 0)),
        "impeller_b2": float(getattr(sizing, "impeller_b2", 0)),
        "blade_count": int(getattr(sizing, "blade_count", 6)),
        "beta1": float(getattr(sizing, "beta1", 0)),
        "beta2": float(getattr(sizing, "beta2", 0)),
        "estimated_efficiency": float(getattr(sizing, "estimated_efficiency", 0.8)),
        "estimated_power": float(getattr(sizing, "estimated_power", 0)),
    }

    ctx = ReportContext(
        project_name=args.project,
        sizing=sizing_dict,
    )

    out_path = Path(args.out)
    final = generate_report(ctx, out_path, format=args.format)
    print(f"Relatório gerado: {final}")
    return 0


def cmd_adjoint(args: argparse.Namespace) -> int:
    """`hpe adjoint` — rodar loop de otimização adjoint."""
    from hpe.cfd.adjoint_loop import run_adjoint_loop, AdjointConfig
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    import tempfile

    print(f"Rodando adjoint loop: Q={args.Q}, H={args.H}, n={args.n}, max_iter={args.max_iter}")

    op = OperatingPoint(flow_rate=args.Q, head=args.H, rpm=args.n)
    sizing = run_sizing(op)

    config = AdjointConfig(
        max_iter=args.max_iter,
        step_size=args.step,
        tol=args.tol,
        output_dir=args.out or str(Path(tempfile.gettempdir()) / "hpe_adjoint_cli"),
    )

    result = run_adjoint_loop(sizing, Path(config.output_dir), config)

    print(f"\nLoop completed: {result.n_iterations} iterações")
    print(f"Converged: {result.converged}")
    print(f"Best objective: {result.best_objective}")
    if result.improvement_pct is not None:
        print(f"Improvement: {result.improvement_pct:.2f}%")

    print("\nHistory:")
    print(f"{'iter':>4} {'objective':>14} {'|∇J|':>12}")
    for h in result.history:
        obj = f"{h.objective:.6f}" if h.objective else "—"
        print(f"{h.iteration:>4} {obj:>14} {h.gradient_norm:>12.6f}")

    return 0


def register_advanced_commands(subparsers: argparse._SubParsersAction) -> None:
    """Registrar todos os subcomandos em um parser argparse existente."""

    # benchmarks
    p = subparsers.add_parser("benchmarks", help="Validar HPE contra benchmarks")
    p.add_argument("--method", default="meanline", choices=["meanline", "surrogate"])
    p.set_defaults(func=cmd_benchmarks)

    # report
    p = subparsers.add_parser("report", help="Gerar relatório técnico PDF/HTML/MD")
    p.add_argument("--Q", type=float, required=True, help="Vazão [m³/s]")
    p.add_argument("--H", type=float, required=True, help="Altura [m]")
    p.add_argument("--n", type=float, required=True, help="rpm")
    p.add_argument("--project", default="HPE Pump Design")
    p.add_argument("--out", default="report")
    p.add_argument("--format", default="auto", choices=["auto", "pdf", "html", "markdown"])
    p.set_defaults(func=cmd_report)

    # adjoint
    p = subparsers.add_parser("adjoint", help="Rodar loop de otimização adjoint")
    p.add_argument("--Q", type=float, required=True)
    p.add_argument("--H", type=float, required=True)
    p.add_argument("--n", type=float, required=True)
    p.add_argument("--max-iter", type=int, default=5, dest="max_iter")
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_adjoint)


def main() -> int:
    """Standalone entry point — `python -m hpe.cli.advanced_commands`."""
    parser = argparse.ArgumentParser(prog="hpe-advanced", description="HPE advanced CLI commands")
    sub = parser.add_subparsers(dest="cmd", required=True)
    register_advanced_commands(sub)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
