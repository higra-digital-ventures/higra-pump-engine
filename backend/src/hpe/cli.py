"""HPE Command-Line Interface.

Usage:
    hpe sizing --flow 0.05 --head 30 --rpm 1750
    hpe sizing --flow 0.05 --head 30 --rpm 1750 --export pump.step
    hpe curves --flow 0.05 --head 30 --rpm 1750 --output curves.csv
    hpe analyze --flow 0.05 --head 30 --rpm 1750
    hpe cfd --flow 0.05 --head 30 --rpm 1750 --output ./case_pump
"""

from __future__ import annotations

import argparse
import csv
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hpe",
        description="Higra Pump Engine — hydraulic turbomachinery design platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- sizing ---
    sp_sizing = subparsers.add_parser("sizing", help="Run 1D meanline sizing")
    _add_operating_point_args(sp_sizing)
    sp_sizing.add_argument("--export", type=str, help="Export geometry to STEP file")
    sp_sizing.add_argument("--stl", type=str, help="Export geometry to STL file")

    # --- curves ---
    sp_curves = subparsers.add_parser("curves", help="Generate performance curves")
    _add_operating_point_args(sp_curves)
    sp_curves.add_argument("--output", "-o", type=str, help="Save curves to CSV file")
    sp_curves.add_argument("--points", type=int, default=25, help="Number of curve points")

    # --- analyze ---
    sp_analyze = subparsers.add_parser("analyze", help="Run stability analysis")
    _add_operating_point_args(sp_analyze)

    # --- cfd ---
    sp_cfd = subparsers.add_parser("cfd", help="Generate OpenFOAM CFD case")
    _add_operating_point_args(sp_cfd)
    sp_cfd.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    sp_cfd.add_argument("--procs", type=int, default=4, help="Number of processors")
    sp_cfd.add_argument("--run", action="store_true", help="Attempt to run OpenFOAM")

    # --- batch ---
    sp_batch = subparsers.add_parser("batch", help="Batch sizing from JSON input file")
    sp_batch.add_argument("--input", "-i", type=str, required=True, help="Input JSON file (list of {flow_rate, head, rpm, name})")
    sp_batch.add_argument("--output", "-o", type=str, default="results.json", help="Output file (default: results.json)")
    sp_batch.add_argument("--format", "-f", type=str, default="json", choices=["json", "csv"], help="Output format: json or csv")

    # --- optimize ---
    sp_opt = subparsers.add_parser("optimize", help="Run multi-objective optimization")
    _add_operating_point_args(sp_opt)
    sp_opt.add_argument("--method", choices=["nsga2", "bayesian"], default="nsga2", help="Optimization method")
    sp_opt.add_argument("--pop", type=int, default=40, help="Population size (NSGA-II)")
    sp_opt.add_argument("--gen", type=int, default=50, help="Generations (NSGA-II) or trials (Bayesian)")
    sp_opt.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "sizing":
        _cmd_sizing(args)
    elif args.command == "curves":
        _cmd_curves(args)
    elif args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "cfd":
        _cmd_cfd(args)
    elif args.command == "optimize":
        _cmd_optimize(args)
    elif args.command == "batch":
        _cmd_batch(args)


def _add_operating_point_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--flow", "-Q", type=float, required=True, help="Flow rate Q [m3/s]")
    parser.add_argument("--head", "-H", type=float, required=True, help="Head H [m]")
    parser.add_argument("--rpm", "-n", type=float, required=True, help="Rotational speed [rev/min]")
    parser.add_argument(
        "--type", type=str, default="centrifugal_pump",
        choices=["centrifugal_pump", "axial_pump", "mixed_flow_pump", "francis_turbine", "pump_turbine"],
        help="Machine type",
    )


def _make_operating_point(args: argparse.Namespace):
    from hpe.core.enums import MachineType
    from hpe.core.models import OperatingPoint

    return OperatingPoint(
        flow_rate=args.flow,
        head=args.head,
        rpm=args.rpm,
        machine_type=MachineType(args.type),
    )


def _cmd_sizing(args: argparse.Namespace) -> None:
    from hpe.sizing import run_sizing

    op = _make_operating_point(args)
    result = run_sizing(op)

    print("=" * 60)
    print("  HIGRA PUMP ENGINE — 1D Meanline Sizing")
    print("=" * 60)
    print()
    print(f"  Input: Q={op.flow_rate*3600:.1f} m3/h, H={op.head:.1f} m, n={op.rpm:.0f} rpm")
    print()
    print("  Specific Speed")
    print(f"    Nq = {result.specific_speed_nq:.1f}")
    print(f"    Type: {result.meridional_profile.get('impeller_type', 'N/A')}")
    print()
    print("  Impeller Dimensions")
    print(f"    D2 = {result.impeller_d2*1000:.1f} mm")
    print(f"    D1 = {result.impeller_d1*1000:.1f} mm")
    print(f"    b2 = {result.impeller_b2*1000:.1f} mm")
    print(f"    Blades: {result.blade_count}")
    print(f"    beta1 = {result.beta1:.1f} deg")
    print(f"    beta2 = {result.beta2:.1f} deg")
    print()
    print("  Performance Estimates")
    print(f"    Efficiency: {result.estimated_efficiency:.1%}")
    print(f"    Power: {result.estimated_power/1000:.1f} kW")
    print(f"    NPSHr: {result.estimated_npsh_r:.1f} m")
    print(f"    Sigma: {result.sigma:.4f}")
    print()
    print("  Velocity Triangles")
    tri = result.velocity_triangles
    print(f"    Inlet:  u1={tri['inlet']['u']:.1f} m/s, cm1={tri['inlet']['cm']:.1f} m/s, beta1={tri['inlet']['beta']:.1f} deg")
    print(f"    Outlet: u2={tri['outlet']['u']:.1f} m/s, cm2={tri['outlet']['cm']:.1f} m/s, cu2={tri['outlet']['cu']:.1f} m/s")
    print(f"    Euler Head: {tri['euler_head']:.1f} m")

    if result.warnings:
        print()
        print("  Warnings")
        for w in result.warnings:
            print(f"    ! {w}")

    # Export geometry
    if args.export or args.stl:
        from hpe.core.enums import GeometryFormat
        from hpe.geometry.runner import generate_runner_from_sizing
        from hpe.geometry.runner.export import export_runner

        runner = generate_runner_from_sizing(result)

        if args.export:
            path = export_runner(runner, args.export, GeometryFormat.STEP)
            print(f"\n  STEP exported: {path}")

        if args.stl:
            path = export_runner(runner, args.stl, GeometryFormat.STL)
            print(f"\n  STL exported: {path}")

    print()


def _cmd_curves(args: argparse.Namespace) -> None:
    from hpe.physics.curves import generate_curves
    from hpe.sizing import run_sizing

    op = _make_operating_point(args)
    sizing = run_sizing(op)
    curves = generate_curves(sizing, n_points=args.points)

    print("=" * 60)
    print("  HIGRA PUMP ENGINE — Performance Curves")
    print("=" * 60)
    print()
    print(f"  {'Q [m3/h]':>10}  {'H [m]':>8}  {'eta [%]':>8}  {'P [kW]':>8}  {'NPSHr [m]':>10}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    for i in range(len(curves.flow_rates)):
        q_h = curves.flow_rates[i] * 3600
        h = curves.heads[i]
        eta = curves.efficiencies[i] * 100
        p = curves.powers[i] / 1000
        npsh = curves.npsh_required[i]
        print(f"  {q_h:10.1f}  {h:8.1f}  {eta:8.1f}  {p:8.1f}  {npsh:10.1f}")

    if args.output:
        _save_curves_csv(curves, args.output)
        print(f"\n  CSV saved: {args.output}")

    print()


def _cmd_analyze(args: argparse.Namespace) -> None:
    from hpe.physics.stability import analyze_stability
    from hpe.sizing import run_sizing

    op = _make_operating_point(args)
    sizing = run_sizing(op)
    analysis = analyze_stability(sizing)

    print("=" * 60)
    print("  HIGRA PUMP ENGINE — Stability Analysis")
    print("=" * 60)
    print()
    print("  Best Efficiency Point (BEP)")
    print(f"    Q_bep = {analysis.bep_flow*3600:.1f} m3/h")
    print(f"    H_bep = {analysis.bep_head:.1f} m")
    print(f"    eta_max = {analysis.bep_efficiency:.1%}")
    print()
    print("  Operating Limits")
    print(f"    Shutdown head: {analysis.shutdown_head:.1f} m (ratio: {analysis.shutdown_ratio:.2f})")
    print(f"    Min flow: {analysis.min_flow*3600:.1f} m3/h ({analysis.min_flow_ratio:.0%} of design)")
    print(f"    Stable curve: {'Yes' if analysis.is_stable else 'No'}")

    if analysis.unstable_regions:
        print(f"    Unstable zones: {len(analysis.unstable_regions)}")
        for qs, qe in analysis.unstable_regions:
            print(f"      Q = {qs*3600:.1f} - {qe*3600:.1f} m3/h")

    if analysis.warnings:
        print()
        print("  Warnings")
        for w in analysis.warnings:
            print(f"    ! {w}")

    print()


def _cmd_optimize(args: argparse.Namespace) -> None:
    from hpe.optimization.problem import OptimizationProblem

    op = _make_operating_point(args)

    print("=" * 60)
    print("  HIGRA PUMP ENGINE — Multi-Objective Optimization")
    print("=" * 60)
    print()
    print(f"  Input: Q={op.flow_rate*3600:.1f} m3/h, H={op.head:.1f} m, n={op.rpm:.0f} rpm")
    print(f"  Method: {args.method}")
    print()

    problem = OptimizationProblem.default(op.flow_rate, op.head, op.rpm)

    if args.method == "nsga2":
        from hpe.optimization.nsga2 import run_nsga2

        print(f"  Running NSGA-II (pop={args.pop}, gen={args.gen})...")
        result = run_nsga2(problem, pop_size=args.pop, n_gen=args.gen, seed=args.seed)

        print(f"  Evaluations: {result.all_evaluations}")
        print(f"  Pareto front: {len(result.pareto_front)} designs")
        print()

        if result.best_efficiency:
            b = result.best_efficiency
            print("  Best by Efficiency:")
            for k, v in b["variables"].items():
                print(f"    {k} = {v:.2f}" if not isinstance(v, int) else f"    {k} = {v}")
            print(f"    eta = {b['objectives']['efficiency']:.1%}")
            print(f"    NPSHr = {b['objectives']['npsh_r']:.2f} m")
            print(f"    Robustness = {b['objectives']['robustness']:.1%}")

        if result.best_npsh:
            b = result.best_npsh
            print()
            print("  Best by NPSH (lowest cavitation risk):")
            for k, v in b["variables"].items():
                print(f"    {k} = {v:.2f}" if not isinstance(v, int) else f"    {k} = {v}")
            print(f"    eta = {b['objectives']['efficiency']:.1%}")
            print(f"    NPSHr = {b['objectives']['npsh_r']:.2f} m")

    elif args.method == "bayesian":
        from hpe.optimization.bayesian import run_bayesian

        print(f"  Running Bayesian optimization ({args.gen} trials)...")
        result = run_bayesian(problem, n_trials=args.gen, seed=args.seed)

        print(f"  Best efficiency: {result['best_value']:.1%}")
        print("  Best parameters:")
        for k, v in result["best_params"].items():
            print(f"    {k} = {v:.2f}" if isinstance(v, float) else f"    {k} = {v}")

    print()


def _cmd_cfd(args: argparse.Namespace) -> None:
    from hpe.pipeline import run_cfd_pipeline
    from hpe.sizing import run_sizing

    op = _make_operating_point(args)
    sizing = run_sizing(op)

    print("=" * 60)
    print("  HIGRA PUMP ENGINE — CFD Case Generation")
    print("=" * 60)
    print()
    print(f"  Input: Q={op.flow_rate*3600:.1f} m3/h, H={op.head:.1f} m, n={op.rpm:.0f} rpm")
    print(f"  Output: {args.output}")
    print()

    result = run_cfd_pipeline(
        sizing, args.output,
        run_solver=args.run,
        n_procs=args.procs,
    )

    print(f"  Case directory: {result.case_dir}")
    if result.step_file:
        print(f"  STEP file: {result.step_file}")
    print(f"  OpenFOAM available: {'Yes' if result.openfoam_available else 'No'}")

    if result.ran_simulation:
        print("  Simulation: Completed")
        if result.performance:
            print(f"  Head: {result.performance.head:.1f} m")
            print(f"  Efficiency: {result.performance.total_efficiency:.1%}")
            print(f"  Power: {result.performance.power/1000:.1f} kW")
    else:
        print("  Simulation: Not executed")
        print(f"  Run manually: cd {result.case_dir} && ./run.sh")

    if result.errors:
        print()
        print("  Notes")
        for e in result.errors:
            print(f"    - {e}")

    print()


def _cmd_batch(args: argparse.Namespace) -> None:
    """Batch sizing from JSON input file.

    Input format: list of {flow_rate, head, rpm, name} dicts.
    Output format: json (default) or csv.
    """
    import json
    from hpe.core.models import OperatingPoint
    from hpe.sizing import run_sizing

    with open(args.input) as f:
        jobs = json.load(f)

    results = []
    for job in jobs:
        try:
            op = OperatingPoint(
                flow_rate=job["flow_rate"],
                head=job["head"],
                rpm=job["rpm"],
            )
            r = run_sizing(op)
            entry = {
                "name": job.get("name", ""),
                "flow_rate": job["flow_rate"],
                "head": job["head"],
                "rpm": job["rpm"],
                "nq": round(r.specific_speed_nq, 1),
                "d2_mm": round(r.impeller_d2 * 1000, 1),
                "b2_mm": round(r.impeller_b2 * 1000, 1),
                "eta_pct": round(r.estimated_efficiency * 100, 1),
                "npsh_r": round(r.estimated_npsh_r, 2),
                "power_kw": round(r.estimated_power / 1000, 2),
                "warnings": r.warnings,
            }
            results.append(entry)
            print(f"  ✓ {job.get('name', job['flow_rate'])}: D2={entry['d2_mm']}mm η={entry['eta_pct']}%")
        except Exception as e:
            results.append({"name": job.get("name", ""), "error": str(e)})
            print(f"  ✗ {job.get('name', '')}: {e}", file=sys.stderr)

    if args.format == "csv":
        import csv
        import io as _io
        buf = _io.StringIO()
        if results:
            fieldnames = [k for k in results[0] if k != "warnings"]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: v for k, v in r.items() if k != "warnings"})
        with open(args.output, "w") as f:
            f.write(buf.getvalue())
    else:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ {len(results)} resultados → {args.output}")


def _save_curves_csv(curves, filepath: str) -> None:
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Q_m3s", "Q_m3h", "H_m", "eta_total", "P_W", "P_kW", "NPSHr_m", "eta_h", "T_Nm"])
        for i in range(len(curves.flow_rates)):
            writer.writerow([
                f"{curves.flow_rates[i]:.6f}",
                f"{curves.flow_rates[i]*3600:.2f}",
                f"{curves.heads[i]:.2f}",
                f"{curves.efficiencies[i]:.4f}",
                f"{curves.powers[i]:.1f}",
                f"{curves.powers[i]/1000:.2f}",
                f"{curves.npsh_required[i]:.2f}",
                f"{curves.hydraulic_efficiencies[i]:.4f}",
                f"{curves.torques[i]:.2f}",
            ])


if __name__ == "__main__":
    main()
