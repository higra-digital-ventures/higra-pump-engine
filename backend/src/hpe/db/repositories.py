"""CRUD repository functions for HPE database.

Each function is a thin wrapper over raw SQL, returning dicts.
"""
from __future__ import annotations

import json
import logging
from uuid import uuid4

from hpe.db.connection import get_connection

log = logging.getLogger(__name__)


# ─── Projects ─────────────────────────────────────────────────────────────────

def create_project(name: str, machine_type: str = "centrifugal_pump",
                   description: str = "", user_id: str | None = None) -> dict:
    uid = str(uuid4())
    user_id = user_id or "00000000-0000-0000-0000-000000000001"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO projects (id, user_id, name, description, machine_type)
                   VALUES (%s, %s, %s, %s, %s) RETURNING *""",
                (uid, user_id, name, description, machine_type),
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
        conn.commit()
    return dict(zip(cols, row))


def list_projects(user_id: str | None = None, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            sql = """
                SELECT p.*, COUNT(d.id) AS n_sizing_results
                FROM projects p
                LEFT JOIN designs d ON d.project_id = p.id
                {where}
                GROUP BY p.id
                ORDER BY p.updated_at DESC
                LIMIT %s
            """
            if user_id:
                cur.execute(sql.format(where="WHERE p.user_id=%s"), (user_id, limit))
            else:
                cur.execute(sql.format(where=""), (limit,))
            cols = [d[0] for d in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                d["n_sizing_results"] = int(d.get("n_sizing_results") or 0)
                rows.append(d)
            return rows


def update_project(project_id: str, name: str | None = None, description: str | None = None) -> dict:
    """Update a project's name and/or description."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            parts, vals = [], []
            if name is not None:
                parts.append("name=%s")
                vals.append(name)
            if description is not None:
                parts.append("description=%s")
                vals.append(description)
            if not parts:
                return get_project(project_id) or {}
            parts.append("updated_at=NOW()")
            vals.append(project_id)
            cur.execute(
                f"UPDATE projects SET {', '.join(parts)} WHERE id=%s RETURNING *",
                tuple(vals),
            )
            row = cur.fetchone()
            if not row:
                return {}
            cols = [d[0] for d in cur.description]
        conn.commit()
    return dict(zip(cols, row))


def get_project(project_id: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM projects WHERE id=%s", (project_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


# ─── Designs ──────────────────────────────────────────────────────────────────

def save_design(project_id: str, sizing_result: dict, op: dict,
                overrides: dict | None = None, notes: str = "") -> dict:
    """Persist a sizing result to the designs table."""
    uid = str(uuid4())
    r = sizing_result
    ov = overrides or {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO designs (
                    id, project_id, flow_rate_m3s, head_m, rpm, fluid_density,
                    machine_type, nq, impeller_type, d2_m, d1_m, b2_m,
                    blade_count, beta1_deg, beta2_deg, eta_total, power_w,
                    npsh_r, sigma, diffusion_ratio, throat_area_m2, slip_factor,
                    pmin_pa, convergence_iters, endwall_loss, leakage_loss_m,
                    profile_loss_total, warnings, result_json,
                    override_d2, override_b2, override_d1,
                    tip_clearance_mm, roughness_ra_um, notes
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                ) RETURNING *""",
                (
                    uid, project_id,
                    op.get("flow_rate"), op.get("head"), op.get("rpm"),
                    op.get("fluid_density", 998.0), op.get("machine_type", "centrifugal_pump"),
                    r.get("specific_speed_nq"), r.get("impeller_type"),
                    r.get("impeller_d2"), r.get("impeller_d1"), r.get("impeller_b2"),
                    r.get("blade_count"), r.get("beta1"), r.get("beta2"),
                    r.get("estimated_efficiency"), r.get("estimated_power"),
                    r.get("estimated_npsh_r"), r.get("sigma"),
                    r.get("diffusion_ratio"), r.get("throat_area"),
                    r.get("slip_factor"), r.get("pmin_pa"), r.get("convergence_iterations"),
                    r.get("endwall_loss"), r.get("leakage_loss_m"), r.get("profile_loss_total"),
                    r.get("warnings", []), json.dumps(r),
                    ov.get("override_d2"), ov.get("override_b2"), ov.get("override_d1"),
                    ov.get("tip_clearance_mm"), ov.get("roughness_ra_um"),
                    notes,
                ),
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
        conn.commit()
    return dict(zip(cols, row))


def list_designs(project_id: str, limit: int = 100) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM designs WHERE project_id=%s ORDER BY created_at DESC LIMIT %s",
                (project_id, limit),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_design(design_id: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM designs WHERE id=%s", (design_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))


# ─── Performance curves ───────────────────────────────────────────────────────

def save_performance_curve(design_id: str, points: list[dict]) -> int:
    """Bulk-insert performance curve points. Returns number inserted."""
    if not points:
        return 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            args = [
                (str(uuid4()), design_id, p.get("flow_rate"), p.get("head"),
                 p.get("efficiency"), p.get("power"), p.get("npsh_required"),
                 p.get("is_unstable", False))
                for p in points
            ]
            cur.executemany(
                """INSERT INTO performance_curves
                   (id, design_id, flow_rate_m3s, head_m, efficiency, power_w,
                    npsh_r, is_unstable)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                args,
            )
        conn.commit()
    return len(args)


def get_performance_curve(design_id: str) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM performance_curves WHERE design_id=%s ORDER BY flow_rate_m3s",
                (design_id,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ─── Delete ───────────────────────────────────────────────────────────────────

def delete_design(design_id: str) -> bool:
    """Delete a single design and its performance curves (CASCADE)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM performance_curves WHERE design_id=%s", (design_id,)
            )
            cur.execute(
                "DELETE FROM designs WHERE id=%s RETURNING id", (design_id,)
            )
            deleted = cur.fetchone() is not None
        conn.commit()
    return deleted


def delete_project(project_id: str) -> bool:
    """Delete a project and ALL its designs + performance curves (CASCADE)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Delete curves for all designs of this project
            cur.execute(
                """DELETE FROM performance_curves
                   WHERE design_id IN (
                       SELECT id FROM designs WHERE project_id=%s
                   )""",
                (project_id,),
            )
            # Delete all designs
            cur.execute(
                "DELETE FROM designs WHERE project_id=%s", (project_id,)
            )
            # Delete the project itself
            cur.execute(
                "DELETE FROM projects WHERE id=%s RETURNING id", (project_id,)
            )
            deleted = cur.fetchone() is not None
        conn.commit()
    return deleted
