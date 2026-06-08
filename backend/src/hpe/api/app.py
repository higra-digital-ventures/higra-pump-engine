"""FastAPI application for Higra Pump Engine."""

from __future__ import annotations

# Load .env at startup
from pathlib import Path
import os
_env = Path(__file__).parents[3] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hpe.api.routes.sizing import router as sizing_router
from hpe.api.routes.analysis import router as analysis_router
from hpe.api.routes.geometry import router as geometry_router
from hpe.api.routes.report import router as report_router
from hpe.api.routes.ws_optimize import router as ws_optimize_router
from hpe.api.routes.surrogate import router as surrogate_router
from hpe.api.routes.inverse_design import router as inverse_design_router
from hpe.api.routes.optimize_ext import router as optimize_ext_router
from hpe.api.routes.io_routes import router as io_router
from hpe.api.routes.blade_ext import router as blade_ext_router
from hpe.api.auth import router as auth_router
from hpe.api.routes.design_db_routes import router as design_db_router
from hpe.api.routes.db_routes import router as db_router
from hpe.api.routes.convergence_routes import router as convergence_router
from hpe.api.routes.inverse_design_3d import router as inverse_design_3d_router
from hpe.api.routes.volute_routes import router as volute_router
from hpe.api.routes.mri_routes import router as mri_router
from hpe.api.routes.turbotype_routes import router as turbotype_router
from hpe.api.routes.lete_routes import router as lete_router
from hpe.api.routes.noise_routes import router as noise_router
from hpe.api.routes.batch_routes import router as batch_router
from hpe.api.routes.cfd_loop_routes import router as cfd_loop_router
from hpe.api.routes.phase_17_20_routes import router as cfd_advanced_router
from hpe.api.routes.physics_routes import router as physics_router
from hpe.api.routes.infra_routes import router as infra_router
from hpe.api.routes.rrs_routes import router as rrs_router
from hpe.api.routes.blade_collision_routes import router as blade_collision_router
from hpe.api.routes.template_routes import router as template_router
from hpe.api.routes.domain_routes import router as domain_router
from hpe.api.routes.udp_routes import router as udp_router
from hpe.api.routes.blockage_routes import router as blockage_router
from hpe.api.routes.ansys_routes import router as ansys_router
from hpe.api.routes.lean_sweep_routes import router as lean_sweep_router
from hpe.api.routes.version_routes import router as version_router
from hpe.api.volute_endpoint import router as volute_v2_router
from hpe.api.websocket import router as pipeline_ws_router
from hpe.api.routes.assistant_routes import router as assistant_router
from hpe.api.routes.geometry_modes import router as geometry_modes_router

app = FastAPI(
    title="Higra Pump Engine",
    description="API for hydraulic turbomachinery design, analysis, and optimization",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from hpe.api.middleware import RateLimitMiddleware, MultitenancyMiddleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=120)
app.add_middleware(MultitenancyMiddleware)

app.include_router(auth_router)
app.include_router(sizing_router)
app.include_router(analysis_router)
app.include_router(geometry_router)
app.include_router(inverse_design_router)
app.include_router(report_router)
app.include_router(ws_optimize_router)
app.include_router(surrogate_router)
app.include_router(optimize_ext_router)
app.include_router(io_router)
app.include_router(blade_ext_router)
app.include_router(design_db_router)
app.include_router(db_router)
app.include_router(convergence_router)
app.include_router(inverse_design_3d_router)
app.include_router(volute_router)
app.include_router(mri_router)
app.include_router(turbotype_router)
app.include_router(lete_router)
app.include_router(noise_router)
app.include_router(batch_router)
app.include_router(cfd_loop_router)
app.include_router(cfd_advanced_router)
app.include_router(physics_router)
app.include_router(infra_router)
app.include_router(rrs_router)
app.include_router(blade_collision_router)
app.include_router(template_router)
app.include_router(domain_router)
app.include_router(udp_router)
app.include_router(blockage_router)
app.include_router(ansys_router)
app.include_router(lean_sweep_router)
app.include_router(version_router)
app.include_router(volute_v2_router)
app.include_router(pipeline_ws_router)
app.include_router(assistant_router)
app.include_router(geometry_modes_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "higra-pump-engine"}
