"""Tests for the geometry-mode routes (M2).

The router is mounted on a fresh FastAPI app so the test does not import the
full application (which pulls optional deps like optuna).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hpe.api.routes.geometry_modes import router


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_list_backends(client):
    r = client.get("/api/v1/geometry/backends")
    assert r.status_code == 200
    names = {b["name"]: b for b in r.json()["backends"]}
    assert names["parametric"]["mode"] == "classic"
    assert names["implicit"]["mode"] == "free"
    assert names["implicit"]["capabilities"]["supports_voxel"] is True


def test_generate_classic_mode(client):
    r = client.post("/api/v1/geometry/generate",
                    json={"Q": 0.05, "H": 30.0, "n": 1750, "mode": "classic"})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "parametric"
    assert body["mesh_strategy"] == "body_fitted"
    assert "D2_mm" in body["artifact"]["params"]
    assert body["summary"]["mode"] == "classic"
    assert "geometry" in body["summary"]["stages_run"]


def test_generate_free_mode_not_implemented(client):
    r = client.post("/api/v1/geometry/generate",
                    json={"Q": 0.05, "H": 30.0, "n": 1750, "mode": "free"})
    assert r.status_code == 501
    assert "not implemented" in r.json()["detail"].lower()


def test_generate_unknown_mode_422(client):
    r = client.post("/api/v1/geometry/generate",
                    json={"Q": 0.05, "H": 30.0, "n": 1750, "mode": "bogus"})
    assert r.status_code == 422


def test_generate_defaults_to_classic(client):
    r = client.post("/api/v1/geometry/generate",
                    json={"Q": 0.05, "H": 30.0, "n": 1750})
    assert r.status_code == 200
    assert r.json()["mode"] == "classic"
