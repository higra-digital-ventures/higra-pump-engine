"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from hpe.api.app import app

client = TestClient(app)


class TestHealthCheck:
    def test_health(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestSizingEndpoint:
    def test_sizing_success(self) -> None:
        response = client.post("/api/v1/sizing", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["specific_speed_nq"] > 0
        assert data["impeller_d2"] > 0
        assert data["blade_count"] >= 5
        assert 0 < data["estimated_efficiency"] < 1

    def test_sizing_validation(self) -> None:
        response = client.post("/api/v1/sizing", json={
            "flow_rate": -1, "head": 30.0, "rpm": 1750,
        })
        assert response.status_code == 422  # Validation error

    def test_sizing_missing_field(self) -> None:
        response = client.post("/api/v1/sizing", json={
            "flow_rate": 0.05, "head": 30.0,
        })
        assert response.status_code == 422


class TestCurvesEndpoint:
    def test_curves_success(self) -> None:
        response = client.post("/api/v1/curves", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750, "n_points": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["points"]) == 10
        assert data["bep_efficiency"] > 0

    def test_curves_points_structure(self) -> None:
        response = client.post("/api/v1/curves", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750, "n_points": 5,
        })
        data = response.json()
        point = data["points"][0]
        assert "flow_rate" in point
        assert "head" in point
        assert "efficiency" in point
        assert "power" in point


class TestOptimizeEndpoint:
    def test_optimize_nsga2(self) -> None:
        response = client.post("/api/v1/optimize", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750,
            "method": "nsga2", "pop_size": 10, "n_gen": 5, "seed": 42,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["pareto_front"]) > 0
        assert data["n_evaluations"] > 0

    def test_optimize_bayesian(self) -> None:
        pytest.importorskip("optuna", reason="Bayesian optimize endpoint requires Optuna")
        response = client.post("/api/v1/optimize", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750,
            "method": "bayesian", "n_gen": 10, "seed": 42,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["pareto_front"]) > 0
