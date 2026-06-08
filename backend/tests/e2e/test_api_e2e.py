"""End-to-end API tests — full workflow through all endpoints.

Tests the complete user journey: auth → sizing → curves → losses →
stress → inverse → geometry, verifying data consistency across calls.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hpe.api.app import app

client = TestClient(app)

# Shared state across the test flow
_state: dict = {}


class TestE2EFullWorkflow:
    """Test the complete design workflow end-to-end."""

    def test_01_health(self) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_02_register(self) -> None:
        r = client.post("/api/v1/auth/register", json={
            "email": "e2e@higra.com.br", "password": "test123456", "name": "E2E Test",
        })
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        _state["token"] = data["access_token"]
        _state["user"] = data["user"]

    def test_03_login(self) -> None:
        r = client.post("/api/v1/auth/login", json={
            "email": "e2e@higra.com.br", "password": "test123456",
        })
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_04_me(self) -> None:
        r = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {_state['token']}",
        })
        assert r.status_code == 200
        assert r.json()["email"] == "e2e@higra.com.br"

    def test_10_sizing(self) -> None:
        r = client.post("/api/v1/sizing", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["impeller_d2"] > 0
        assert data["blade_count"] > 0
        assert data["estimated_efficiency"] > 0.5
        _state["sizing"] = data

    def test_11_sizing_consistency(self) -> None:
        """Verify sizing output fields are physically consistent."""
        s = _state["sizing"]
        assert s["impeller_d1"] < s["impeller_d2"]  # D1 < D2
        assert s["impeller_b2"] > 0
        assert 10 < s["beta1"] < 50  # Typical range
        assert 15 < s["beta2"] < 50
        assert s["estimated_npsh_r"] > 0
        assert s["specific_speed_nq"] > 0

    def test_20_curves(self) -> None:
        r = client.post("/api/v1/curves", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750, "n_points": 15,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["points"]) == 15
        assert data["bep_efficiency"] > 0
        _state["curves"] = data

    def test_21_curves_bep_in_range(self) -> None:
        """BEP flow should be within the curve range."""
        c = _state["curves"]
        flows = [p["flow_rate"] for p in c["points"]]
        assert min(flows) <= c["bep_flow"] <= max(flows)

    def test_30_losses(self) -> None:
        r = client.post("/api/v1/losses", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["total_head_loss"] > 0
        assert data["profile_loss_total"] > 0
        assert data["loss_coefficient"] > 0
        assert data["loss_coefficient"] < 1.0
        _state["losses"] = data

    def test_31_losses_breakdown_sums(self) -> None:
        """Individual losses should sum to total."""
        d = _state["losses"]
        individual = (
            d["profile_loss_total"] + d["tip_leakage"] + d["endwall_total"]
            + d["mixing"] + d["incidence"] + d["recirculation"]
        )
        assert abs(individual - d["total_head_loss"]) < 0.001

    def test_40_stress(self) -> None:
        r = client.post("/api/v1/stress", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["von_mises_max"] > 0
        assert data["sf_yield"] > 0
        assert isinstance(data["is_safe"], bool)
        assert data["first_natural_freq"] > 0

    def test_50_inverse(self) -> None:
        r = client.post("/api/v1/inverse", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750,
            "loading_type": "mid_loaded",
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["beta_inlet"]) == 5
        assert len(data["wrap_angles"]) == 5
        assert data["diffusion_ratio"] > 0

    def test_60_geometry_impeller(self) -> None:
        pytest.importorskip("cadquery", reason="3D impeller endpoint requires CadQuery")
        r = client.post("/api/v1/geometry/impeller", json={
            "flow_rate": 0.05, "head": 30.0, "rpm": 1750,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["blades"]) > 0
        assert len(data["blades"][0]) > 0
        assert data["blade_count"] == _state["sizing"]["blade_count"]


class TestE2EEdgeCases:
    """Test error handling and edge cases."""

    def test_invalid_flow_rate(self) -> None:
        r = client.post("/api/v1/sizing", json={
            "flow_rate": -1, "head": 30.0, "rpm": 1750,
        })
        assert r.status_code == 422

    def test_missing_fields(self) -> None:
        r = client.post("/api/v1/sizing", json={"flow_rate": 0.05})
        assert r.status_code == 422

    def test_duplicate_register(self) -> None:
        client.post("/api/v1/auth/register", json={
            "email": "dup@test.com", "password": "123456", "name": "Dup",
        })
        r = client.post("/api/v1/auth/register", json={
            "email": "dup@test.com", "password": "123456", "name": "Dup2",
        })
        assert r.status_code == 400

    def test_wrong_password(self) -> None:
        r = client.post("/api/v1/auth/login", json={
            "email": "e2e@higra.com.br", "password": "wrongpassword",
        })
        assert r.status_code == 401

    def test_extreme_operating_point(self) -> None:
        """Very high head pump — should still return valid result."""
        r = client.post("/api/v1/sizing", json={
            "flow_rate": 0.001, "head": 500.0, "rpm": 3500,
        })
        assert r.status_code == 200
        assert r.json()["impeller_d2"] > 0

    def test_very_small_pump(self) -> None:
        r = client.post("/api/v1/sizing", json={
            "flow_rate": 0.0001, "head": 5.0, "rpm": 2900,
        })
        assert r.status_code == 200
