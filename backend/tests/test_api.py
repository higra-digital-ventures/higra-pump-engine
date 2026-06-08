"""Tests for API endpoints."""
from fastapi.testclient import TestClient
from hpe.api.app import app

client = TestClient(app)

class TestHealthCheck:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

class TestSizingAPI:
    def test_sizing_endpoint(self):
        resp = client.post("/api/v1/sizing", json={
            "flow_rate": 100/3600, "head": 32, "rpm": 1750,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "specific_speed_nq" in data
        assert "impeller_d2" in data

    def test_curves_endpoint(self):
        resp = client.post("/api/v1/curves", json={
            "flow_rate": 100/3600, "head": 32, "rpm": 1750,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "points" in data
        assert len(data["points"]) > 0

    def test_losses_endpoint(self):
        resp = client.post("/api/v1/losses", json={
            "flow_rate": 100/3600, "head": 32, "rpm": 1750,
        })
        assert resp.status_code == 200

class TestVersionsAPI:
    def test_save_and_list(self):
        # Save
        resp = client.post("/api/v1/versions", json={
            "operating_point": {"flow_rate": 0.0278, "head": 32, "rpm": 1750},
            "sizing_result": {"specific_speed_nq": 21.7, "impeller_d2": 0.288},
        })
        assert resp.status_code in [200, 201]

        # List
        resp = client.get("/api/v1/versions")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

class TestTemplatesAPI:
    def test_list_templates(self):
        resp = client.get("/api/v1/templates")
        assert resp.status_code == 200
        templates = resp.json()
        assert len(templates) >= 10
