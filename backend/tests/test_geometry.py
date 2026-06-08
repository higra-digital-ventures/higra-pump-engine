"""Tests for 3D geometry generation."""
import math
from fastapi.testclient import TestClient
from hpe.api.app import app

client = TestClient(app)

class TestGeometry:
    def test_impeller_basic(self):
        """Generate impeller geometry and check structure."""
        resp = client.post("/api/v1/geometry/impeller", json={
            "flow_rate": 100/3600, "head": 32, "rpm": 1750,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["blade_count"] >= 5
        assert len(data["blade_surfaces"]) == data["blade_count"]
        assert len(data["hub_profile"]) > 10
        assert len(data["shroud_profile"]) > 10

    def test_blade_thickness(self):
        """PS-SS distance should be ~1.3% D2."""
        resp = client.post("/api/v1/geometry/impeller", json={
            "flow_rate": 100/3600, "head": 32, "rpm": 1750,
        })
        data = resp.json()
        D2 = data["d2"] * 1000
        surf = data["blade_surfaces"][0]
        n_c = len(surf["ps"][0])
        dists = [math.sqrt(sum((surf["ps"][0][c][k]-surf["ss"][0][c][k])**2
                 for k in ["x","y","z"])) for c in range(n_c)]
        t_ratio = max(dists) / D2
        assert 0.008 < t_ratio < 0.025  # 0.8% to 2.5%

    def test_blade_within_d2(self):
        """All blade points should be within D2 radius."""
        resp = client.post("/api/v1/geometry/impeller", json={
            "flow_rate": 100/3600, "head": 32, "rpm": 1750,
        })
        data = resp.json()
        D2 = data["d2"] * 1000
        for surf in data["blade_surfaces"]:
            for row in surf["ps"]:
                for p in row:
                    r = math.sqrt(p["x"]**2 + p["y"]**2)
                    assert r <= D2/2 + 1  # +1mm tolerance

    def test_resolution_presets(self):
        """Resolution presets should change point counts."""
        for preset, expected_chord in [("low", 30), ("medium", 60), ("high", 89)]:
            resp = client.post("/api/v1/geometry/impeller", json={
                "flow_rate": 100/3600, "head": 32, "rpm": 1750,
                "resolution_preset": preset,
            })
            data = resp.json()
            n_c = len(data["blade_surfaces"][0]["ps"][0])
            assert n_c == expected_chord

    def test_wrap_angle(self):
        """Wrap angle should be 80-200 degrees."""
        resp = client.post("/api/v1/geometry/impeller", json={
            "flow_rate": 100/3600, "head": 32, "rpm": 1750,
        })
        data = resp.json()
        wrap = abs(data.get("actual_wrap_angle", 0))
        assert 80 < wrap < 200
