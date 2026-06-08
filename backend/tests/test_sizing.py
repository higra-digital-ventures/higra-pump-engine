"""Tests for 1D meanline sizing."""
import pytest
from hpe.core.models import OperatingPoint
from hpe.sizing.meanline import run_sizing

class TestSizing:
    def test_basic_pump(self):
        """Dragon template: Q=100m3/h, H=32m, n=1750rpm"""
        op = OperatingPoint(flow_rate=100/3600, head=32, rpm=1750)
        r = run_sizing(op)
        assert 15 < r.specific_speed_nq < 35
        assert 200 < r.impeller_d2 * 1000 < 400  # D2 in mm
        assert 5 <= r.blade_count <= 8
        assert 0.60 < r.estimated_efficiency < 0.95
        assert r.estimated_npsh_r > 0
        assert r.estimated_power > 0

    def test_high_nq_pump(self):
        """High flow pump: Q=1000m3/h, H=20m, n=1750rpm"""
        op = OperatingPoint(flow_rate=1000/3600, head=20, rpm=1750)
        r = run_sizing(op)
        assert 60 < r.specific_speed_nq < 120
        assert r.blade_count <= 6  # fewer blades at high Nq

    def test_low_nq_pump(self):
        """High pressure pump: Q=50m3/h, H=80m, n=3550rpm"""
        op = OperatingPoint(flow_rate=50/3600, head=80, rpm=3550)
        r = run_sizing(op)
        assert 10 < r.specific_speed_nq < 25

    def test_invalid_params(self):
        """Should handle edge cases."""
        op = OperatingPoint(flow_rate=0.001, head=1, rpm=500)
        r = run_sizing(op)
        assert r.specific_speed_nq > 0

    def test_velocity_triangles(self):
        """Velocity triangles should be physically consistent."""
        op = OperatingPoint(flow_rate=100/3600, head=32, rpm=1750)
        r = run_sizing(op)
        vt = r.velocity_triangles
        assert "inlet" in vt
        assert "outlet" in vt
        assert vt["inlet"]["u"] > 0
        assert vt["outlet"]["u"] > vt["inlet"]["u"]  # u2 > u1
        assert vt["euler_head"] > 0
