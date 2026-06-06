"""Tests for hpe.core.design_state — the unified design object."""

from __future__ import annotations

import pytest

from hpe.core.design_state import (
    DesignState, DesignConstraints, ConstraintReport, STAGES,
)


def _good_sizing() -> dict:
    """A sizing result that satisfies the default constraints."""
    return {
        "specific_speed_nq": 35.0,
        "impeller_d2": 0.30,            # m
        "estimated_efficiency": 0.82,
        "estimated_npsh_r": 3.0,
        "sigma": 0.12,
        "diffusion_ratio": 0.78,
        "velocity_triangles": {"outlet": {"u": 28.0}},
    }


def _op() -> dict:
    return {"flow_rate": 0.05, "head": 30.0, "rpm": 1750}


# ---------------------------------------------------------------------------
# construction & provenance
# ---------------------------------------------------------------------------

def test_from_operating_point_dict():
    st = DesignState.from_operating_point(_op())
    assert st.operating_point["head"] == 30.0
    assert st.stages_run == []
    assert st.id  # auto-assigned


def test_record_stage_tracks_provenance():
    st = DesignState.from_operating_point(_op())
    st.record_stage("sizing", _good_sizing())
    st.record_stage("geometry", {"D2_mm": 300}, note="cadquery")
    assert st.stages_run == ["sizing", "geometry"]
    assert [p.stage for p in st.provenance] == ["sizing", "geometry"]
    assert st.provenance[1].note == "cadquery"


def test_record_stage_rejects_unknown_stage():
    st = DesignState.from_operating_point(_op())
    with pytest.raises(ValueError):
        st.record_stage("bogus", {})


def test_record_stage_accepts_object_with_to_dict():
    class R:
        def to_dict(self):
            return {"x": 1}
    st = DesignState.from_operating_point(_op())
    st.record_stage("surrogate", R())
    assert st.surrogate == {"x": 1}


# ---------------------------------------------------------------------------
# constraint evaluation
# ---------------------------------------------------------------------------

def test_feasible_design_passes():
    st = DesignState.from_operating_point(_op())
    st.record_stage("sizing", _good_sizing())
    report = st.evaluate_constraints()
    assert isinstance(report, ConstraintReport)
    assert report.feasible is True
    assert report.failed == []
    assert st.is_feasible is True


def test_low_efficiency_fails():
    st = DesignState.from_operating_point(_op())
    sizing = _good_sizing()
    sizing["estimated_efficiency"] = 0.55  # below default 0.70
    st.record_stage("sizing", sizing)
    report = st.evaluate_constraints()
    assert report.feasible is False
    assert "min_efficiency" in [c.name for c in report.failed]


def test_excessive_tip_speed_fails():
    st = DesignState.from_operating_point(_op())
    sizing = _good_sizing()
    sizing["velocity_triangles"] = {"outlet": {"u": 60.0}}  # above 45 m/s
    st.record_stage("sizing", sizing)
    assert "max_tip_speed" in [c.name for c in st.evaluate_constraints().failed]


def test_tip_speed_computed_when_missing_from_triangles():
    st = DesignState.from_operating_point(_op())
    sizing = _good_sizing()
    sizing["velocity_triangles"] = {}            # force fallback
    sizing["impeller_d2"] = 0.30                 # u2 = pi*0.30*1750/60 ≈ 27.5 m/s
    st.record_stage("sizing", sizing)
    report = st.evaluate_constraints()
    tip = next(c for c in report.checks if c.name == "max_tip_speed")
    assert tip.value == pytest.approx(27.49, abs=0.1)
    assert tip.ok is True


def test_custom_constraints_npsh_and_sigma():
    st = DesignState.from_operating_point(
        _op(), constraints=DesignConstraints(max_npsh_r=2.0, max_sigma=0.10),
    )
    st.record_stage("sizing", _good_sizing())  # npsh=3.0 > 2.0, sigma=0.12 > 0.10
    failed = [c.name for c in st.evaluate_constraints().failed]
    assert "max_npsh_r" in failed
    assert "max_sigma" in failed


def test_no_sizing_yields_only_input_checks():
    st = DesignState.from_operating_point(_op())
    report = st.evaluate_constraints()
    # Valid Q/H/n and no sizing → feasible with no engineering checks.
    assert report.feasible is True
    assert report.checks == []


# ---------------------------------------------------------------------------
# serialisation round-trip
# ---------------------------------------------------------------------------

def test_to_from_dict_roundtrip():
    st = DesignState.from_operating_point(
        _op(), constraints=DesignConstraints(min_efficiency=0.75),
    )
    st.record_stage("sizing", _good_sizing())
    data = st.to_dict()
    restored = DesignState.from_dict(data)
    assert restored.id == st.id
    assert restored.operating_point == st.operating_point
    assert restored.constraints.min_efficiency == 0.75
    assert restored.sizing == st.sizing
    assert [p.stage for p in restored.provenance] == ["sizing"]
    assert restored.evaluate_constraints().feasible == st.evaluate_constraints().feasible


def test_summary_shape():
    st = DesignState.from_operating_point(_op())
    st.record_stage("sizing", _good_sizing())
    summ = st.summary()
    assert summ["operating_point"]["Q_m3h"] == pytest.approx(180.0)
    assert summ["nq"] == 35.0
    assert summ["feasible"] is True
    assert summ["stages_run"] == ["sizing"]


def test_stages_constant_order():
    assert STAGES == ("sizing", "geometry", "physics", "surrogate", "cfd")


def test_mode_defaults_classic_and_roundtrips():
    from hpe.core.enums import DesignMode
    st = DesignState.from_operating_point(_op())
    assert st.mode == DesignMode.CLASSIC
    st_free = DesignState.from_operating_point(_op(), mode=DesignMode.FREE)
    restored = DesignState.from_dict(st_free.to_dict())
    assert restored.mode == DesignMode.FREE
    assert st_free.to_dict()["mode"] == "free"
    assert st_free.summary()["mode"] == "free"
