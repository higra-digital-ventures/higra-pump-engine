"""Tests for design database (preliminary design correlations)."""
from hpe.sizing.design_db import get_design_recommendation, list_machine_types, DESIGN_DATABASE


class TestDesignDB:
    def test_all_machine_types_present(self):
        types = list_machine_types()
        ids = [t['id'] for t in types]
        assert 'centrifugal_pump' in ids
        assert 'francis_turbine' in ids
        assert 'centrifugal_compressor' in ids

    def test_centrifugal_pump_recommendation(self):
        r = get_design_recommendation('centrifugal_pump', nq=30.0)
        assert r.machine_type == 'centrifugal_pump'
        assert 5 <= r.blade_count_recommended <= 9
        assert 17 <= r.beta2_recommended_deg <= 35
        assert 0.6 < r.eta_expected < 0.95
        assert r.nq_assessment in ('optimal', 'within_range', 'below_range', 'above_range')

    def test_nq_below_range_warning(self):
        r = get_design_recommendation('centrifugal_pump', nq=3.0)
        assert r.nq_assessment == 'below_range'
        assert len(r.warnings) > 0

    def test_splitter_recommended_for_compressor(self):
        r = get_design_recommendation('centrifugal_compressor', nq=50.0)
        assert r.splitter_recommended

    def test_blade_count_override(self):
        r = get_design_recommendation('centrifugal_pump', nq=30.0, blade_count_override=7)
        assert r.blade_count_recommended == 7

    def test_all_types_return_valid_result(self):
        for type_id, db in DESIGN_DATABASE.items():
            r = get_design_recommendation(type_id, nq=db.nq_optimal)
            assert r.eta_expected > 0
            assert r.blade_count_recommended >= 3
