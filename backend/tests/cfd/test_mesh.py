"""Tests for mesh generators."""


from hpe.cfd.mesh.blockmesh import generate_blockmesh_dict
from hpe.cfd.mesh.snappy import generate_snappy_dict


class TestBlockMesh:
    def test_generates_valid_dict(self) -> None:
        content = generate_blockmesh_dict(0.25, 0.02)
        assert "FoamFile" in content
        assert "blockMeshDict" in content
        assert "vertices" in content
        assert "blocks" in content
        assert "boundary" in content

    def test_contains_patches(self) -> None:
        content = generate_blockmesh_dict(0.25, 0.02)
        assert "inlet" in content
        assert "outlet" in content
        assert "walls" in content

    def test_domain_scales_with_d2(self) -> None:
        small = generate_blockmesh_dict(0.10, 0.01)
        large = generate_blockmesh_dict(0.50, 0.05)
        # Larger D2 should have larger domain coordinates
        assert "0.500000" in large  # r_domain = 0.5 * 2 / 2
        assert "0.100000" in small


class TestSnappy:
    def test_generates_valid_dict(self) -> None:
        content = generate_snappy_dict("impeller.stl", 0.25)
        assert "FoamFile" in content
        assert "snappyHexMeshDict" in content
        assert "impeller" in content

    def test_contains_sections(self) -> None:
        content = generate_snappy_dict("impeller.stl", 0.25)
        assert "castellatedMeshControls" in content
        assert "snapControls" in content
        assert "addLayersControls" in content
        assert "meshQualityControls" in content

    def test_geometry_file_referenced(self) -> None:
        content = generate_snappy_dict("my_pump.stl", 0.25)
        assert "my_pump.stl" in content
        assert "my_pump" in content
