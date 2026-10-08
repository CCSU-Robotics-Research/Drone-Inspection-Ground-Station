"""Tests for the tiling math in DINO detector."""

import pytest

torch = pytest.importorskip("torch")

from ai.dino import STRIDE, TILE, tile_starts  # noqa: E402


class TestTileStarts:

    def test_frame_smaller_than_tile(self):
        assert tile_starts(100) == [0]

    def test_exact_tile(self):
        assert tile_starts(TILE) == [0]

    def test_strided_coverage(self):
        starts = tile_starts(500)
        assert starts == [0, 112, 224, 276]
        assert starts[-1] + TILE == 500

    def test_exact_multiple(self):
        starts = tile_starts(TILE + STRIDE)
        assert starts == [0, STRIDE]

    def test_720p_grid(self):
        assert len(tile_starts(720)) == 6
        assert len(tile_starts(1280)) == 11
