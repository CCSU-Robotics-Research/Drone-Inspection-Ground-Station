"""Tests for the tiling math in DINO detector."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai.dino import (  # noqa: E402
    STRIDE,
    TILE,
    assemble,
    tile_starts,
)


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

    def test_720p_grid_shape(self):
        assert len(tile_starts(720)) == 6
        assert len(tile_starts(1280)) == 11


class TestAssemble:

    def test_blocks_stitch_row_major(self):
        blocks = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
        grid = assemble(blocks, 2, 2)
        assert grid.shape == (4, 4)
        assert grid[0, 0] == 0 and grid[0, 2] == 4
        assert grid[2, 0] == 8 and grid[3, 3] == 15
