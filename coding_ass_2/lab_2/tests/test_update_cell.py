import pytest
import numpy as np

@pytest.mark.parametrize("num", range(5))
def test(map, num):
    """Test that cells are being updated properly by randomly choosing 5 cells to change"""
    og = map[0].ogrid
    idx = np.random.randint(0, 20, size=2)
    val = np.random.random()*10 - 5

    after = np.log(0.5/0.5)+val
    og.update_cell_with_meas_logodds(idx[0], idx[1], val)

    assert og.grid[idx[1],idx[0]] == after, f"Cell {idx} failed to update properly"