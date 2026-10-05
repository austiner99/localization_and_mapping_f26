import pytest
import numpy as np

expected_cells = {
      -90: [[10,10],[11,10],[12,10],[13,10],[14,10],[15,10]],
        0: [[10,10],[10,11],[10,12],[10,13],[10,14],[10,15]],
       90: [[10,10],[ 9,10],[ 8,10],[ 7,10],[ 6,10],[ 5,10]],
      180: [[10,10],[10, 9],[10, 8],[10, 7],[10, 6],[10, 5]],
}

@pytest.mark.parametrize("theta", list(expected_cells.keys()))
def test(map, theta):
    """Tests to make sure the correct cells are being updated"""
    opm, x_t = map
    got = np.array(opm.find_cells_to_update_for_ray(x_t, theta, 5))

    # verify there's no duplicates
    assert got.shape[0] == np.unique(got, axis=0).shape[0], "There was a duplicate cell"

    expected = expected_cells[theta]
    got = got.tolist()

    # verify there was no incorrect hits
    for g in got:
        assert g in expected, f"Idx {g} should not have been included"

    # verify we got all of them
    for e in expected:
        assert e in got, f"Missing idx {e}"