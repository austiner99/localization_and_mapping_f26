from lab3.occupancy_grid_map import OccupancyGrid
from lab3.particle_filter_localization import ParticleFilterLocalizer
import numpy as np
from scipy.stats import norm
import pytest

@pytest.mark.parametrize("ans", [( 0, 25),
                                ( 45, 1.42),
                                ( 90, 1),
                                (135, 1.42),
                                (180, 4),
                                (225, 1.42),
                                (270, 5)])
def test(ans, omap):
    angle, expected = ans

    # set everything up
    n = 100
    pf = ParticleFilterLocalizer(omap, [-20, 40], [-20, 20], n, np.array([0.5, 0, 0.25, 0.25]), np.zeros(3))
    pf.particles[0] = np.array([1.5, 1.5, 0])

    got = pf.expected_measurement(np.array([angle]), pf.particles[0],
                                pf.priormap.grid,
                                pf.priormap.xlim[0], pf.priormap.xlim[1],
                                pf.priormap.ylim[0], pf.priormap.ylim[1],
                                pf.priormap.resolution)[0]

    assert np.isclose(got, expected, atol=0.5)