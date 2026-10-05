from lab3.occupancy_grid_map import OccupancyGrid
from lab3.particle_filter_localization import ParticleFilterLocalizer
import numpy as np
from scipy.stats import norm
import pytest

@pytest.mark.parametrize("ans", [(25,  20, .25), 
                                (4, 4.5, norm.pdf(0.5)*.5),
                                (5, 3, norm.pdf(2)*.5)])
def test(ans, omap):
    z_k, z_t, ans = ans

    # set everything up
    n = 100
    pf = ParticleFilterLocalizer(omap, [-20, 40], [-20, 20], n, np.array([0.5, 0, 0.25, 0.25]), np.zeros(3))
    pf.particles[0] = np.array([1.5, 1.5, 0])

    weight = pf.update_weight(np.array([z_k]), np.array([z_t]),
                                    pf.alphas)

    # Should some from p_hit/p_max and .0125 from p_random
    assert np.isclose(weight, ans+.0125)