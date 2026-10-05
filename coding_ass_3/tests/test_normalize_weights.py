from lab3.particle_filter_localization import ParticleFilterLocalizer
from lab3.occupancy_grid_map import OccupancyGrid
import numpy as np


def test(omap):
    # set everything up
    n = 100
    pf = ParticleFilterLocalizer(omap, [-20, 40], [-20, 20], n, np.zeros(4), np.zeros(3))  

    # Set random weights
    pf.weights = np.random.uniform(0, 100, size=n)

    pf.normalize_weights()

    assert np.isclose(1, np.sum(pf.weights))