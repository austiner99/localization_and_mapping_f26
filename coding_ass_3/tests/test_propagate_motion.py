from lab3.particle_filter_localization import ParticleFilterLocalizer
from lab3.occupancy_grid_map import OccupancyGrid
import numpy as np
import os

def test(omap):
    # set everything up
    n = 100
    pf = ParticleFilterLocalizer(omap, [-20, 40], [-20, 20], n, np.zeros(4), np.zeros(3))  

    # load our data that has the answers :)
    file = 'prop_motion_test.npz'
    if os.getcwd()[-5:] != "tests":
        file = "tests/" + file
    data = np.load(file)
    pf.particles = data['before']
    deltas = data['deltas']
    expected = data['expected']
    
    # Check weights
    pf.propagate_motion(deltas)
    got = pf.particles

    assert np.allclose(expected, got)
    