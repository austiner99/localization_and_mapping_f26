from lab3.occupancy_grid_map import OccupancyGrid
from lab3.particle_filter_localization import ParticleFilterLocalizer
import numpy as np
from scipy.stats import norm
import os

def datafile():
    file = "data/localization-dataset.npz"
    if os.getcwd()[-5:] == "tests":
        file = "../" + file

    data = np.load(file)

    u = data['U_t']
    z = data["Z_tp1"]
    angles = data["angles"]

    return z, angles

def test(omap):
    # set everything up
    n = 4
    pf = ParticleFilterLocalizer(omap, [-20, 40], [-20, 20], n,
                                    np.array([0.5, 0, 0.25, 0.25]), np.zeros(3))
    pf.particles = np.array([[1.5, 1.5, 0],
                            [20, 20, 0],
                            [-10, -10, 90],
                            [-15, 15, 270]])

    # Do the first step, all REALLY bad particles should be removed easily
    z, angles = datafile()
    u = np.array([.01, .01, 0])
    pf.iterate(u, z[0], angles)

    assert np.allclose(pf.particles, np.array([1.51, 1.51, 0]))