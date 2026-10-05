from lab3.particle_filter_localization import ParticleFilterLocalizer
from lab3.occupancy_grid_map import OccupancyGrid
import numpy as np

def test(omap):
    # set everything up
    n = 100
    pf = ParticleFilterLocalizer(omap, [-20, 40], [-20, 20], n, np.zeros(4), np.zeros(3))

    # Setup distribution
    weights = np.random.rand(n)
    weights /= np.sum(weights)

    # resample a bunch of times to match distribution
    num_times = 1000
    out_particles = np.zeros(n*num_times)
    for i in range(num_times):
        pf.particles[:,0] = np.arange(n)
        pf.weights = weights
        pf.resample_particles()
        out_particles[i*n:n*(i+1)] = pf.particles[:,0]

    # check that our resampled particles match our distribution
    out_dist = np.zeros(n)
    for i in range(n):
        out_dist[i] += np.sum( out_particles == i )
    out_dist /= np.sum(out_dist)

    assert np.allclose(weights, out_dist, atol=1e-2)