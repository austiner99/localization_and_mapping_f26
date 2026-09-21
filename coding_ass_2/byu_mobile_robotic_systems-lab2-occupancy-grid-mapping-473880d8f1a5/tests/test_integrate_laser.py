import numpy as np

def test(map):
    """Test to make sure all cells are being found and updated properly"""
    opm, x_t = map
    z_theta_t = -90
    z_t = 5

    exp_free  = np.log(0.1/0.9)
    exp_occup = np.log(0.2/0.8)
    exp_prior = np.log(0.5/0.5)

    opm.integrate_laser_range_ray(x_t, z_theta_t, z_t)
    
    # for free space
    for i in range(5):
        assert opm.ogrid.grid[10,10+i] == exp_free

    # hit space
    assert opm.ogrid.grid[10,15] == exp_occup

    # and after hit space (should remain the same)
    for i in range(4):
        assert opm.ogrid.grid[10,16+i] == exp_prior