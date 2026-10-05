import numpy as np

z_t = 5
idx_y = 10

def test_free(map):
    """Test to make sure it returns free space log-prob"""
    opm, x_t = map
    idx_x = 10

    l = opm.laser_range_inverse_sensor_model(idx_x, idx_y, x_t, z_t)

    assert l == np.log(.1/.9)

def test_occup(map):
    """Test to make sure it returns occup space log-prob"""
    opm, x_t = map
    idx_x = 15

    l = opm.laser_range_inverse_sensor_model(idx_x, idx_y, x_t, z_t)

    assert l == np.log(.2/.8)

def test_prior(map):
    """Test to make sure it returns empty space log-prob"""
    opm, x_t = map
    idx_x = 18

    l = opm.laser_range_inverse_sensor_model(idx_x, idx_y, x_t, z_t)

    assert l == np.log(.5/.5)