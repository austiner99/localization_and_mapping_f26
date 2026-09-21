from src.occupancy_grid_map import OccupancyGridMap, RobotState
import pytest

"""Used to set up a simple OGM and a sample robot state"""
@pytest.fixture
def map():
    opm = OccupancyGridMap(1, [-10,10], [-10,10], 0.1, 0.2, 0.5)
    x_t = RobotState(0.5,0.5,90)
    return opm, x_t