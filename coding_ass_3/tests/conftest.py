from lab3.occupancy_grid_map import OccupancyGrid
import numpy as np
import os
import csv
import pytest

@pytest.fixture(scope="session")
def omap():
    file = "data/ogm_ground_truth.csv"
    if os.getcwd()[-5:] == "tests":
        file = "../" + file

    omap = OccupancyGrid(resolution=0.5,
                            xlim=[-20,50],
                            ylim=[-30,30],
                            prior=0.5)

    reader = csv.reader(open(file), delimiter=",")
    x = list(reader)
    omap.grid = np.array(x).astype("float")

    omap.grid[omap.grid == 1] = 100
    omap.grid[omap.grid == 0] = -100

    return omap