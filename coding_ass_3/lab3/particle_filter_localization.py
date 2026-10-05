#!/usr/bin/env python3
"""Module defining an implementation of particle filter localization.

Defined Classes: 
ParticleFilterLocalizer - Implements particle filter localization.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import argparse

from numba import njit
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from lab3.occupancy_grid_map import OccupancyGrid 

__author__ = "Joshua Mangelson"
__copyright__ = "Copyright 2020, Joshua Mangelson, Brigham Young University"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"

# Helper functions
def convert_xyt_to_H(dx:float, dy:float, dtheta:float) -> 'np.ndarray[(3,3) , np.dtype[np.float64]]':
    """Convert X-Y-Theta to a transformation matrix.

    Parameters: 
    dx: Change in x. 
    dy: Change in y.
    dtheta: Change in orientation (In degrees).
    
    Return: 
    H: A 2D homogeous transformation matrix encoding the input parameters.
    """
    dtheta = math.radians(dtheta)

    H = np.array(
        [ [ math.cos(dtheta), -math.sin(dtheta), dx],
          [ math.sin(dtheta), math.cos(dtheta), dy],
          [ 0, 0, 1] ] )
    return H

@njit
def get_cell_odds(grid: 'np.ndarray[(num_grid_squares_x, num_grid_squares_y) , np.dtype[np.float64]]', x: float, y: float,
                  xmin: float, ymin: float, res: float) -> float:
    """Get cell odds of a specific x,y coordinate. Note that this
    function DOES NOT check for out of bounds x, y coordinates.
    
    Parameters:
    grid: 2D numpy array of grid probabilities, found in OccupancyGrid.grid
    x: x coordinate (not an index)
    y: y coordinate (not an index)
    xmin: Minimum x-coordinate of the grid. Found in OccupancyGrid.xlim[0]
    ymin: Minimum y-coordinate of the grid. Found in OccupancyGrid.ylim[0]
    res: Resolution of the grid. Found in OccupancyGrid.resolution

    Returns:
    Log-prob of that x-y coordinate being occupied.
    
    """
    x_idx = int((x - xmin) / res)
    y_idx = int((y - ymin) / res)
    
    return grid[y_idx, x_idx]

    
class ParticleFilterLocalizer():
    """Class that utilizes a particle filter to localize in a prior ogm."""

    def __init__(self, priormap: OccupancyGrid, xlim: float, ylim: float, N: int,
                 alphas: 'np.ndarray[(4,) , np.dtype[np.float64]]', cov: 'np.ndarray[(3,) , np.dtype[np.float64]]'):
        """Create and initialize the particle filter.

        Parameters:
        priormap: An OccupancyGrid object representing the underlying map.
        xlim: A two element list defining the x bounds to be searched.
        ylim: A two element list defining the y bounds to be searched.
        N: The number of particles to generate
        alphas: A 4-array of values make up the laser sensor model. In the order of [p_hit, p_unexp, p_random, p_max]. Should sum to 1.
        cov: A 3-array of cov for the motion model. In the order of [x, y, theta]
        """
        self.priormap = priormap
        self.N = N
        self.weights = np.ones(N) / N
        
        #set up particles
        self.particles = np.random.uniform([xlim[0], ylim[0], 0],
                                        [xlim[1], ylim[1], 360], (N,3))
        # We sample one particle exactly to jumpstart the algorithm
        self.particles[0] = np.array([1.5, 1.5, 0])

        # Set up noise for motion/measurement models
        # You may do whatever calculations you want, but 
        # make sure you use what's passed into __init__()
        self.cov = cov
        self.alphas = alphas
        self.z_max = 20


    def propagate_motion(self, u: 'np.ndarray[(3,) , np.dtype[np.float64]]'):
        """Propagate motion noisily through all particles in place.

        Make sure you use self.cov somewhere for the noise added.

        Parameters:
        u: A 3-array representing movement from the previous step 
            as a delta x, delta y, delta theta (in degrees).
        """
        ####################################
        # Finish this implemenation!!

        ####################################

    @staticmethod
    @njit
    def expected_measurement(angles: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]',
                             pos: 'np.ndarray[(3,) , np.dtype[np.float64]]',
                             grid: 'np.ndarray[(num_grid_squares_x,num_grid_squares_y) , np.dtype[np.float64]]',
                             xmin: float, xmax: float, ymin: float, ymax: float,
                             res: float) -> 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]':
        """Get the expected distances of a laser range finder based on it's
            laser angles and robot position

        NOTE: To use @njit (please do it'll make it tons faster), use 
            get_cell_odds to get the odds of an x,y coordinate. Note get_cell_odds
            does NOT check for out of bounds, you'll have to do that if you use it.
            You'll also have to only use other functions that are also @njit, or are
            in the numpy library.

        If you don't want to use @njit, remove @njit from a couple of lines up.

        Note there is a few ways of doing this, the tests are flexible and accept a given range.
        Also, search out to 25 meters, 5 meters beyond the max range of sensor.

        Parameters:
        angles: Angles the LiDAR was sampled at, with respect to the local x-axis of 
            the robot (straight ahead) with positive angles laying to the left 
            of the x-axis following the right hand rule (counter clock-wise in
            the xy plane). (In Degrees).      
        pos: The position of the particle to sample the distribution of.
        grid: The 2D numpy array as stored in (get from self.priormap.grid)
        xmin: Minimum of the map in the x-direction
        xmax: Maximum of the map in the x-direction
        ymin: Minimum of the map in the y-direction
        ymax: Maximum of the map in the y-direction
        res: Resolution of the map.

        Returns:
        Numpy array of expected distances
        """
        ####################################
        # Finish this implemenation!!


        ####################################

    @staticmethod
    @njit
    def update_weight(z_k: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]',
                      z_t: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]',
                      alphas: 'np.ndarray[(4,) , np.dtype[np.float64]]'):
        """Update a single particle's probability according to expected distances.

        NOTE: To use @njit (please do it'll make it tons faster), use 
            get_cell_odds to get the odds of an x,y coordinate. Note get_cell_odds
            does NOT check for out of bounds, you'll have to do that if you use it.
            You'll also have to only use other functions that are also @njit, or are
            in the numpy library.

        If you don't want to use @njit, remove @njit from a couple of lines up.

        Parameters:
        z_k: An array of expected measurements given robot position
        z_t: An array of range measurements from the LiDAR
        alphas: The various weights of the different probability distributions in the order
            p_hit, p_random, p_max (get from self.alphas)
        """
        ####################################
        # Finish this implemenation!!

        ####################################

    def normalize_weights(self):
        """Normalize self.weights in place."""
        ####################################
        # Finish this implemenation!!

        ####################################

    def resample_particles(self):
        """Resample particles in place according to the probabilities in self.weights"""
        ####################################
        # Finish this implemenation!!

        ####################################

    def iterate(self, u: 'np.ndarray[(3,) , np.dtype[np.float64]]',
                z_t: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]',
                angles: 'np.ndarray[(num_measurements,) , np.dtype[np.float64]]'):
        """Propagate motion according to control, and correct using laser measurements.
        
        Parameters:
        u: A 3-array representing movement from the previous step 
            as a delta x, delta y, delta theta. 
        z_t: An array of range measurements from the LiDAR
        angles: Angles the LiDAR was sampled at, with respect to the local x-axis of 
            the robot (straight ahead) with positive angles laying to the left 
            of the x-axis following the right hand rule (counter clock-wise in
            the xy plane). (In Degrees).
        """
        ####################################
        # Finish this implemenation!!

        ####################################


def main(plot_live: bool, mapfile: str, datafile: str, num: int):
    #################################################
    # Tweak these like you want
    alphas = np.array([0.8, 0.0, 0.05, 0.15])
    cov = np.array([.04, .04, .01])
    #################################################

    # Load prior map
    prior_map = pickle.load(open(mapfile, "rb"))

    # Load data stream
    data = np.load(datafile)
    X_t = data['X_t']
    U_t = data['U_t']
    Z_tp1 = data['Z_tp1']
    angles = data['angles']
    
    # Initialize particle filter
    pf = ParticleFilterLocalizer(prior_map, [-20, 40], [-20, 20], num, alphas, cov)    

    # Setup plotting
    if plot_live:
        plt.ion()
        fig, ax = plt.subplots()    

        prior_map.plot_grid(ax)
        true_loc = ax.scatter(X_t[0][0], X_t[0][1], 2, 'b')
        particles = ax.scatter(pf.particles[:,0], pf.particles[:,1], 0.5, 'r')          

    # Loop through data stream
    for t in tqdm(range(len(X_t)-1)):
        # Extract data
        u_t = U_t[t]
        z_tp1 = Z_tp1[t]

        #######################################################
        # Operate on Data to run the particle filter algorithm

        #######################################################

        # Plot
        if plot_live and t % 5 == 0:
            true_loc.set_offsets(X_t[t+1:t+2,:2])
            particles.set_offsets(pf.particles[:,:2])

            fig.canvas.draw()
            fig.canvas.flush_events()
    
    if plot_live:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupancy Grid Map")
    parser.add_argument("-p", "--plot_live", action="store_true", help="Whether we should plot as we go")
    parser.add_argument("-d", "--datafile", type=str, default="coding_ass_3/data/localization-dataset.npz", help="Location of localization data. Defaults to data/localization-data.npz")
    parser.add_argument("-m", "--mapfile", type=str, default="coding_ass_3/data/map.p", help="Location of map data. Defaults to data/map.p")
    parser.add_argument("-n", "--num", type=int, default=100, help="Number of particles to use")
    args = vars(parser.parse_args())

    main(**args)
