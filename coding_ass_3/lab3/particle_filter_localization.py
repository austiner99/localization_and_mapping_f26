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

def mean_pose(particles):
    """Compute the mean pose of a set of particles.

    Parameters:
    particles: A numpy array of shape (N, 3) where each row is [x, y, theta].

    Returns:
    mean_pose: A numpy array of shape (3,) representing the mean [x, y, theta].
    """
    th = np.radians(particles[:, 2])
    mean_theta = np.degrees(np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th))))
    return np.array([np.mean(particles[:, 0]), np.mean(particles[:, 1]), mean_theta])

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
        # self.particles[0] = np.array([1.5, 1.5, 0])

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
        theta = np.radians(self.particles[:, 2])
        c, s = np.cos(theta), np.sin(theta)
        new_x = self.particles[:, 0] + u[0] * c - u[1] * s
        new_y = self.particles[:, 1] + u[0] * s + u[1] * c
        new_theta = (self.particles[:, 2] + u[2])

        noise = np.random.normal(0, np.sqrt(self.cov), self.particles.shape)
        
        self.particles[:, 0] = new_x + noise[:, 0]
        self.particles[:, 1] = new_y + noise[:, 1]
        self.particles[:, 2] = (new_theta + noise[:, 2]) % 360

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

        max_range = 25.0
        distances = np.empty(len(angles), dtype=np.float64)

        for i in range(len(angles)):
            angle = np.radians(angles[i] + pos[2])
            dx = np.cos(angle)
            dy = np.sin(angle)
            distance = 0.0
            step = res/2.0

            # get_cell_odds converts world coordinates to [y, x] map indices.
            # Bounds are checked here because get_cell_odds assumes valid input.
            while distance <= max_range:
                x = pos[0] + distance * dx
                y = pos[1] + distance * dy
                if x < xmin or x >= xmax or y < ymin or y >= ymax:
                    distance = max_range
                    break
                if get_cell_odds(grid, x, y, xmin, ymin, res) > 0.0:
                    break
                distance += step

            distances[i] = min(distance, max_range)

        return distances

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

        weight = 1.0
        for i in range(len(z_k)):
            diff = z_k[i] - z_t[i]
            p_hit = np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)
            p_unexp = 0.1 * np.exp(-0.1 * z_t[i]) if z_t[i] >= 0.0 else 0.0
            p_random = 1.0 / 20.0
            p_max = 1.0 if z_t[i] >= 20.0 else 0.0
            probability = (alphas[0] * p_hit + alphas[1] * p_unexp +
                           alphas[2] * p_random + alphas[3] * p_max)
            weight *= max(probability, 1e-300)
        return weight

    def normalize_weights(self):
        """Normalize self.weights in place."""
        ####################################
        # Finish this implemenation!!
        
        self.weights /= np.sum(self.weights)

        ####################################

    def resample_particles(self):
        """Resample particles in place according to the probabilities in self.weights"""
        ####################################
        # Finish this implemenation!!
        new_particles = np.empty((self.N, 3), dtype=np.float64)
        r = np.random.uniform(0, 1/self.N)
        c = self.weights[0]
        i = 0
        for m in range(self.N):
            pointer = r+m/self.N
            while pointer > c:
                i += 1
                c += self.weights[i]
            new_particles[m] = self.particles[i]
        self.particles = new_particles
        self.weights = np.ones(self.N) / self.N
        

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
        self.propagate_motion(u)
        
        grid = self.priormap.grid
        xmin, xmax = self.priormap.xlim
        ymin, ymax = self.priormap.ylim
        res = self.priormap.resolution
        for i in range(self.N):
            z_k = self.expected_measurement(angles, self.particles[i], grid, xmin, xmax, ymin, ymax, res)
            self.weights[i] = self.update_weight(z_k, z_t, self.alphas)
            
        self.normalize_weights()
        self.resample_particles()

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
    
    # print("Loaded data: X_t: \n", X_t, "\n U_t:", U_t, "\n Z_tp1: \n", Z_tp1, "\n angles: \n", angles)
    
    # Initialize particle filter
    pf = ParticleFilterLocalizer(prior_map, [-20, 40], [-20, 20], num, alphas, cov)    

    # Setup plotting
    if plot_live:
        plt.ion()
        fig, ax = plt.subplots()    

        prior_map.plot_grid(ax)
        true_loc = ax.scatter(X_t[0][0], X_t[0][1], 2, 'b')
        particles = ax.scatter(pf.particles[:,0], pf.particles[:,1], 0.5, 'r')          

    pos_err = np.zeros(len(X_t)-1)
    heading_err = np.zeros(len(X_t)-1)
    # Loop through data stream
    for t in tqdm(range(len(X_t)-1)):
        # Extract data
        u_t = U_t[t]
        z_tp1 = Z_tp1[t]

        #######################################################
        # Operate on Data to run the particle filter algorithm
        
        pf.iterate(u_t, z_tp1, angles)
        
        #for plotting mean pose difference:
        avg_pose = mean_pose(pf.particles)
        pos_err[t] = np.hypot(avg_pose[0] - X_t[t+1,0], avg_pose[1] - X_t[t+1,1])
        heading_err[t] = np.abs(np.radians(avg_pose[2]) - np.radians(X_t[t+1,2]))
        
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
        
    print(f"mean position error: {np.mean(pos_err)}")
    print(f"mean heading error: {np.mean(heading_err)}")
    print(f"max position error: {np.max(pos_err)}")
    print(f"max heading error: {np.max(heading_err)}")
    
    # Plot position and heading errors over time
    fig2, (ax_pos, ax_heading) = plt.subplots(2, 1, sharex=True, figsize = (9,6))
    ax_pos.plot(pos_err, label="Position Error")
    ax_pos.set_title(f"Position Error Over Time(N={num})")
    ax_pos.set_ylabel("Position Error (m)")
    ax_pos.legend()
    ax_heading.plot(heading_err, label="Heading Error", color="orange")
    ax_heading.set_xlabel("Time Step")
    ax_heading.set_ylabel("Heading Error (rad)")
    ax_heading.legend()
    fig2.tight_layout()
    fig2.savefig("position_heading_errors.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Occupancy Grid Map")
    parser.add_argument("-p", "--plot_live", action="store_true", help="Whether we should plot as we go")
    parser.add_argument("-d", "--datafile", type=str, default="data/localization-dataset.npz", help="Location of localization data. Defaults to data/localization-data.npz")
    parser.add_argument("-m", "--mapfile", type=str, default="data/map.p", help="Location of map data. Defaults to data/map.p")
    parser.add_argument("-n", "--num", type=int, default=100, help="Number of particles to use")
    args = vars(parser.parse_args())

    main(**args)
