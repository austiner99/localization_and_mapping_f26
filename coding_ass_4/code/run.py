#!/usr/bin/env python3
"""Module for the main control in the landmark/localization lab: gathers data and executes loop

Defined Classes: 
ParticleFilterLocalizer - Implements particle filter localization.
"""
import matplotlib
matplotlib.use("Qt5Agg") # Maybe needed for MAC
import matplotlib.pyplot as plt
import numpy as np
import argparse

from generator import generate
from fieldSettings import field
import helpers
helpers.printDebug = False # Can change this to true if you need some debugs in your code, I did it on specific timesteps where I had issues
from helpers import debugPrint

from ekfUpdate import ekfUpdate
from ukfUpdate import ukfUpdate
from pfUpdate import pfUpdate

from tqdm import tqdm
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)}) # I found I wanted more decimals when debug printing floats, but you may not.

__author__ = "Joshua Mangelson, Joshua Broekhuijsen(translation from MATLAB), and Ryan Eustice(MATLAB)"
__copyright__ = "Copyright 2022 - Joshua Mangelson, Joshua Broekhuijsen, and Ryan Eustice"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"

def main(num, datafile, make_gif = False, watch_speed = 0.3, algorithm="all"):
    """Executes landmark/localization using the number of steps and/or data, and will potentially write the output to a gif.

    Parameters:
    num: The number of steps run - if the data is provided, this argument is optional
    datafile: the data, previously generated, on which to test the algorithm - if number of steps is provided, this argument is optional
    make_gif: Whether to write the output to a gif
    watchSpeed: time per frame while watching live.
    alg: Which algorithm to use - EKF, UKF, or PF, or all

    Returns:
    Potentially will output a gif
    """
    #################################################
    # Noise parameters
    #################################################
    # Motion noise (in odometry space, see Table 5.5, p.134 in book).
    alphas = np.array([0.05, 0.001, 0.05, 0.01])**2

    # Standard deviation of Gaussian sensor noise (independent of distance)
    beta = np.radians(20)
    
    #################################################
    # Graphics (You shouldn't need to modify this)
    #################################################

    noiseFreePathColor = '#00FF00'
    noisyPathColor = '#0000FF'

    noiseFreeBearingColor = 'c'
    observedBearingColor = '#FF0000'

    ekfColor = '#FF8000'
    ukfColor = '#8000FF'

    # Step size between filter updates, must be less than 1.
    deltaT = 0.1

    initialStateMean = np.array([180, 50, 0])

    ################################################
    # Data Reading or Generation (You Shouldn't have to modify this)
    ################################################
    
    data = None
    if (datafile is not None):
        data = np.load(datafile)['data']

    if (num is None):
        if (data is None):
            num = 100 # default value I guess?
        else:
            num = len(data)
    if (num is not None):
        if (data is not None):
            data = data[:num]
        else:
            landmarkStyle = "round-robin" # this can also be "random", "closest", or "farthest" - the last isn't very useful, but it was easy to implement
            landmarkPeriod = 2 # this only matters if landmarkStyle is "round-robin"
            data = generate(initialStateMean, num, alphas, beta, deltaT, landmark=landmarkStyle, period=landmarkPeriod)
    algorithm = algorithm.lower()
    if make_gif:
        from PIL import Image

    #######################################################
    # TODO: Initialize Your Filters Here
    #
    # You can set the initial mean and variance of the EKF
    # to the true mean and some uncertainty.
    #######################################################
    
    # Uncomment and finish these lines

    ekfMu = initialStateMean.astype(float)
    ekfCov = np.diag([1.0,1.0,0.01]) #CHECK: consider changing this if needed

    ukfMu = initialStateMean.astype(float)
    ukfCov = np.diag([1.0,1.0,0.01]) #CHECK: consider changing this if needed

    #######################################################
    # Some Plotting Code and Setup
    #######################################################
    plt.ion()
    gifFrames = []
    
    results = {'ekf':np.zeros((num, 6)), 'ukf':np.zeros((num, 6)), 'pf':[]}


    ###################################################
    # Call ekfUpdate, ukfUpdate and pfUpdate
    # in every iteration of this loop.    
    for t in tqdm(range(num)):

        # You can use lines like this to print debug info only
        # on specific time steps.
        # helpers.printDebug = t == 39 

        timeData = data[t]
        debugPrint("Got time data: ", timeData)
        #######################################################
        # data at this time step:
        # 0: landmark ID of observation
        # 1: bearing of noisy observation
        # 2: bearing of noise-free observation (for analysis, not available to algorithm)
        # 3: noise-free control d_rot_1
        # 4: noise-free control d_trans
        # 5: noise-free control d_rot_2
        # 6: noisy robot x (ground truth, not available to algorithm)
        # 7: noisy robot y (ground truth, not available to algorithm)
        # 8: noisy robot theta (ground truth, not available to algorithm)
        # 9: noise-free robot x (for analysis, not available to algorithm)
        # 10: noise-free robot y (for analysis, not available to algorithm)
        # 11: noise-free robot theta (for analysis, not available to algorithm)
        #######################################################

        #######################################################
        # data available to your filter at this time step
        # Index 0, 1, 3, 4, 5
        #######################################################
        motionCommand = timeData[3:6] # [drot1, dtrans, drot2] noisefree control command
        observation = timeData[0:2] # [bearing, landmark_id] noisy observation
        debugPrint("Motion command: ", motionCommand)
        debugPrint("Observation: ", observation)

        #######################################################
        # data *not* available to your filter, i.e., known
        # only by the simulator, useful for making error plots
        # Index 2, 6, 7, 8, 9, 10, 11
        #######################################################
        # actual position (i.e., ground truth)
        x = timeData[6]
        y = timeData[7]
        theta = timeData[8]
        debugPrint("Ground truth: ", x, y, theta)

        # noise-free observation
        noiseFreeBearing = timeData[2]
        debugPrint("Noise-free bearing: ", noiseFreeBearing)

        #################################################
        # Graphics (You should not have to modify this)
        #################################################
        plt.clf() # clear the frame.
        helpers.plotField(observation[0]) # Plot the field with the observed landmark highlighted

        # draw actual path and path that would result if there was no noise in
        # executing the motion command
        plt.plot(np.array([initialStateMean[0], *data[:t, 9]]), np.array([initialStateMean[1], *data[:t, 10]]), color=noiseFreePathColor)
        plt.plot(timeData[9], timeData[10], '*', color=noiseFreePathColor, label="No Noise")

        # draw the path that has resulted from the movement with noise
        plt.plot(np.array([initialStateMean[0], *data[:t, 6]]), np.array([initialStateMean[1], *data[:t, 7]]), color=noisyPathColor, label="True (Noisy)")
        helpers.plotRobot(timeData[6:9], "black", "#00FFFF")

        # indicate observed angle relative to actual position
        plt.plot(np.array([x, x+np.cos(theta + observation[1])*100]), np.array([y, y+np.sin(theta + observation[1])*100]), color=observedBearingColor)

        # indicate ideal noise-free angle relative to actual position
        plt.plot(np.array([x, x+np.cos(theta + timeData[2])*100]), np.array([y, y+np.sin(theta + timeData[2])*100]), color=noiseFreeBearingColor)

        ##################################################################
        # TODO: Call ekfUpdate, ukfUpdate, pfUpdate For Current Time Step
        ##################################################################

        # Calculate Control Noise Here (Uncomment and Complete)
        # M = fill this in 
        # debugPrint("M: ", M)
        th1, d, th2 = motionCommand
        M = np.diag([alphas[0]*th1**2 + alphas[1]*d**2,
                     alphas[2]*d**2 + alphas[3]*(th1**2+th2**2),
                     alphas[0]*th2**2 + alphas[1]*d**2])

        # Calculate Measurement Noise (Uncomment)
        # Q = np.array([[beta**2]])
        # debugPrint("Q: ", Q)
        Q = np.array([[beta**2]])

        # Setup Observation/Measurements/IDs?
        z = np.array([[observation[1]]])
        debugPrint("z: ", z)
        markerId = int(observation[0])
        debugPrint("MarkerID: ", markerId)
        
        # Sample Debug Output Modify/add as needed 
        #debugPrint("ukfMu: ", ukfMu)
        #debugPrint("ukfCov: ", ukfCov)
        
        
        # You'll need to pass the relevant arguments here -
        # commented out until then.            
        if (algorithm in ['all', 'ekf']):
            ekfMu, ekfCov = ekfUpdate(ekfMu, ekfCov, motionCommand, M, z, Q, markerId)
            results['ekf'][t] = np.concatenate([ekfMu, np.sqrt(np.diag(ekfCov))])

        if (algorithm in ['all', 'ukf']):
            ukfMu, ukfCov = ukfUpdate(ukfMu, ukfCov, motionCommand, M, z, Q, markerId)
            results['ukf'][t] = np.concatenate([ukfMu, np.sqrt(np.diag(ukfCov))])

        if (algorithm in ['all', 'pf']):
            # pfUpdate()
            # results['pf'][t] = 
            pass

        if (algorithm not in ['all', 'ekf', 'ukf', 'pf']):
            raise Exception('Invalid argument for algorithm.  Must be "EKF", "UKF", "PF", or "all", not "' + algorithm + '"')


        #################################################
        # TODO: Plot Filter Results Live
        #
        # Use helpers.plotCov2D and plt.plot to plot the
        # results live at each timestep for each filter
        #
        # Make sure you add a label (for the trajectory only)
        # so that the Legend includes an entry for your data
        #
        # Make sure to use the colors defined above
        ##############################################
        
        if (algorithm in ['all', 'ekf']):
            
            plt.plot(results['ekf'][:t+1,0], results['ekf'][:t+1,1], color=ekfColor, label="EKF")
            helpers.plotCov2D(ekfMu[:2], ekfCov[:2,:2], color=ekfColor, nSigma=3)
            # plt.gca().lines[-1].set_linewidth(3)

        if (algorithm in ['all', 'ukf']):
            plt.plot(results['ukf'][:t+1,0], results['ukf'][:t+1,1], color=ukfColor, label="UKF")
            helpers.plotCov2D(ukfMu[:2], ukfCov[:2,:2], color=ukfColor, nSigma=3)
            # plt.gca().lines[-1].set_linestyle('--')

        if (algorithm in ['all', 'pf']):
            pass # this allows the empty block

        if (algorithm not in ['all', 'ekf', 'ukf', 'pf']):
            raise Exception('Invalid argument for algorithm.  Must be "EKF", "UKF", "PF", or "all", not "' + algorithm + '"')

        #################################################
        # Some More Plotting Code (Don't Modify)
        ################################################

        plt.legend()
        plt.gcf().canvas.draw()
        if (make_gif):
            # Capture the current figure as an image
            imgData = np.frombuffer(plt.gcf().canvas.buffer_rgba(), dtype=np.uint8)
            w, h = plt.gcf().canvas.get_width_height()
            mod = np.sqrt(imgData.shape[0]/(4*w*h))            # 4 channels now (RGBA), not 3
            im = imgData.reshape((int(h*mod), int(w*mod), -1))
            gifFrames.append(Image.fromarray(im).convert('RGB'))
        time.sleep(watch_speed)
        plt.gcf().canvas.flush_events()

    ##################################################
    # Plotting and Video Code (Don't Modify)
    ##################################################
    plt.ioff()
    plt.show(block=False)

    if (make_gif):
        # Save into a GIF file that loops forever
        gifFrames[0].save('gifOutput.gif', format='GIF',
            append_images=gifFrames[1:],
            save_all=True,
            duration=num*2*deltaT, loop=1)

    plt.show(block=False)


    #########################################################
    # TODO: Generate Result Plots After Simulation Finishes
    #########################################################

    timeIndex = np.arange(num)
    ekfError = results['ekf'][:, 0:3] - data[:, 6:9]
    ekfThreeSigma = 3 * results['ekf'][:, 3:6]
    #minimizedAngle error for theta
    ekfError[:, 2] = helpers.minimizedAngle(ekfError[:, 2])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    labels = [('X error', 'cm'), ('Y error', 'cm'), ('Theta error', 'rad')]
    for i, (label, unit) in enumerate(labels):
        axes[i].plot(timeIndex, ekfError[:, i], label='Error')
        axes[i].plot(timeIndex, ekfThreeSigma[:, i], '--', color='red', label='+3 sigma')
        axes[i].plot(timeIndex, -ekfThreeSigma[:, i], '--', color='red', label='-3 sigma')
        axes[i].set_ylabel(f'{label} ({unit})')
        axes[i].grid(True)
        axes[i].legend(loc='upper left')
    axes[-1].set_xlabel('Timestep index')
    fig.suptitle('EKF State Error with 3-Sigma Bounds')
    fig.tight_layout()

    plt.show(block=False)
    
    #plotting for ukf error stuff
    ukfError = results['ukf'][:, 0:3] - data[:, 6:9]
    ukfThreeSigma = 3 * results['ukf'][:, 3:6]
    #minimizedAngle error for theta
    ukfError[:, 2] = helpers.minimizedAngle(ukfError[:, 2])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    labels = [('X error', 'cm'), ('Y error', 'cm'), ('Theta error', 'rad')]
    for i, (label, unit) in enumerate(labels):
        axes[i].plot(timeIndex, ukfError[:, i], label='Error', color='blue')
        axes[i].plot(timeIndex, ukfThreeSigma[:, i], '--', color='red', label='+3 sigma')
        axes[i].plot(timeIndex, -ukfThreeSigma[:, i], '--', color='red', label='-3 sigma')
        axes[i].set_ylabel(f'{label} ({unit})')
        axes[i].grid(True)
        axes[i].legend(loc='upper left')
    axes[-1].set_xlabel('Timestep index')
    fig.suptitle('UKF State Error with 3-Sigma Bounds')
    fig.tight_layout()

    plt.show()

    #compute and print the fraction of steps where abs(error) <= 3 sigma for each x, y and theta
    x_fraction = np.mean(np.abs(ekfError[:, 0]) <= ekfThreeSigma[:, 0])
    y_fraction = np.mean(np.abs(ekfError[:, 1]) <= ekfThreeSigma[:, 1])
    theta_fraction = np.mean(np.abs(ekfError[:, 2]) <= ekfThreeSigma[:, 2])

    print(f"Fraction of steps with |X error| <= 3 sigma: {x_fraction:.2%}")
    print(f"Fraction of steps with |Y error| <= 3 sigma: {y_fraction:.2%}")
    print(f"Fraction of steps with |Theta error| <= 3 sigma: {theta_fraction:.2%}")
    
    x_fraction_ukf = np.mean(np.abs(ukfError[:, 0]) <= ukfThreeSigma[:, 0])
    y_fraction_ukf = np.mean(np.abs(ukfError[:, 1]) <= ukfThreeSigma[:, 1])
    theta_fraction_ukf = np.mean(np.abs(ukfError[:, 2]) <= ukfThreeSigma[:, 2])

    print(f"Fraction of steps with |X error| <= 3 sigma (UKF): {x_fraction_ukf:.2%}")
    print(f"Fraction of steps with |Y error| <= 3 sigma (UKF): {y_fraction_ukf:.2%}")
    print(f"Fraction of steps with |Theta error| <= 3 sigma (UKF): {theta_fraction_ukf:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmark/Localization Lab")
    parser.add_argument("-g", "--make_gif", action="store_true", help="Whether to save a gif of all the frames")
    # NOTE: Setting the data file as a default left us without a way in the command line to generate new data, so... I made it not default.
    parser.add_argument("-d", "--datafile", type=str, default=None, help="Location of landmark/localization data. Defaults to none.")
    parser.add_argument("-a", "--algorithm", type=str, default="all", help="Which algorithm to use.  'EKF', 'UKF', 'PF', or 'all'")
    parser.add_argument("-w", "--watch_speed", type=float, default=0.3, help="Time to pause on each frame when viewing live.")
    parser.add_argument("-n", "--num", type=int, default=None, help="Number of steps to take")
    args = vars(parser.parse_args())

    main(**args)
