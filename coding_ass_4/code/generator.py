#!/usr/bin/env python3
"""Module defining the data generation for testing landmark/localization: this can make a new noisy data set for any number of steps.

Defined functions: 
generate - Calculates trajectory data for use in testing
"""
import numpy as np
import fieldSettings
import helpers

__author__ = "Joshua Mangelson, Joshua Broekhuijsen(translation from MATLAB), and Ryan Eustice(MATLAB)"
__copyright__ = "Copyright 2022 - Joshua Mangelson, Joshua Broekhuijsen, and Ryan Eustice"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"

# Helper functions
def prediction(state, motion):
    """Propagates motion through the state

    Parameters:
    state: The current state (noisy or not) of the robot [x, y, theta]
    motion: Motion for this step [d_rot_1, d_trans, d_rot_2]

    Returns:
    Prediction of the state given the motion
    """
    angle = state[2] + motion[0]
    x = state[0] + motion[1]*np.cos(angle)
    y = state[1] + motion[1]*np.sin(angle)
    angle += motion[2]
    angle = helpers.minimizedAngle(angle)

    return np.array([x, y, angle])

def sampleOdometry(motion, state, alphas):
    """Calculates odometry data with noise determined by alphas, returns prediction based on state.

    Parameters:
    motion: Noise-free motion for this step [d_rot_1, d_trans, d_rot_2]
    state: The current state (noisy or not) of the robot [x, y, theta]
    alphas: Variance coefficients, see probabilistic robotics chapter 5, eq 5.37-3.39, [a0, a1, a2, a3]

    Returns:
    Prediction of the state given the motion corrupted by the noise determined by the alphas
    """
    drot1 = motion[0]
    dtran = motion[1]
    drot2 = motion[2]

    noisyMotion = np.zeros(3)
    sqrMag = np.abs(motion)**2
    noisyMotion[0] = np.random.normal(motion[0], np.sqrt(alphas[0]*sqrMag[0] + alphas[1]*sqrMag[1]))
    noisyMotion[1] = np.random.normal(motion[1], np.sqrt(alphas[2]*sqrMag[1] + alphas[3]*(sqrMag[0]+sqrMag[2])))
    noisyMotion[2] = np.random.normal(motion[2], np.sqrt(alphas[0]*sqrMag[2] + alphas[1]*sqrMag[1]))

    return prediction(state, noisyMotion)

def genOneMotion(index, movement, step):
    if (index == 2*step):
        return np.array([np.radians(45), movement, np.radians(45)])
    elif (index == 4*step):
        return np.array([np.radians(45), 0, np.radians(45)])
    else:
        return np.array([0, movement, 0])

def generateMotion(t):
    """Generates a path the robot walks, returning each step for time t

    Parameters: 
    t: array of timesteps for which the path should be generated
    deltaT: Gap between timesteps.  This COULD be found, frankly.
    
    Return: 
    M: A len(t)-by-3 array where each row is [rotation, translation, rotation] of movement
    """

    deltaT = np.mean(np.diff(t))

    if (deltaT > 1.):
        raise Exception('deltaT should be < 1.0')

    step = np.floor(1/deltaT)
    movement = 100*deltaT
    index = np.arange(len(t))%(5*step)

    out = np.zeros((len(t), 3))
    for i in range(len(t)):
        out[i] = genOneMotion(index[i], movement, step)
    # return genMotionV(index, movement, step)
    return out


def observation(robot, marker):
    """Calculates the observation for a robot to a given marker

    Parameters: 
    robot: this should be the noise free [x, y, theta]
    marker: the integer index of the marker in the field
    
    Return: 
    theta: the measured angle betwen the bearing of the robot and the marker relative to the robot
    """
    field = fieldSettings.field
    # the observation is just a bearing relative to the robot
    return helpers.minimizedAngle(np.arctan2(field['markerPosY'][marker] - robot[1], field['markerPosX'][marker] - robot[0]) - robot[2])


def generate(initialStateMean, numSteps, alphas, beta, deltaT, landmark="round-robin", period=2):
    """Calculates trajectory data for use in testing

    Parameters: 
    initialStateMean: Initial mean pose of the robot, [x, y, theta]
    numSteps: Number of steps to generate
    alphas: 4D noise for robot motion
    beta: Noise for observations
    deltaT: Time between each step
    landmark: What method to use for landmark selection.  Default "round-robin" with period of 2, but can also be "random", "closest", or "farthest"
    period: period of the landmark when method is "round-robin"
    
    Return: 
    M: A numSteps-by-12 array, each row has the format:
     0: landmark ID of observation
     1: bearing of noisy observation
     2: bearing of noise-free observation (for analysis, not available to algorithm)
     3: noise-free control d_rot_1
     4: noise-free control d_trans
     5: noise-free control d_rot_2
     6: noisy robot x
     7: noisy robot y
     8: noisy roboy theta
     9: noise-free robot x (for analysis, not available to algorithm)
    10: noise-free robot y (for analysis, not available to algorithm)
    11: noise-free robot theta (for analysis, not available to algorithm)
    """

    observationDim = 2
    realRobot = initialStateMean.copy()
    noiseFreeRobot = initialStateMean.copy()

    field = fieldSettings.field

    data = np.zeros((numSteps, 12))
    t = deltaT*np.array(range(numSteps))
    noiseFreeMotion = generateMotion(t)
    for n in range(numSteps):
        # Noise-free robot update
        noiseFreeRobot = sampleOdometry(noiseFreeMotion[n], noiseFreeRobot, [0, 0, 0, 0])

        # Noisy robot update
        realRobot = sampleOdometry(noiseFreeMotion[n], realRobot, alphas)

        if (landmark == "round-robin"):
            markerId = int(np.floor(n/period) % field['numMarkers'])
        elif (landmark == "random"):
            markerId = int(np.floor(np.random.uniform() * field['numMarkers']))
        elif (landmark == "closest" or landmark == "farthest"):
            # find dist to landmarks
            dist = np.sqrt((field['markerPosX'].copy() - noiseFreeRobot[0])**2 + (field['markerPosY'].copy() - noiseFreeRobot[1])**2)
            funcToUse = min if landmark == "closest" else max
            markerId = np.where(dist == funcToUse(dist))[0][0]
        else:
            raise Exception('Landmark selection method "' + landmark + '" is unknown.')

        noiseFreeObservation = observation(realRobot, markerId)
        noisyObservation = noiseFreeObservation + np.random.normal(0, beta)

        data[n, 0] = markerId
        data[n, 1] = noisyObservation
        data[n, 2] = noiseFreeObservation
        data[n, 3:6] = noiseFreeMotion[n]
        data[n, 6:9] = realRobot.copy()
        data[n, 9:] = noiseFreeRobot.copy()

    return data
