#!/usr/bin/env python3
"""Module defining the data generation for testing SLAM: this can make a new noisy data set for any number of steps.

Defined functions: 
generate - Calculates trajectory data for use in testing
"""
import numpy as np
import fieldSettings
import helpers
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

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
    [id, dist, theta]
    id: which marker
    dist: how far from the robot to the marker
    theta: the measured angle betwen the bearing of the robot and the marker relative to the robot
    """
    field = fieldSettings.field
    # the observation is just a bearing relative to the robot
    dx = field['markerPosX'][marker] - robot[0]
    dy = field['markerPosY'][marker] - robot[1]
    dist = np.sqrt(dx**2 + dy**2)
    theta = helpers.minimizedAngle(np.arctan2(dy, dx) - robot[2])
    return np.array([marker, dist, theta])


def markersInFov(robot, maxObs, fov):
    """Will find what markers are in the field of view of the robot,
    sort by magnitude of bearing, and return up to maxObs marker IDs

    Parameters:
    robot: the true position of the robot [x, y, theta]
    maxObs: the maximum number of marker IDs to return
    fov: the field of view (in radians) of the robot, centered on the robot's theta bearing
    
    Return:
    [markerID_0, markerID_1, markerID_2...]
    Will return as many marker IDs (sorted by minimum bearing) as it can - up to maxObs,
    so long as they are within the FOV
    """
    field = fieldSettings.field
    allObs = np.zeros((field['numMarkers'], 3))
    for i in range(len(allObs)):
        allObs[i] = observation(robot, i) # Generate the observation for each marker
    inFov = allObs[np.argwhere(np.abs(allObs[:,2]) <= fov/2)].reshape((-1, 3)) # Find only the elements within the FOV
    distSort = inFov[np.argsort(np.abs(inFov[:,1]))].reshape((-1, 3)) # Sort by distance for conflict resolution
    sortedData = distSort[np.argsort(np.abs(distSort[:,2]))].reshape((-1, 3)) # Then sort by theta, which is our primary sort
    return sortedData[:min(len(sortedData), maxObs), 0].astype(int) # Return at MOST maxObs markerIDs


def generate(initialStateMean, numSteps, alphas, beta, deltaT, maxObs=2, landmark="fov", landmarkControl=np.pi, forceNew = False): #period=2):
    """Calculates trajectory data for use in testing

    Parameters: 
    initialStateMean: Initial mean pose of the robot, [x, y, theta]
    numSteps: Number of steps to generate
    alphas: 4D noise for robot motion
    beta: Noise for observations
    deltaT: Time between each step
    maxObs: The maximum number of landmarks that can be returned at a single timestep
    landmark: What method to use for landmark selection.  Default "fov" with a range of pi (centered), but can also be "round-robin" (must set landmarkControl), "random", "closest", or "farthest"
    landmarkControl: period of the landmark when method is "round-robin" (so 2 will dwell on each landmark for 2 cycles), range of FOV when landmark metho is "fov"
    
    Return: 
    M: A numSteps-by-(9+5*maxObs) array, each row has the format:
     0: noise-free control d_rot_1
     1: noise-free control d_trans
     2: noise-free control d_rot_2
     3: noisy robot x
     4: noisy robot y
     5: noisy robot theta
     6: noise-free robot x (for analysis, not available to algorithm)
     7: noise-free robot y (for analysis, not available to algorithm)
     8: noise-free robot theta (for analysis, not available to algorithm)
     9: marker ID of observed landmark (available for known data association, or if -1 there is no observation)
    10: dist of noisy observation
    11: theta of noisy observation
    12: dist of noise-free observation (for analysis, not available to algorithm)
    13: theta of noise-free observation (for analysis, not available to algorithm)
    14-18: repeat 9-13 through maxObs: if a timestep does not have an observation, markerID will be -1
    19-23: repeat... etc.
    """

    if (not forceNew):
        try:
            data = np.load("savedData/" + str(numSteps) + ".npz")['data']
            print("Loaded saved data")
            return data
        except:
            pass

    observationDim = 3
    realRobot = initialStateMean.copy()
    noiseFreeRobot = initialStateMean.copy()

    field = fieldSettings.field
    Q = np.diag(np.array([0, *(beta**2)]))

    data = np.zeros((numSteps, 9+(5*maxObs)))
    for i in range(maxObs):
        col = 9 + i*5
        data[:, col] -= 1 # set to 'no observation' preemptively
    t = deltaT*np.array(range(numSteps))
    noiseFreeMotion = generateMotion(t)
    for n in range(numSteps):
        # Noise-free robot update
        noiseFreeRobot = sampleOdometry(noiseFreeMotion[n], noiseFreeRobot, [0, 0, 0, 0])

        # Noisy robot update
        realRobot = sampleOdometry(noiseFreeMotion[n], realRobot, alphas)
        markerIds = []

        if (landmark == "fov"):
            markerIds = markersInFov(realRobot, maxObs, landmarkControl)
        elif (landmark == "round-robin"):
            markerIds = np.array([int(np.floor(n/period) % field['numMarkers'])])
        elif (landmark == "random"):
            count = int(np.floor(np.random.uniform() * (maxObs  + 1)))
            for i in range(count):
                markerIds.append(int(np.floor(np.random.uniform() * field['numMarkers'])))
        elif (landmark == "closest" or landmark == "farthest"):
            # find dist to landmarks
            dist = np.sqrt((field['markerPosX'].copy() - noiseFreeRobot[0])**2 + (field['markerPosY'].copy() - noiseFreeRobot[1])**2)
            funcToUse = min if landmark == "closest" else max
            markerIds = np.array([np.where(dist == funcToUse(dist))[0][0]])
        else:
            raise Exception('Landmark selection method "' + landmark + '" is unknown.')

        data[n, 0:3] = noiseFreeMotion[n]
        data[n, 3:6] = realRobot.copy()
        data[n, 6:9] = noiseFreeRobot.copy()
        for i in range(len(markerIds)):
            obs = observation(realRobot, markerIds[i])
            noisyObs = obs + np.random.multivariate_normal(np.zeros(observationDim), Q)
            start = 9+5*i
            data[n, start:start+3] = obs
            data[n, start+3:start+5] = noisyObs[1:]

        np.savez("savedData/" + str(numSteps) + ".npz", data=data)

    return data
