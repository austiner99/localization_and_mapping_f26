#!/usr/bin/env python3
"""Module defining several "helper" or "utility" functions useful for the SLAM lab.

Defined Functions: 
-Helpers:
minimizedAngle - normalizes angle in radians from -pi to +pi
meanAndVariance - returns the mean and variance for a set of non-weighted samples (illustrates handling of angles)
debugPrint - just runs print when "printDebug" in this file is set to "True" - default False

-Display:
plotCircle - It, uh... well, it plots a circle.
plotCov2D - Plots an error ellipse for the given 2d covariance matrix
plotField - draws the field with landmarks
plotMarker - Not included any more, just use plotSamples
plotRobot - draws the robot all nice and pretty
plotSamples - plots multiple dots, for use with the particle filter samples
"""
import matplotlib
# matplotlib.use("Qt5Agg") # Might need this on mac if live plotting doesn't work.
import matplotlib.pyplot as plt
import numpy as np
import fieldSettings 

__author__ = "Joshua Mangelson, Joshua Broekhuijsen(translation from MATLAB), and Ryan Eustice(MATLAB)"
__copyright__ = "Copyright 2022 - Joshua Mangelson, Joshua Broekhuijsen, and Ryan Eustice"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"

printDebug = False

# Helper functions
def debugPrint(*args):
    if (printDebug):
        print(*args)

def minimizedAngle(theta):
    return (theta + 3*np.pi)%(2*np.pi)-np.pi

def meanAndVariance(samples, numToCount=None):
    """Calculates the mean and variance of a set of samples in the format [x, y, theta]

    Parameters: 
    samples: list-compatible collection of samples in the format [x, y, theta] - theta should be radians
    numToCount: If, for some reason, you didn't want to average all the samples I guess you can do that here.  Mostly left for legacy reasons. So.  Probably ignore it.
    
    Return: 
    mu: An array of [mu_x, mu_y, mu_theta] with mu_theta in radians.
    Sigma: An array of [s_x, s_y, s_theta] where s_* is the variance for that variable.
    """
    if numToCount is None:
        numToCoun = len(samples)
    
    toAverage = samples[:numToCount].copy()
    mu = np.mean(toAverage, axis=0)
    mu[2] = np.arctan2(np.sum(np.sin(toAverage[:,2])), np.sum(np.cos(toAverage[:,2])))
    zeroMean = toAverage - mu
    zeroMean[:,2] = minimizedAngle(zeroMean[:,2])
    var = np.mean(zeroMean**2, axis=0)

    return mu, var

# Display functions
def plotCircle(center=[0, 0], radius=1, NOP=360, color="red", fillColor=None):
    """This will plot a circle - which I sincerely hope you can figure out.

    Parameters: 
    center: Center of the circle to be plotted, [x, y] default [0, 0]
    radius: The radius of the circle to be plotted, default 1
    NOP: The number of points the define the circle, default 360 which might be overkill
    color: What color (use hex for #RGBA) to make the border - default red, full opacity
    fillColor: what color to fill the circle, defaults to none.
    
    Return: 
    None
    """
    center = np.array(center)
    t = np.linspace(0, 2*np.pi, NOP)
    X = center[0] + radius*np.cos(t)
    Y = center[1] + radius*np.sin(t)
    plt.plot(X, Y, color=color)
    if (fillColor is not None):
        plt.fill(X, Y, color=fillColor)

def plotCov2D(center=[0,0], cov=[[1, 0],[0, 1]], color="red", fillColor=None, nSigma=1, NOP=360):
    """This will plot a 2d error ellipse for the given covariance matrix

    Parameters: 
    center: Center of the circle to be plotted, [x, y] default [0, 0]
    cov: The covariance matrix we're plotting - defaults to the identity matrix, but that's not useful.
    color: What color (use hex for #RGBA) to make the border - default red, full opacity - can provide array for multiple nSigma
    fillColor: What color to fill the circle, defaults to none - can provide array for multiple nSigma
    nSigma: The mahalanobis distance to plot - can provide array
    NOP: The number of points used to draw the error ellipses.  This cannot be an array, they all have to be the same.
    
    Return: 
    None
    """
    color = np.array([color]).reshape(-1)
    fillColor = np.array([fillColor]).reshape(-1)
    nSigma = np.array([nSigma]).reshape(-1)

    color = (np.tile(color, int(np.ceil(len(nSigma)/len(color))))).flatten()[:len(nSigma)] if len(color) < len(nSigma) else color
    fillColor = (np.tile(fillColor, int(np.ceil(len(nSigma)/len(fillColor))))).flatten()[:len(nSigma)] if len(fillColor) < len(nSigma) else fillColor

    evals, evects = np.linalg.eig(cov)
    debugPrint("evals:", evals)
    debugPrint("evects:", evects)
    maxIndex = 0 if np.max(evals) == evals[0] else 1
    minIndex = 1 if maxIndex == 0 else 0
    angle = -np.arctan2(evects[:,maxIndex][1],evects[:,maxIndex][0])

    tSpace = np.linspace(0, 2*np.pi, NOP)
    ellMat = np.column_stack([np.cos(tSpace), np.sin(tSpace)])

    axisBase = np.sqrt(evals)

    for index in range(len(nSigma)):
        sigma = nSigma[index]
        thisColor = color[index]
        thisFill = fillColor[index]

        majLen = axisBase[maxIndex]*sigma
        minLen = axisBase[minIndex]*sigma
        ellMatCopy = ellMat.copy()
        ellMatCopy[:,0] *= majLen
        ellMatCopy[:,1] *= minLen
        rotMatrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        rotatedEll = ellMatCopy@rotMatrix
        plt.plot(center[0] + rotatedEll[:,0], center[1] + rotatedEll[:,1], color=thisColor)
        if (thisFill is not None):
            plt.fill(center[0] + rotatedEll[:,0], center[1] + rotatedEll[:,1], color=thisFill)

def plotRobot(pose=[0, 0, 0], color="red", fillColor=None, r=13):
    """This will draw the robot on the field, all nice and pretty

    Parameters: 
    pose: Pose [x, y, theta] of the robot to plot, default [0, 0, 0] but why would you use the default?
    color: What color (use hex for #RGBA) to make the outline of the robot - default red, full opacity
    fillColor: What color to fill the robot, defaults to none
    
    Return: 
    None
    """
    plotCircle(pose[:2], r, color=color, fillColor = fillColor)
    plt.plot(np.array([pose[0], pose[0] + np.cos(pose[2])*r*1.5]), np.array([pose[1], pose[1] + np.sin(pose[2])*r*1.5]), color="black")

def plotSamples(samples, color="red"):
    """This will draw samples on the field

    Parameters: 
    samples: Collection of samples, format [[x, y], [x, y], ...] to plot
    color: What color (use hex for #RGBA) to make the points - default red, full opacity
    
    Return: 
    None
    """
    plt.plot(samples[:,0], samples[:, 1], color=color)

def plotField(detectedMarkers=np.array([])):
    """This will plot the field at a given state, pulling from the fieldSettings imported from fieldSettings.py

    Parameters: 
    detectedMarkers: An array of specific marker indices which will be highlighted in the plot.  Default none.
    
    Return: 
    None
    """
    field = fieldSettings.field

    margin = 200
    plt.axis('equal')
    plt.xlim(-margin, field['completeSizeX'] + margin)
    plt.ylim(-margin, field['completeSizeY'] + margin)

    for k in range(field['numMarkers']):
        plotCircle(np.array([field['markerPosX'][k], field['markerPosY'][k]]), 15, 200, 'black', '#00000040' if k in detectedMarkers else '#FFFFFF')
        plt.text(field['markerPosX'][k], field['markerPosY'][k], str(k), ha='center', va='center')
