#!/usr/bin/env python3
"""Module defining the settings of the field used in the landmark/localization lab.  Probably don't change these or else the test dataset will probaby not work.

Defined Data: 
-field:
completeSize(X, Y) - The total size of the field in the X or Y direction
numMarkers - The total number of markers, determined by numMarkersX and numMarkersY
markerPos(X, Y) - The X or Y coordinates of a given marker index (markerPosX[1] is the x-coord of marker 1, etc)


"""
import numpy as np

__author__ = "Joshua Mangelson, Joshua Broekhuijsen(translation from MATLAB), and Ryan Eustice(MATLAB)"
__copyright__ = "Copyright 2022 - Joshua Mangelson, Joshua Broekhuijsen, and Ryan Eustice"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"

def settings():
    fieldSettings = {}

    numMarkersX = 3
    numMarkersY = 2

    innerSizeX = 420
    innerSizeY = 270

    innerOffsetX = 32
    innerOffsetY = 13

    markerOffsetX = 21
    markerOffsetY = 0

    fieldSettings['numMarkers'] = numMarkersX * numMarkersY

    fieldSettings['completeSizeX'] = innerSizeX + 2 * innerOffsetX
    fieldSettings['completeSizeY'] = innerSizeY + 2 * innerOffsetY
    markerDistX = fieldSettings['completeSizeX'] - 2 * markerOffsetX
    markerDistY = fieldSettings['completeSizeY'] - 2 * markerOffsetY


    oneXCoords = np.linspace(markerOffsetX, markerDistX, numMarkersX)
    oneYCoords = np.linspace(markerOffsetY, markerDistY, numMarkersY)

    xCoords = []
    yCoords = []

    for i in range(numMarkersY):
        xCoords.append(oneXCoords.copy() if not i%2 else np.flip(oneXCoords.copy()))
        yCoords.append(np.full(oneXCoords.shape, oneYCoords[i]))
    
    fieldSettings['markerPosX'] = np.array(xCoords).flatten()
    fieldSettings['markerPosY'] = np.array(yCoords).flatten()

    return fieldSettings

field = settings()
