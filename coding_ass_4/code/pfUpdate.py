#!/usr/bin/env python3
"""Module for the PF portion of the landmark/localization lab

Defined Functions: 
pfUpdate - the update function for the PF
"""

from fieldSettings import field
import helpers

def pfUpdate(mu, Sigma, u, deltaT, M, z, Q, markerId):
    # NOTE: The header is not set in stone.  You may change it if you like.
    landmark_x = field['markerPosX'][markerId]
    landmark_y = field['markerPosY'][markerId]

    stateDim=3
    motionDim=3
    observationDim=1

    #######################################################
    # Prediction step
    #######################################################

    # some stuff goes here

    # Compute mean and variance of estimate. Not really needed for inference.
    # predMu, predSigma = helpers.meanAndVariance(samples, numSamples)


    #######################################################
    # Correction step
    #######################################################

    # more stuff goes here
