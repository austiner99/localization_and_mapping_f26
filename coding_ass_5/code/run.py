#!/usr/bin/env python3
"""Module for the main control in the SLAM lab: gathers data and executes loop

Defined Classes: 
ParticleFilterLocalizer - Implements particle filter localization.
"""

__author__ = "Joshua Mangelson, Joshua Broekhuijsen(translation from MATLAB), and Ryan Eustice(MATLAB)"
__copyright__ = "Copyright 2022 - Joshua Mangelson, Joshua Broekhuijsen, and Ryan Eustice"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"


import numpy as np
import argparse

def run(numSteps, dataType, dataAssociation, updateMethod="batch", pauseLen=0.3, makeGif = False):
    """Function that will actually execute SLAM - you tell it:

    Parameters:
    numSteps: The number of steps to test in the dataset (whether simulated or Victoria Park)
    dataType: "sim" to generate new simulation data, or "vp" for victoria park - defaults to "sim"
    dataAssociation: "known", "nn", "nndg", or "jcbb" to select how data association is performed.
    updateMethod: Whether to use "batch" or "seq"uential updating - defaults to "batch"
    pauseLen: How long to pause when displaying each step.  Defaults to 0.3s
    """
 
    if (dataType == "sim"):
        from simSLAM import run
    elif (dataType == "vp"):
        from vpSLAM import run
    else:
        raise Exception("Unknown data type when attempting to run.")
    
    run(numSteps, dataAssociation, updateMethod, pauseLen, makeGif)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EKF SLAM Lab")
    parser.add_argument("-g", "--makeGif", action="store_true", help="Whether to save a gif of all the frames")
    parser.add_argument("-t", "--dataType", type=str, default=None, required=True, help="Which type of data? 'sim' or 'vp'")
    parser.add_argument("-u", "--updateMethod", type=str, default="batch", help="Which update type to use? 'batch' or 'seq'")
    parser.add_argument("-d", "--dataAssociation", type=str, default=None, required=True, help="Which data association method to use? 'known', 'nn', 'nndg', or 'jcbb'")    
    parser.add_argument("-w", "--pauseLen", type=float, default=0.1, help="Time to pause on each frame when viewing live.")
    parser.add_argument("-n", "--numSteps", type=int, default=None, required=True, help="Number of steps to take")
    args = vars(parser.parse_args())

    run(**args)
