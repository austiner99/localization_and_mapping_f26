import numpy as np
import helpers

# Nearest neighbor is in terms of mahalanobis distance

# Lower than this, we associate.  Higher is a new landmark.
startingThresh = # TODO: Tune your threshold 

####################################
# TODO: Implement Data Association
####################################
# Output:
#  - an "association" vector that specifies the landmark index of each measurement. (-1 means new landmark, -2 means ignore)
#  - Since MOST data associations (NN, JCBB) end up computing innovation and measurement jacobians,
#    it can save computation to pass them on from this function. 
#
# Inputs:
#  - measurements obtained at this time step
#  - R covariance matrix
#  - Curren state estimate (mu, Sigma)
#  - Landmark signatures (only available for use in da_known)
#  - Short circuit threshhold may be helpful for use in jcbb, not needed in the other algorithms
def associateData(measurements, R, mu, Sigma, landmarkSignatures = [], shortCircuitThresh = 40.0**2):

    return np.array(association), np.array(H), np.array(innovation)
