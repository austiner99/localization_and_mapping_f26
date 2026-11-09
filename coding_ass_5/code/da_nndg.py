import numpy as np
import helpers

# NN double gate is the same as "Nearest Neighbor," except...
# We add in a "no-man's land."  If we are below a certain threshold (mahalanobis dist), we'll associate to that landmark.
# If we're above another threshhold, it'll be used to create a new landmark in our state.
# If we're in between those thresholds, the measurement is too ambiguous and we'll just ignore it.

# Below this we'll associate
botThresh = # TODO: Tune your threshold
# Higher than this is a new landmark
topThresh = # TODO: Tune your threshold

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
