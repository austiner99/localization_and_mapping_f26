import numpy as np
import helpers

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
    # Short circuit threshhold is used in jcbb, not implemented here
def associateData(measurements, R, mu, Sigma, landmarkSignatures = [], shortCircuitThresh = 40.0**2):
    association = [] # Each index of this will hold the index of the landmark in the state.
    innovation = [] # Indexed the same as the above, this is used in the update step
    H = [] # Indexed the same as the above, this is used in the update step
    
    x, y, theta = mu[:3] # We'll use this for innovation and H, even if not for data association.

    for z in measurements: # OK, for each measurement we have...
        dist, bearing, sig = z # Let's extract the components
        if sig in landmarkSignatures: # If we've seen this signature (markerID) before, it's in the state

            # Association
            ind = np.flatnonzero(np.asarray(landmarkSignatures == sig))[0] # We just need to find WHERE in the state
            association.append(ind) # And make sure we return that

            # Innovation
            # TODO: Calculate the Innovation HERE

            # Jacobian
            # TODO: Calculate the Jacobian HERE

        # TODO: Read this so you understand what its doing
        else: # If the landmark signature (markerID) HASN'T been seen yet, we need to create a new landmark
            association.append(-1) # We say the association is "-1" to tell future code to add new landmarks
            innovation.append([0, 0]) # We won't use the innovation anyway, but we need to return SOMETHING
            H.append(np.zeros((2, len(mu)))) # Same for our H jacobian

    return np.array(association), np.array(H), np.array(innovation)
