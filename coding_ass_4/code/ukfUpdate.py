#!/usr/bin/env python3
"""Module for the UKF portion of the landmark/localization lab

Defined Functions: 
UkfUpdate - the update function for the UKF
"""

from fieldSettings import field
import numpy as np
import helpers
from helpers import debugPrint

#################################################
# TODO: Implement the Motion Model for the robot
def g(mu, u):

    th1 = u[0]
    d = u[1]
    th2 = u[2]
    phi = mu[2] + th1
    x = mu[0] + d*np.cos(phi)
    y = mu[1] + d*np.sin(phi)
    t = helpers.minimizedAngle(mu[2] + th1 + th2)

    return np.array([x, y, t])

######################################################
# TODO: Implement the Measurement Model 
# NOTE: The header is not set in stone.  You may change it if you like.
def h(mu_bar, markerLocation): #Header Option 1

    dx = markerLocation[0] - mu_bar[0]
    dy = markerLocation[1] - mu_bar[1]
    th = mu_bar[2]
    angle = np.arctan2(dy, dx) - th
    angle = helpers.minimizedAngle(angle)

    return angle

#######################################################
# TODO: Write a Function to Generate your sigma points
def makeSigmaPoints(mu, Sigma):
    
    #sigma point eq: x_0 = mu; x_i = mu +/- (sqrt((n+lamda)*Sigma)) - sigma square root will have to use a matrix root method
    #weight eq: w_m_0 = lambda/(n+lambda); w_c_0 = lambda/(n+lambda) + (1 - alpha**2 + beta); w_m_i=w_c_i = 1/(2(n+lambda))
    n = len(mu)
    alpha = 1
    beta = 2
    k = 0
    lam = 0
    w_m_0 = lam/(n+lam)
    w_c_0 = lam/(n+lam) + (1 - alpha**2 + beta)
    w_m_i = 1/(2*(n+lam))
    w_c_i = 1/(2*(n+lam))
    
    # Generate sigma points
    sigmaPoints = [mu]
    root = np.sqrt((n + lam))*np.linalg.cholesky(Sigma)
    for i in range(n):
        sigmaPoints.append(mu + root[:, i])
        sigmaPoints.append(mu - root[:, i])

    # Generate weights
    weights_m = [w_m_0]
    for i in range(n):
        weights_m.append(w_m_i)
        weights_m.append(w_m_i)

    weights_c = [w_c_0]
    for i in range(n):
        weights_c.append(w_c_i)
        weights_c.append(w_c_i)

    sigmaPoints = np.array(sigmaPoints)
    weights = np.zeros([len(weights_m),2])
    weights[:,0] = weights_m
    weights[:,1] = weights_c
    return sigmaPoints, weights

###########################################################
# TODO: Implement the UKF Update
#
# Make sure you use the unscented transform to
# propogate the uncertainty through both the prediction
# (Update Step) and measurement models (Correction Step).
#
# This function should return an updated mean
# and covariance for your state.
def ukfUpdate(mu, Sigma, u, M, z, Q, markerId):
    # NOTE: The header is not set in stone.  You may change it if you like.
    landmark_x = field['markerPosX'][markerId]
    landmark_y = field['markerPosY'][markerId]

    stateDim=3
    motionDim=3
    observationDim=1

    #######################################################
    # Setup UKF
    #######################################################
    
    mu_aug = np.zeros(stateDim + motionDim+observationDim)
    mu_aug[:stateDim] = mu
    Sigma_aug = np.zeros((stateDim + motionDim+observationDim, stateDim + motionDim+observationDim))
    Sigma_aug[:stateDim, :stateDim] = Sigma
    Sigma_aug[stateDim:stateDim+motionDim, stateDim:stateDim+motionDim] = M
    Sigma_aug[stateDim+motionDim:stateDim+motionDim+observationDim, stateDim+motionDim:stateDim+motionDim+observationDim] = Q
    x, weights = makeSigmaPoints(mu_aug, Sigma_aug)

    #######################################################
    # Prediction step
    #######################################################
    
    # Unscented Transform Prediction of Propogated Mean and Covariance
    x_bar_star = []
    for i in range(len(x)):
        x_bar_star.append(g(x[i, :stateDim], u+x[i, stateDim:stateDim+motionDim]))
    x_bar_star =  np.array(x_bar_star)
    
    mu_bar = np.zeros(stateDim)    
    mu_bar[:2] = weights[:,0] @ x_bar_star[:, :2]
    mu_bar[2] = np.arctan2(weights[:,0] @ np.sin(x_bar_star[:, 2]), weights[:,0] @ np.cos(x_bar_star[:, 2]))

    sigma_bar = np.zeros((stateDim, stateDim))
    for i in range(len(x_bar_star)):
        diff = x_bar_star[i] - mu_bar
        diff[2] = helpers.minimizedAngle(diff[2])
        sigma_bar += weights[i, 1] * np.outer(diff, diff)
    

    #######################################################
    # Correction step
    #######################################################

    #Unscented Transform Prediction of Measurement/Innovation Mean/Covariance
    
    x_bar, weights = makeSigmaPoints(mu_bar, sigma_bar)
    
    z_bar = np.array([h(x_bar[i], (landmark_x, landmark_y)) for i in range(len(x_bar))])     
    z_carrot = np.arctan2(weights[:, 0] @ np.sin(z_bar), weights[:, 0] @ np.cos(z_bar))
        
    dz = helpers.minimizedAngle(z_bar-z_carrot)
    s_t = float(weights[:, 1] @ dz**2) + float(np.squeeze(Q))
    
    dx = x_bar-mu_bar
    dx[:, 2] = helpers.minimizedAngle(dx[:, 2])
    sig_xz = (weights[:, 1] * dz @ dx)
    
    # UKF Correction Equations
    K = sig_xz/s_t
    innov = helpers.minimizedAngle(float(np.squeeze(z))-z_carrot)
    mu_new = mu_bar + K * innov
    mu_new[2] = helpers.minimizedAngle(mu_new[2])
    Sigma_new = sigma_bar - s_t*np.outer(K, K)
    return mu_new, Sigma_new