#!/usr/bin/env python3
"""Module for the EKF portion of the landmark/localization lab

Defined Functions: 
ekfUpdate - the update function for the EKF
"""

from fieldSettings import field
import helpers
import numpy as np

#################################################
# TODO: Implement the Motion Model for the robot
#
# This should output a vector with an updated pose
# and two Jacobians - one (R) w/respect to the motion
# commands and one (G) w/respect to the previous state.
def g(motion, mu):
    th1 = motion[0]
    d = motion[1]
    th2 = motion[2]
    phi = mu[2]+th1
    x = mu[0] + d*np.cos(phi)
    y = mu[1] + d*np.sin(phi)
    angle = helpers.minimizedAngle(mu[2] + th1 + th2)
    
    G = np.array([[1, 0, -d*np.sin(phi)],
                  [0, 1, d*np.cos(phi)],
                  [0, 0, 1]])
    
    R = np.array([[-d*np.sin(phi), np.cos(phi), 0],
                  [d*np.cos(phi), np.sin(phi), 0],
                  [1, 0, 1]])

    return np.array([x, y, angle]), G, R

######################################################
# TODO: Implement the Measurement Model 
#
# This should return the predicted measurement and
# the jacobian (H) with respect to the previous state
def h(mu_bar, markerPos): 
    
    dx = markerPos[0] - mu_bar[0]
    dy = markerPos[1] - mu_bar[1]
    th = mu_bar[2]
    v = dx**2 + dy**2
    z = np.arctan2(dy, dx) - th
    z = helpers.minimizedAngle(z)
    
    H = np.array([[dy/v, -dx/v, -1]]) #CHECK: check signs for first two if wrong

    return np.array(z), H


########################################################
# TODO: Implement the ekfUpdate Equations
#
# This should return the new state and covariance after
# updating for motion and the recieved measurement
def ekfUpdate(mu, Sigma, u, M, z, Q, markerId):
    # NOTE: The header is not set in stone.  You may change it if you like.
    landmark_x = field['markerPosX'][markerId]
    landmark_y = field['markerPosY'][markerId]

    stateDim=3
    motionDim=3
    observationDim=1

    #######################################################
    # Prediction step
    #######################################################
    # EKF prediction of mean and covariance
    
    mu_bar, G, R = g(u, mu)
    sigma_bar = G @ Sigma @ G.T + R @ M @ R.T

    #######################################################
    # Correction step
    #######################################################

    # Compute expected observation and Jacobian
    z_pred, H = h(mu_bar, (landmark_x, landmark_y))

    # Innovation / residual covariance
    S = H @ sigma_bar @ H.T + Q

    # Kalman gain
    K = sigma_bar @ H.T @ np.linalg.inv(S)

    # Correction
    mu_new = mu_bar + (K * helpers.minimizedAngle(float(np.squeeze(z)) - float(z_pred))).ravel()
    Sigma_new = (np.eye(stateDim) - K@H) @ sigma_bar

    return mu_new[:stateDim], Sigma_new[0:stateDim, 0:stateDim]
