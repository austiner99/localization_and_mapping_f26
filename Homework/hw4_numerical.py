#code to complete SCT portion of homework using numerical methods

import numpy as np
from ssc import ssc_tail2tail, ssc_head2tail, ssc_inverse, draw_ellipse
import matplotlib.pyplot as plt
from scipy.io import loadmat

np.set_printoptions(precision=4, suppress=True, linewidth=150)

def show_pose(title, x):
    x = np.asarray(x).ravel()
    print(title)
    print("   [x y z] (m):                ", x[:3])
    print("   [roll pitch heading] (deg): ", np.degrees(x[3:]))
    print()

def show_mat(title, M):
    M = M[:, :, 0] if M.ndim == 3 else M
    print(f"{title}  (shape {M.shape})\n{M}\n")
    
def numerical_jacobian(f, x):
    """Compute the Jacobian of a vector-valued function f at point x using numerical differentiation."""
    n = x.size
    x = np.asarray(x, dtype=np.float64).ravel()
    f0 = f(x)
    J = np.zeros((f0.size, n))
    for i in range(n):
        delta = max(abs(1e-4*x[i]]), 1e-6)
        x_plus = x.copy()
        x_plus[i] += delta
        diff = f(x_plus) - f0
        diff[3:] = np.arctan2(np.sin(diff[3:]), np.cos(diff[3:]))
        J[:, i] = diff / delta
    return J

col = lambda v: np.asarray(v).reshape(6,1)

np.set_printoptions(precision=4, suppress=True, linewidth=150)

data = loadmat('Homework/data.mat')
xrs1 = data['x_rs1'].astype(np.float64)
xs1s2 = data['x_s1s2'].astype(np.float64)
muwr1 = data['mu_wr1'].astype(np.float64)
sig1 = data['Sigma_1'].astype(np.float64)
muwr2 = data['mu_wr2'].astype(np.float64)
sig2 = data['Sigma_2'].astype(np.float64)
sig12 = data['Sigma_12'].astype(np.float64)

#SCT-Q1: What is the pose of the sensor s1 w.r.t. sensor s2? What is the Jacobian of the associated transformation?

Jxs2s1_num = numerical_jacobian(lambda x: ssc_inverse(col(x))[0].ravel(), xs1s2)

# SCT-Q2. What is the mean pose of sensor s1 in reference frame w at time t1? What is the Jacobian of the associated transformation? 
# To first order, what is the covariance of s1’s pose in reference frame w at time t1?

Ja6_num = numerical_jacobian(lambda x: ssc_head2tail(col(muwr1), col(x))[0].ravel(), xrs1)
sig_sw1_num = Ja6_num @ sig1 @ Ja6_num.T

# SCT-Q3. What is the mean pose, Jacobian, and first-order covariance of sensor s2’s pose in reference frame wat time t1? 
# (Hint: think chain rule.)

j_sw2_num = numerical_jacobian(lambda x: ssc_head2tail(ssc_head2tail(col(x),xrs1)[0], xs1s2)[0].ravel(), muwr1)
sig_sw2_num = j_sw2_num @ sig1 @ j_sw2_num.T

# SCT-Q4. Plot the 3-sigma (x, y) confidence ellipses for global robot poses xwr1 and xwr2 in the world frame. (Hint: extract the 2 × 2 (x, y) 
# submatrix for each 6x6 covariance matrix and use the draw ellipse() function to plot.) Note that each pose has approximately 10m of error in 
# the world reference frame w. Now compute the relative pose xr1r2, i.e., the robo pose at time t2 w.r.t. the robot pose at time
# t1 and plot its first-order (x, y)3 − sigma confidence ellipse. Comment on what you see and explain why

Jr_num = numerical_jacobian(lambda x: ssc_tail2tail(col(x[:6]), col(x[:6]))[0].ravel(), np.concatenate([muwr1.ravel(), muwr2.ravel()]))
sig_r1r2_num = Jr_num @ np.block([[sig1, sig12], [sig12.T, sig2]]) @ Jr_num.T

print("SCT-Q5: analytical vs numerical")
compare("Q1 Jacobian",            Jxs2s1,    Jxs2s1_num)
compare("Q2 Jacobian (d/d x_wr1)", Ja6,       Ja6_num)
compare("Q2 covariance",          sig_ws1,   sig_ws1_num)
compare("Q3 Jacobian (d/d x_wr1)", J_ws2,     J_ws2_num)
compare("Q3 covariance",          sig_ws2,   sig_ws2_num)
compare("Q4 Jacobian (6x12)",     Jr,        Jr_num)
compare("Q4 relative covariance", sig_r1r2,  sig_r1r2_num)
print()
print("SCT-Q4 relative (x, y) std [m]   analytical:", np.sqrt(np.diag(sig_r1r2)[:2]),
      "  numerical:", np.sqrt(np.diag(sig_r1r2_num)[:2]))