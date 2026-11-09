#code to complete SCT portion of homework

# Each of the above functions returns the 6-DOF pose vector result of the coordinate frame operation and (optionally) the resulting analytical 
# Jacobian. Use these functions to answer the scenario described below. For these problems, load the provided data file data.mat into your workspace.
# Suppose that we have a robot instrumented with two sensors s1 and s2. Sensor s1 has a static pose in the
# robot coordinate frame given by the variable xrs1, and sensor s2 has a static pose defined with respect to (w.r.t.) sensor s1 given by variable xs1s2.
# These transforms are assumed to be known exactly (i.e. well-calibrated). The robot’s own pose is less certain and is expressed w.r.t. a world reference 
# frame w. The robot’s pose sampled at time t1, xwr1, has a mean and covariance given by variables µwr1 and Σ1, respectively. And at a short time later, t2,
# after making an odometry move of a couple meters, the robot has a pose xwr2 with mean and covariance given by variables µwr2 and Σ2, respectively. 
# The cross-covariance between the robot’s pose at time t1 and t2 is given by the 6 × 6 matrix variable Σ12.

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

xs2s1, Jxs2s1 = ssc_inverse(xs1s2)

show_pose("SCT-Q1: pose of sensor s1 w.r.t. sensor s2", xs2s1)
show_mat("SCT-Q1: Jacobian d(xs1s2)/d(xs1s2)", Jxs2s1)

# SCT-Q2. What is the mean pose of sensor s1 in reference frame w at time t1? What is the Jacobian of the associated transformation? 
# To first order, what is the covariance of s1’s pose in reference frame w at time t1?

x_ws1, Ja = ssc_head2tail(muwr1, xrs1)
Ja6 = Ja[:, :6, 0]
sig_ws1 = Ja6 @ sig1 @ Ja6.T

show_pose("SCT-Q2: mean pose of sensor s1 in reference frame w at time t1", x_ws1)
show_mat("SCT-Q2: Jacobian (d(s_ws1)/d(x_ws1, muwr1))", Ja6)
show_mat("SCT-Q2: first-order covariance of x_ws1", sig_ws1)

# SCT-Q3. What is the mean pose, Jacobian, and first-order covariance of sensor s2’s pose in reference frame wat time t1? 
# (Hint: think chain rule.)

x_ws2, Jb = ssc_head2tail(x_ws1, xs1s2)
Jb6 = Jb[:, :6, 0]
J_ws2 = Jb6 @ Ja6
sig_ws2 = J_ws2 @ sig1 @ J_ws2.T

show_pose("SCT-Q3: mean pose of sensor s2 in reference frame w at time t1", x_ws2[:6, 0])
show_mat("SCT-Q3: Jacobian (d(s_ws2)/d(x_ws1, xs1s2))", J_ws2)
show_mat("SCT-Q3: first-order covariance of x_ws2", sig_ws2)

# SCT-Q4. Plot the 3-sigma (x, y) confidence ellipses for global robot poses xwr1 and xwr2 in the world frame. (Hint: extract the 2 × 2 (x, y) 
# submatrix for each 6x6 covariance matrix and use the draw ellipse() function to plot.) Note that each pose has approximately 10m of error in 
# the world reference frame w. Now compute the relative pose xr1r2, i.e., the robo pose at time t2 w.r.t. the robot pose at time
# t1 and plot its first-order (x, y)3 − sigma confidence ellipse. Comment on what you see and explain why

plt.figure()
draw_ellipse(muwr1[:2, 0], sig1[:2, :2], 9, color='r', show=False)
draw_ellipse(muwr2[:2, 0], sig2[:2, :2], 9, color='b', show=False)
plt.plot(*muwr1[:2, 0], 'r.', label='x_wr1')
plt.plot(*muwr2[:2, 0], 'b.', label='x_wr2')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('SCTQ4_global.png')

#relative pose

xr1r2, Jr1r2 = ssc_tail2tail(muwr1, muwr2)
Jr1r2_6 = Jr1r2[:, :, 0]
P = np.block([[sig1, sig12], [sig12.T, sig2]])
sig_r1r2 = Jr1r2_6 @ P @ Jr1r2_6.T

show_pose("SCT-Q4: relative pose of robot at time t2 w.r.t. robot at time t1", xr1r2[:6, 0])
print("SCT-Q4: (x,y) standard deviations [m]")
print("   global x_wr1:                ", np.sqrt(np.diag(sig1)[:2]))
print("   global x_wr2:                ", np.sqrt(np.diag(sig2)[:2]))
print("   relative x_r1r2:             ", np.sqrt(np.diag(sig_r1r2)[:2]))
print()

plt.figure()
draw_ellipse(xr1r2[:2, 0], sig_r1r2[:2, :2], 9, color='g', show=False)
plt.plot(*xr1r2[:2, 0], 'g.', label='x_r1r2')
plt.legend()
plt.savefig('SCTQ4_relative.png')

#SCT-Q5. Sometimes, analytically computing the Jacobian is too error-prone and/or tedious. Instead, we can numerically compute the Jacobian 
# with a good degree of accuracy. Excerpted below from the textbook “Multiple View Geometry” by Hartley and Zissermen is the outline of an 
# algorithm for numerically computing a vector-valued function’s Jacobian.

    # Numerical differentiation may be carried out as follows. Each independent variable xi is incremented in turn to xi + δ, the resulting 
    # function value is computed using the routine provided for computing f and the derivative is computed as a ratio. Good results have
    # been found by setting the value δ to the maximum of |10−4 × xi | and 10−6. This choice seemingly gives a good approximation to the derivative. 
    # In practice, there seems to be little disadvantage in using numerical differentiation, though for simple functions f one may prefer to provide
    # a routine to compute J, partly for aesthetic reasons, partly because of a possible slightly improved convergence and partly for speed.
    
# Rewrite your software for SCT-Q1 to SCT-Q4 to use numerically computed Jacobians instead of the analytically evaluated ones returned by the 
# ssc * family of functions. Compare your results to those obtained using the analytical Jacobians.

def numerical_jacobian(f, x):
    """Compute the Jacobian of a vector-valued function f at point x using numerical differentiation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    f0 = f(x)
    n = x.size
    J = np.zeros((f0.size, n))
    for i in range(n):
        delta = max(abs(1e-4*x[i]), 1e-6)
        x_plus = x.copy()
        x_plus[i] += delta
        diff = f(x_plus) - f0
        diff[3:] = np.arctan2(np.sin(diff[3:]), np.cos(diff[3:]))
        J[:, i] = diff / delta
    return J

col = lambda v: np.asarray(v).reshape(6,1)

#SCT-Q1: What is the pose of the sensor s1 w.r.t. sensor s2? What is the Jacobian of the associated transformation?

Jxs2s1_num = numerical_jacobian(lambda x: ssc_inverse(col(x))[0].ravel(), xs1s2)

# SCT-Q2. What is the mean pose of sensor s1 in reference frame w at time t1? What is the Jacobian of the associated transformation? 
# To first order, what is the covariance of s1’s pose in reference frame w at time t1?

Ja6_num = numerical_jacobian(lambda x: ssc_head2tail(col(x), xrs1)[0].ravel(), muwr1)
sig_ws1_num = Ja6_num @ sig1 @ Ja6_num.T

# SCT-Q3. What is the mean pose, Jacobian, and first-order covariance of sensor s2’s pose in reference frame wat time t1? 
# (Hint: think chain rule.)

J_ws2_num = numerical_jacobian(lambda x: ssc_head2tail(ssc_head2tail(col(x),xrs1)[0], xs1s2)[0].ravel(), muwr1)
sig_ws2_num = J_ws2_num @ sig1 @ J_ws2_num.T

# SCT-Q4. Plot the 3-sigma (x, y) confidence ellipses for global robot poses xwr1 and xwr2 in the world frame. (Hint: extract the 2 × 2 (x, y) 
# submatrix for each 6x6 covariance matrix and use the draw ellipse() function to plot.) Note that each pose has approximately 10m of error in 
# the world reference frame w. Now compute the relative pose xr1r2, i.e., the robo pose at time t2 w.r.t. the robot pose at time
# t1 and plot its first-order (x, y)3 − sigma confidence ellipse. Comment on what you see and explain why

Jr_num = numerical_jacobian(lambda x: ssc_tail2tail(col(x[:6]), col(x[6:]))[0].ravel(), np.concatenate([muwr1.ravel(), muwr2.ravel()]))
sig_r1r2_num = Jr_num @ np.block([[sig1, sig12], [sig12.T, sig2]]) @ Jr_num.T

def compare(name, A, B):
    A = A[:, :, 0] if A.ndim == 3 else A
    B = B[:, :, 0] if B.ndim == 3 else B
    err = np.abs(A - B).max()
    rel = err / np.abs(A).max()
    status = "OK" if rel < 1e-3 else "MISMATCH  <-- check this one"
    print(f"    {name:26s} max abs err = {err:9.2e}   relative = {rel:9.2e}   {status}")

print("SCT-Q5: analytical vs numerical")
compare("Q1 Jacobian",             Jxs2s1,   Jxs2s1_num)
compare("Q2 Jacobian (d/d x_wr1)", Ja6,      Ja6_num)
compare("Q2 covariance",           sig_ws1,  sig_ws1_num)
compare("Q3 Jacobian (d/d x_wr1)", J_ws2,    J_ws2_num)
compare("Q3 covariance",           sig_ws2,  sig_ws2_num)
compare("Q4 Jacobian (6x12)",      Jr1r2,    Jr_num)
compare("Q4 relative covariance",  sig_r1r2, sig_r1r2_num)
print()
print("SCT-Q4 relative (x, y) std [m]   analytical:", np.sqrt(np.diag(sig_r1r2)[:2]),
      "  numerical:", np.sqrt(np.diag(sig_r1r2_num)[:2]))

plt.show()