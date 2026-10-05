# In most SLAM or localization approaches, the noise characteristics of individual observations are linearized,
# resulting in a Gaussian covariance matrix. In this task, we explore the effects of the error introduced via linearization.
# Consider a robot sensor that observes range and bearing to nearby landmarks. In this case, the range error
# is relatively small, but the bearing error is large. We are interested in determining the (x, y) position of the
# beacon based on observations obtained by the robot at the origin (0, 0). For this problem, the robot does not move.
# Suppose you obtain a (range, bearing) observation with mean (10.0m, 0rad) whose range standard deviation is 0.5m, and 
# whose bearing standard deviation is 0.25rad. The noise in range and bearing measurements is Gaussian and independent.
# Use python to answer the following questions:

# Q 0:  Generate a function in python that will plot the n-sigma ellipse for a 2d-guassian distribution given
#       a mean vector and covariance matrix. Use the matlab file plotcov2d.m for reference. Try generating
#       10,000 samples from several different distributions (by varying the mean and covariance matrix) and
#       then plot the points and the covariance ellipse.

import numpy as np
import matplotlib.pyplot as plt

def plot_cov_2d(mean, cov, color='black', filled=False, ax=None, fillColor='blue', falpha=1.0, nsigma=1, label = None):
    '''
    Plot a 2D covariance ellipse.
    Input:
        mean: 2D mean vector
        cov: 2x2 covariance matrix
        color: color of the ellipse
        filled: whether the ellipse is filled (bool)
        fillColor: color of the fill
        falpha: alpha value for the fill
        nsigma: number of standard deviations to plot
    '''
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if ax is None:
        ax = plt.gca()
    D, V = np.linalg.eig(cov)
    
    lam1 = D.real[0]
    lam2 = D.real[1]
    v1 = V.real[:, 0]
    v2 = V.real[:, 1]
    if v1[0] == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan2(v1[1], v1[0])
        
    a = nsigma * np.sqrt(max(lam1, 0))
    b = nsigma * np.sqrt(max(lam2, 0))
    
    num_points = 500
    ang = np.arange(num_points+1) * 2 * np.pi / num_points
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])
    pts = mean[:, None] + R@np.vstack([a*np.cos(ang), b*np.sin(ang)])
    
    ax.plot(pts[0, :], pts[1, :], color=color, linewidth=2, label=label)
    
    if filled:
        ax.fill(pts[0, :], pts[1, :], color=fillColor, alpha=falpha)
    
    minor1, minor2 = mean-a*v1, mean+a*v1
    ax.plot([minor1[0], minor2[0]], [minor1[1], minor2[1]], color=color, linewidth=1)
    major1, major2 = mean-b*v2, mean+b*v2
    ax.plot([major1[0], major2[0]], [major1[1], major2[1]], color=color, linewidth=1)

#check function

# mean = np.array([4, 0])
# cov = np.array([[4, 1], [1, 1]])
# plot_cov_2d(mean, cov, filled=True, fillColor='blue', falpha=0.5)
# plt.show()
# plt.savefig('UPQ0.png')

# Q 1:  Generate and plot a point cloud representing 10,000 samples from the distribution over the position
#       of the beacon as measured in
#       i) the sensor frame, i.e. (r, θ) space, and
#       ii) the Cartesian (x, y) coordinate frame.
#       In other words, generate observations of (range, bearing) and project these points into (x, y). (Hint:
#       use the numpy.random.normal function).

N = 10000
r0, theta0 = 10.0, 0
sigma_r, sigma_theta = 0.5, 0.25
rand = np.random.default_rng(42)
r_samples = rand.normal(r0, sigma_r, N)
theta_samples = rand.normal(theta0, sigma_theta, N)
x_samples = r_samples * np.cos(theta_samples)
y_samples = r_samples * np.sin(theta_samples)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].scatter(r_samples, theta_samples, s=1, alpha=0.5)
axs[0].set_xlabel('r')
axs[0].set_ylabel('θ')
axs[0].set_title('Point Cloud in (r, θ) Space')
axs[0].axis('equal')

axs[1].scatter(x_samples, y_samples, s=1, alpha=0.5)
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title('Point Cloud in (x, y) Space')
axs[1].axis('equal')

plt.savefig('UPQ1.png')
plt.show()


# Q 2:  What is the (linearized) covariance of the beacon position in (x, y) coordinates? In other words, write
#       the covariance of an observation in (x, y) coordinates in terms of the covariance of the observation
#       in (range, bearing) coordinates. The transformation is non-linear, so you will need to compute a
#       first-order approximation (Taylor expansion) of the transformation function. Make the appropriate
#       Jacobians easy to read in your source code, using comments if necessary.
import sympy as sp

r, theta = sp.symbols('r theta')
x = r * sp.cos(theta)
y = r * sp.sin(theta)

# The transformation from (r, θ) to (x, y) is:
# x = r * cos(θ)
# y = r * sin(θ)

J = sp.Matrix([[sp.diff(x, r), sp.diff(x, theta)],
              [sp.diff(y, r), sp.diff(y, theta)]])

print("Jacobian:\n")
sp.pprint(J)

sdev_r, sdev_theta = sp.symbols('sigma_r sigma_theta')

sig_rtheta = sp.Matrix([[sdev_r**2, 0], [0, sdev_theta**2]])
sig_rtheta_vals = sp.Matrix([[0.25, 0], [0, 0.0625]])

sig_xy_vals = J @ sig_rtheta_vals @ J.T
sig_xy = J @ sig_rtheta @ J.T
print("\nAnalytical covariance in (x, y) coordinates:\n")
sp.pprint(sp.simplify(sig_xy_vals))
print("\nAnalytical covariance in (x, y) coordinates (symbolic):\n")
sp.pprint(sp.simplify(sig_xy))

# Q 3:  Draw in red the 1-sigma, 2-sigma, and 3-sigma contours of the analytical (linearized) covariance ellipses, 
#       super-imposed over the point clouds generated in parts UP-Q1.i and UP-Q1.ii. Now overlay in
#       blue the actual covariance ellipses computed using sample-based expressions for the first and second
#       moments. Do they agree? Why or why not?

mean_rtheta_sample = np.array([r_samples.mean(), theta_samples.mean()])
sigma_rtheta_sample = np.array([[0.25, 0], [0, 0.0625]])

fig3, ax3 = plt.subplots(1, 2, figsize=(10, 5))
ax3[0].scatter(r_samples, theta_samples, c='black', s=2)
for n in [1, 2, 3]:
    plot_cov_2d(np.array([r0, theta0]), np.array([[0.25, 0], [0, 0.0625]]), color='red', ax=ax3[0], nsigma=n, label = 'analytical')
    plot_cov_2d(mean_rtheta_sample, sigma_rtheta_sample, color='blue', ax=ax3[0], nsigma=n, label='sample')
ax3[0].legend()
ax3[0].set_xlabel('r')
ax3[0].set_ylabel('theta')
ax3[0].set_title('(r, theta): analytical vs. sample')

x_mean = r0*np.cos(theta0)
y_mean = r0*np.sin(theta0)
xy_mean = np.array([x_mean, y_mean])

xy_sigma = np.array(
    sig_xy_vals.subs([(sdev_r, 0.5), (sdev_theta, 0.25), (r, r0), (theta, theta0)]).tolist(),
    dtype=float,
)

xy_sample_cov = np.cov(np.vstack([x_samples, y_samples]))
ax3[1].scatter(x_samples, y_samples, c='black', s=2)
for n in [1, 2, 3]:
    plot_cov_2d(xy_mean, xy_sigma, color='red', ax=ax3[1], nsigma=n, label='analytical')
    plot_cov_2d(np.array([x_samples.mean(), y_samples.mean()]), xy_sample_cov, color='blue', ax=ax3[1], nsigma=n, label='sample')
ax3[1].legend()
ax3[1].set_xlabel('x')
ax3[1].set_ylabel('y')
ax3[1].set_title('(x, y): analytical vs. sample')
plt.tight_layout()
plt.savefig('UPQ3.png')
plt.show()


# The r, theta analytical and sample ellipses are basically the same, whereas the xy ellipses differ.
# Reason for difference between analytical and sample covariance in xy is likely due to the fact that when
# transforming from (r, theta) to (x, y), the covariance structure changes due to the nonlinear transformation (making it non-gaussian).

# Q 4:From a purely theoretical perspective, assuming that the underlying process is truly Gaussian, we
# expect 39.35% of all samples to lie within the 1-sigma contour, 86.47% of samples to lie within the 2-
# sigma contour, and 98.89% to lie within the 3-sigma contour. (These frequencies were computed using
# the cumulative chi-square distribution for two degrees-of-freedom, i.e. χ^2_2. In python, you can do this
# using the scipy.stats.chi2 object and the chi2.cdf function evaluated at chi2.cdf(1, 2), chi2.cdf(4,2), and  chi2cdf(9, 2) 
# respectively.)
# Modify your software to count the samples falling within each (analytical) ellipse for parts UP-Q1.i
# and UP-Q1.ii. The error of a particular sample x, measured in “units” of sigma, is known as the Mahalanobis 
# distance, and can be computed as sqrt((x − µ)⊤Σ−1(x − µ)).

from scipy.stats import chi2

for n in [1, 2, 3]:
    theory = chi2.cdf(n**2, df=2)
    diff = np.vstack([r_samples - r0, theta_samples - theta0])
    inv_sigma = np.linalg.inv(np.array([[sigma_r**2, 0], [0, sigma_theta**2]], dtype=float))
    mahal_dist = np.sqrt(np.einsum('in,ij,jn->n', diff, inv_sigma, diff))
    count = np.sum(mahal_dist <= n)
    print(f"Number of samples within {n}-sigma contour for (r, theta): {count} (theoretical: {theory * len(r_samples):.2f})")
    print(f"Percentage of samples within {n}-sigma contour for (r, theta): {count / len(r_samples) * 100:.2f}% (theoretical: {theory * 100:.2f}%)")

for n in [1, 2, 3]:
    theory = chi2.cdf(n**2, df=2)
    diff = np.vstack([x_samples - x_mean, y_samples - y_mean])
    inv_sigma = np.linalg.inv(xy_sigma)
    mahal_dist = np.sqrt(np.einsum('in,ij,jn->n', diff, inv_sigma, diff))
    count = np.sum(mahal_dist <= n)
    print(f"Number of samples within {n}-sigma contour for (x, y): {count} (theoretical: {theory * len(x_samples):.2f})")
    print(f"Percentage of samples within {n}-sigma contour for (x, y): {count / len(x_samples) * 100:.2f}% (theoretical: {theory * 100:.2f}%)")

# Q 5: If the point samples were truly distributed as a Gaussian (clearly the (x, y) are not), your counts would
#       match the theoretically predicted values. Try varying the noise parameters: under what conditions do
#       the counts come close to matching the theoretically predicted values? What consequences to a state
#       estimation algorithm could these sorts of errors have?

# answer to Q 5: The counts come close to matching the predicted values when the noise parameters are small and the 
# number of samples is large. The consequences to a state estimation algorithm could be significant, 
# as errors in the covariance structure can lead to incorrect estimates of the state and its uncertainty.

# Q 6: Suppose now that the (range, bearing) measurements are not independent but instead jointly correlated under the following three scenarios:
#      a) ρrθ = 0.1,
#      b) ρrθ = 0.5,
#      c) ρrθ = 0.9.
#      Repeat UP-Q1 and UP-Q3. (Hint, use the Cholesky decomposition).

rho_vals = [0.1, 0.5, 0.9]

fig6, ax6 = plt.subplots(2, 3, figsize=(15, 10))

for col, rho in enumerate(rho_vals):
    # Generate correlated samples
    cov_matrix = np.array([[sigma_r**2, rho * sigma_r * sigma_theta], [rho * sigma_r * sigma_theta, sigma_theta**2]])
    chol = np.linalg.cholesky(cov_matrix)
    samples = chol@ np.random.randn(2, N) + np.array([r0, theta0]).reshape(-1, 1)
    r_samples = samples[0, :]
    theta_samples = samples[1, :]
    x_samples = r_samples*np.cos(theta_samples)
    y_samples = r_samples*np.sin(theta_samples)
    
    J_xy = np.array([
        [np.cos(theta0), -r0 * np.sin(theta0)],
        [np.sin(theta0), r0 * np.cos(theta0)]
    ], dtype=float)
    sig_xy_analytical = J_xy @ cov_matrix @ J_xy.T
    mean_xy_analytical = J_xy @ np.array([r0, theta0], dtype=float)
    print(f'Correlation: {rho}, Analytical covariance: \n{sig_xy_analytical}')
    mean_xy_sample = np.array([np.mean(x_samples), np.mean(y_samples)], dtype=float)
    sig_xy_sample = np.cov(np.vstack([x_samples, y_samples]))

    # Plot the samples
    ax6[0, col].scatter(r_samples, theta_samples, s=2)
    ax6[0, col].set_title(f'Correlation: {rho}; Sensor frame (r, theta)')
    ax6[0, col].set_xlabel('r')
    ax6[0, col].set_ylabel('theta')

    # Calculate and plot the contours
    for n in [1, 2, 3]:
        plot_cov_2d(np.array([r_samples.mean(), theta_samples.mean()]), cov_matrix, color = 'r', nsigma=n, ax=ax6[0, col])
    
    ax6[1, col].scatter(x_samples, y_samples, s=2)
    ax6[1, col].set_title(f'Correlation: {rho}')
    ax6[1, col].set_xlabel('x')
    ax6[1, col].set_ylabel('y')

    for n in [1, 2, 3]:
        plot_cov_2d(mean_xy_sample, sig_xy_sample, color = 'r', nsigma=n, ax=ax6[1, col], label = f'Sample {n}-sigma')
        plot_cov_2d(mean_xy_analytical, sig_xy_analytical, color = 'b', nsigma=n, ax=ax6[1, col], label = f'Analytical {n}-sigma')
        
    ax6[1, col].legend()
plt.savefig('UPQ6.png')
plt.show()
