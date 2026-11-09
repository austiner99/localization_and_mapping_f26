import numpy as np
import matplotlib.pyplot as plt
from math import tau, pi

#functions for hw4
def rotxyz3(r,p,h):
    """
    rotation matrices that rotate a vector by roll, pitch, and heading

    :param r: a list of angles in radians specifying the rolls of each rotation matrix
    :param p: a list of angles in radians specifying the pitches of each rotation matrix
    :param h: a list of angles in radians specifying the headings of each rotation matrix
    :returns R: a [3x3xN] rotation matrix
    """
    r = r.reshape(1,1,-1)
    p = p.reshape(1,1,-1)
    h = h.reshape(1,1,-1)
    cr = np.cos(r)
    sr = np.sin(r)
    cp = np.cos(p)
    sp = np.sin(p)
    ch = np.cos(h)
    sh = np.sin(h)
    R = np.vstack((np.hstack((ch*cp,   (-sh*cr + ch*sp*sr),   ( sh*sr + ch*sp*cr))),
                   np.hstack((sh*cp,   ( ch*cr + sh*sp*sr),   (-ch*sr + sh*sp*cr))),
                   np.hstack((-sp,          cp*sr,                  cp*cr))))
    return R
def rotxyz1(rph):
    """
    rotation matrix that rotates a vector by roll, pitch, and heading

    :param rph: a 3 element vector of the form [roll,pitch,heading] with angles in radians
    :returns R: a [3x3] rotation matrix
    """
    return rotxyz3(rph[0],rph[1],rph[2])

def draw_ellipse(mu, Sigma, k2, npoints=100, color=None, show=True):
    """
    Draws the ellipse of the gaussian with mean mu, covariance Sigma, and size k2

    :param mu: a 2 item array specifying the x and y coordinates of the center of the ellipse
    :param Sigma: a 2x2 array specifying the covariance matrix of the gaussian the ellipse traces
    :param k2: confidence region k2, i.e. K2 = 1 means 1-sigma, K2 = 4 means 2-sigma, K2 = 9 means 3-sigma
    :param npoints: the number of points used to trace the ellipse
    :param color: the color the ellipse is drawn with
    :param show: if true, shows the plot immediately, otherwise waits for more stuff to be drawn on top
    """
    #plot the n sigma ellipse
    a = Sigma[0][0]
    b = Sigma[0][1]
    d = Sigma[1][1]
    lambda1 = (a+d)/2+np.sqrt(((a-d)/2)**2+b**2)
    lambda2 = (a+d)/2-np.sqrt(((a-d)/2)**2+b**2)
    theta = 0 if b==0 and a>=d else tau/4 if b==0 and a<d else np.atan2(lambda1-a,b)
    t = np.linspace(0, tau, npoints)
    plt.plot(mu[0]+np.sqrt(k2)*(np.sqrt(lambda1)*np.cos(theta)*np.cos(t)-np.sqrt(lambda2)*np.sin(theta)*np.sin(t)),
             mu[1]+np.sqrt(k2)*(np.sqrt(lambda1)*np.sin(theta)*np.cos(t)+np.sqrt(lambda2)*np.cos(theta)*np.sin(t)),
             color=color)

    plt.axis('equal')
    if show: plt.show()
    
def ssc_head2tail(X_ij,X_jk):
    """
    performs the head-to-tail frame composition X_ik = X_ij ⊕ X_jk

    :param X_ij: a 6 DOF vector of the form [x,y,z,roll,pitch,heading] with angles in radians
    :param X_jk: a 6 DOF vector of the form [x,y,z,roll,pitch,heading] with angles in radians
    :return: [X_ik,Jplus] where X_ik is the composition of the two inputs, and Jplus is a jacobian
    """
    # internally expand a 6x1 to match other arguments dim
    n=X_ij.shape[1]
    m=X_jk.shape[1]
    if n>m and m==1:
      X_jk = np.repeat(X_jk, n, axis=1)
    elif m>n and n==1:
      X_ij = np.repeat(X_ij, m, axis=1)
    elif m!=n:
      raise Exception('X_ij and X_jk must each be 6x1 or 6xN arrays.')
    n=X_ij.shape[1]
    
    # reshape each array from a 6xn to a 6x1xn
    X_ij = np.reshape(X_ij, (6,1,n))
    X_jk = np.reshape(X_jk, (6,1,n))

    # extract pose elements
    x_ij = X_ij[0,0,:].reshape(1,1,n)
    y_ij = X_ij[1,0,:].reshape(1,1,n)
    z_ij = X_ij[2,0,:].reshape(1,1,n)
    r_ij = X_ij[3,0,:].reshape(1,1,n)
    p_ij = X_ij[4,0,:].reshape(1,1,n)
    h_ij = X_ij[5,0,:].reshape(1,1,n)
    R_ij = rotxyz3(r_ij,p_ij,h_ij)
    t_ij = np.vstack((x_ij, y_ij, z_ij))
    
    x_jk = X_jk[0,0,:].reshape(1,1,n)
    y_jk = X_jk[1,0,:].reshape(1,1,n)
    z_jk = X_jk[2,0,:].reshape(1,1,n)
    r_jk = X_jk[3,0,:].reshape(1,1,n)
    p_jk = X_jk[4,0,:].reshape(1,1,n)
    h_jk = X_jk[5,0,:].reshape(1,1,n)
    R_jk = rotxyz3(r_jk,p_jk,h_jk)
    t_jk = np.vstack((x_jk, y_jk, z_jk))

    # translation component of X_ik
    t_ik = np.matmul(R_ij,t_jk, axes=[(0, 1), (0, 1), (0, 1)]) + t_ij
    x_ik = t_ik[0,0,:].reshape(1,1,n)
    y_ik = t_ik[1,0,:].reshape(1,1,n)
    z_ik = t_ik[2,0,:].reshape(1,1,n)

    # angle component of X_ik
    R_ik = np.matmul(R_ij,R_jk, axes=[(0, 1), (0, 1), (0, 1)])
    h_ik = np.atan2(R_ik[1,0,:], R_ik[0,0,:]).reshape(1,1,n)
    ch_ik = np.cos(h_ik)
    sh_ik = np.sin(h_ik)
    p_ik = np.atan2(-R_ik[2,0,:].reshape(1,1,n), R_ik[0,0,:].reshape(1,1,n)*ch_ik + R_ik[1,0,:].reshape(1,1,n)*sh_ik)
    r_ik = np.atan2(R_ik[0,2,:].reshape(1,1,n)*sh_ik - R_ik[1,2,:].reshape(1,1,n)*ch_ik, -R_ik[0,1,:].reshape(1,1,n)*sh_ik + R_ik[1,1,:].reshape(1,1,n)*ch_ik)

    # done with new state!
    X_ik = np.vstack((x_ik[0], y_ik[0], z_ik[0], r_ik[0], p_ik[0], h_ik[0]))

    # compute the Jacobian
    cr_ij = np.cos(r_ij)
    sr_ij = np.sin(r_ij)
    cp_ij = np.cos(p_ij)
    sp_ij = np.sin(p_ij)
    ch_ij = np.cos(h_ij)
    sh_ij = np.sin(h_ij)
    
    cr_ik = np.cos(r_ik)
    sr_ik = np.sin(r_ik)
    cp_ik = np.cos(p_ik)
    tp_ik = np.tan(p_ik)
    
    cp_jk = np.cos(p_jk)
    
    cr_ikmjk = np.cos(r_ik-r_jk)
    sr_ikmjk = np.sin(r_ik-r_jk)
    ch_ikmij = np.cos(h_ik-h_ij)
    sh_ikmij = np.sin(h_ik-h_ij)

    # M
    M = np.zeros((3,3,n))
    M[0,0,:] =  R_ij[0,2,:].reshape(1,1,n)*y_jk - R_ij[0,1,:].reshape(1,1,n)*z_jk
    M[0,1,:] =  (z_ik-z_ij)*ch_ij
    M[0,2,:] = -(y_ik-y_ij)
    M[1,0,:] =  R_ij[1,2,:].reshape(1,1,n)*y_jk - R_ij[1,1,:].reshape(1,1,n)*z_jk
    M[1,1,:] =  (z_ik-z_ij)*sh_ij
    M[1,2,:] =  x_ik-x_ij
    M[2,0,:] =  R_ij[2,2,:].reshape(1,1,n)*y_jk - R_ij[2,1,:]*z_jk
    M[2,1,:] = -x_jk*cp_ij - y_jk*sp_ij*sr_ij - z_jk*sp_ij*cr_ij
    M[2,2,:] =  0

    # K_1
    K_1 = np.zeros((3,3,n))
    K_1[0,0,:] =  cp_ij*ch_ikmij/cp_ik
    K_1[0,1,:] =  sh_ikmij/cp_ik
    K_1[0,2,:] =  0
    K_1[1,0,:] = -cp_ij*sh_ikmij
    K_1[1,1,:] =  ch_ikmij
    K_1[1,2,:] =  0
    K_1[2,0,:] =  ( R_jk[0,1,:].reshape(1,1,n)*sr_ik + R_jk[0,2,:].reshape(1,1,n)*cr_ik )/cp_ik
    K_1[2,1,:] =  tp_ik*sh_ikmij
    K_1[2,2,:] =  1
    
    # K_2
    K_2 = np.zeros((3,3,n))
    K_2[0,0,:] =  1
    K_2[0,1,:] =  tp_ik*sr_ikmjk
    K_2[0,2,:] =  ( R_ij[0,2,:].reshape(1,1,n)*ch_ik + R_ij[1,2,:].reshape(1,1,n)*sh_ik )/cp_ik
    K_2[1,0,:] =  0
    K_2[1,1,:] =  np.cos(r_ik-r_jk)
    K_2[1,2,:] = -cp_jk*sr_ikmjk
    K_2[2,0,:] =  0
    K_2[2,1,:] =  sr_ikmjk/cp_ik
    K_2[2,2,:] =  cp_jk*cr_ikmjk/cp_ik

    I = np.repeat(np.eye(3).reshape(3,3,1), n, axis=2)
    O = np.zeros((3,3,n))
    Jplus = np.vstack((np.hstack((I,  M,    R_ij,  O)),
                       np.hstack((O,  K_1,  O,     K_2))))

    return X_ik, Jplus
    
def ssc_inverse(X_ij):
    """
    performs the Smith, Self, & Cheeseman inverse ⊖X_ij

    :param X_ij: a 6 DOF vector of the form [x,y,z,roll,pitch,heading] with angles in radians
    :return: [X_ji,Jminus] where X_ji is the inverse of the input, and Jminus is d(X_ji)/d(X_ij) evaluated at X_ij.
    """
    n=X_ij.shape[1]
    X_ij = np.reshape(X_ij, (6,1,n))
    x_ij = X_ij[0,0,:].reshape(1,1,n)
    y_ij = X_ij[1,0,:].reshape(1,1,n)
    z_ij = X_ij[2,0,:].reshape(1,1,n)
    r_ij = X_ij[3,0,:].reshape(1,1,n)
    p_ij = X_ij[4,0,:].reshape(1,1,n)
    h_ij = X_ij[5,0,:].reshape(1,1,n)
    R_ij = rotxyz3(r_ij,p_ij,h_ij)

    # Invert

    x_ji = -( R_ij[0,0,:].reshape(1,1,n)*x_ij + R_ij[1,0,:].reshape(1,1,n)*y_ij + R_ij[2,0,:].reshape(1,1,n)*z_ij )
    y_ji = -( R_ij[0,1,:].reshape(1,1,n)*x_ij + R_ij[1,1,:].reshape(1,1,n)*y_ij + R_ij[2,1,:].reshape(1,1,n)*z_ij )
    z_ji = -( R_ij[0,2,:].reshape(1,1,n)*x_ij + R_ij[1,2,:].reshape(1,1,n)*y_ij + R_ij[2,2,:].reshape(1,1,n)*z_ij )
    h_ji = np.atan2(R_ij[0,1,:].reshape(1,1,n), R_ij[0,0,:].reshape(1,1,n))
    ch_ji = np.cos(h_ji)
    sh_ji = np.sin(h_ji)
    p_ji = np.atan2(-R_ij[0,2,:].reshape(1,1,n), R_ij[0,0,:].reshape(1,1,n)*ch_ji + R_ij[0,1,:].reshape(1,1,n)*sh_ji)
    r_ji = np.atan2(R_ij[2,0,:].reshape(1,1,n)*sh_ji - R_ij[2,1,:].reshape(1,1,n)*ch_ji, -R_ij[1,0,:].reshape(1,1,n)*sh_ji + R_ij[1,1,:].reshape(1,1,n)*ch_ji)

    X_ji = np.vstack((x_ji[0], y_ji[0], z_ji[0], r_ji[0], p_ji[0], h_ji[0]))

    # compute the Jacobian
    cr_ij = np.cos(r_ij)
    sr_ij = np.sin(r_ij)
    cp_ij = np.cos(p_ij)
    sp_ij = np.sin(p_ij)
    ch_ij = np.cos(h_ij)
    sh_ij = np.sin(h_ij)

    alpha = 1-R_ij[0,2,:].reshape(1,1,n)**2
    sqrt_alpha = np.sqrt(alpha)

    # N
    N = np.zeros((3,3,n))
    N[0,0,:] =  0
    N[0,1,:] = -R_ij[2,0,:].reshape(1,1,n)*x_ij*ch_ij - R_ij[2,0,:].reshape(1,1,n)*y_ij*sh_ij + z_ij*cp_ij
    N[0,2,:] =  R_ij[1,0,:].reshape(1,1,n)*x_ij - R_ij[0,0,:].reshape(1,1,n)*y_ij
    N[1,0,:] =  z_ji
    N[1,1,:] = -R_ij[2,1,:].reshape(1,1,n)*x_ij*ch_ij - R_ij[2,1,:].reshape(1,1,n)*y_ij*sh_ij + z_ij*sp_ij*sr_ij
    N[1,2,:] =  R_ij[1,1,:].reshape(1,1,n)*x_ij - R_ij[0,1,:].reshape(1,1,n)*y_ij
    N[2,0,:] = -y_ji
    N[2,1,:] = -R_ij[2,2,:].reshape(1,1,n)*x_ij*ch_ij - R_ij[2,2,:].reshape(1,1,n)*y_ij*sh_ij + z_ij*sp_ij*cr_ij
    N[2,2,:] =  R_ij[1,2,:].reshape(1,1,n)*x_ij - R_ij[0,2,:].reshape(1,1,n)*y_ij

    # Q
    Q = np.zeros((3,3,n))
    Q[0,0,:] = -R_ij[0,0,:].reshape(1,1,n)/alpha
    Q[0,1,:] = -R_ij[0,1,:].reshape(1,1,n)*cr_ij/alpha
    Q[0,2,:] =  R_ij[2,2,:].reshape(1,1,n)*R_ij[0,2,:].reshape(1,1,n)/alpha
    Q[1,0,:] =  R_ij[0,1,:].reshape(1,1,n)/sqrt_alpha
    Q[1,1,:] = -R_ij[2,2,:].reshape(1,1,n)*ch_ij/sqrt_alpha
    Q[1,2,:] =  R_ij[1,2,:].reshape(1,1,n)/sqrt_alpha
    Q[2,0,:] =  R_ij[0,0,:].reshape(1,1,n)*R_ij[0,2,:].reshape(1,1,n)/alpha
    Q[2,1,:] = -R_ij[1,2,:].reshape(1,1,n)*ch_ij/alpha
    Q[2,2,:] = -R_ij[2,2,:].reshape(1,1,n)/alpha

    Jminus = np.vstack((np.hstack((-np.transpose(R_ij, axes=(1,0,2)),  N)),
                        np.hstack(( np.zeros((3,3,n)),                 Q))))

    return X_ji, Jminus


def ssc_tail2tail(X_ij,X_ik):
    """
    performs the tail-to-tail frame composition X_jk = ⊖X_ij ⊕ X_ik

    :param X_ij: a 6 DOF vector of the form [x,y,z,roll,pitch,heading] with angles in radians
    :param X_jk: a 6 DOF vector of the form [x,y,z,roll,pitch,heading] with angles in radians
    :return: [X_ik,Jplus] where X_ik is the composition of the two inputs, and Jplus is a jacobian
    """
    # internally expand a 6x1 to match other arguments dim
    n=X_ij.shape[1]
    m=X_ik.shape[1]
    if n>m and m==1:
      X_ik = np.repeat(X_ik, n, axis=1)
    elif m>n and n==1:
      X_ij = np.repeat(X_ij, m, axis=1)
    elif m!=n:
      raise Exception('X_ij and X_ik must each be 6x1 or 6xN arrays.')
    n=X_ij.shape[1]

    [X_ji,Jminus] = ssc_inverse(X_ij)
    [X_jk,Jplus] = ssc_head2tail(X_ji,X_ik)

    J = np.hstack((np.matmul(Jplus[:,0:6,:],Jminus, axes=[(0, 1), (0, 1), (0, 1)]), Jplus[:,6:12,:]))

    return X_jk, J

