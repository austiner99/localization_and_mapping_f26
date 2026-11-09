# import numpy as np
# import helpers
# from fieldSettings import field
# from ekfUpdate import g, h, ekfUpdate

# np.set_printoptions(precision=5, suppress=True, linewidth=120)

# alphas = np.array([0.05, 0.001, 0.05, 0.01])**2
# beta   = np.radians(20)

# data = np.load('data/data.npz')['data']          # adjust the path if yours differs, e.g. '../data/data.npz'
# row  = data[0]                               # first time step
# u        = row[3:6]                          # [drot1, dtrans, drot2]
# markerId = int(row[0])
# z        = np.array([[row[1]]])              # noisy bearing, 1x1 like run.py

# d1, dt, d2 = u
# M = np.diag([alphas[0]*d1**2 + alphas[1]*dt**2,
#              alphas[2]*dt**2 + alphas[3]*(d1**2 + d2**2),
#              alphas[0]*d2**2 + alphas[1]*dt**2])
# Q = np.array([[beta**2]])

# mu    = np.array([180., 50., 0.])
# Sigma = np.diag([1., 1., 0.01])

# # ---- 1. Jacobian checks against finite differences ----
# eps = 1e-6
# Gnum = np.zeros((3, 3)); Vnum = np.zeros((3, 3)); Hnum = np.zeros(3)
# lm = (field['markerPosX'][markerId], field['markerPosY'][markerId])
# def wrap_diff(a, b):
#     out = a - b
#     out[2] = helpers.minimizedAngle(out[2])
#     return out
# for i in range(3):
#     e = np.zeros(3); e[i] = eps
#     Gnum[:, i] = wrap_diff(g(u, mu + e)[0], g(u, mu - e)[0]) / (2*eps)
#     Vnum[:, i] = wrap_diff(g(u + e, mu)[0], g(u - e, mu)[0]) / (2*eps)
#     Hnum[i]    = helpers.minimizedAngle(h(mu + e, lm)[0] - h(mu - e, lm)[0]) / (2*eps)
# _, G, V = g(u, mu)
# _, H    = h(mu, lm)
# print("max |G - Gnum| =", np.abs(G - Gnum).max())
# print("max |V - Vnum| =", np.abs(V - Vnum).max())
# print("max |H - Hnum| =", np.abs(H.ravel() - Hnum).max())   # all three should be ~1e-8 or smaller

# # ---- 2. One full ekfUpdate step ----
# mu_new, Sigma_new = ekfUpdate(mu, Sigma, u, M, z, Q, markerId)
# print("\nmu_new   :", mu_new, mu_new.shape)                  # expect (3,)
# print("Sigma_new:\n", Sigma_new, Sigma_new.shape)            # expect (3, 3)
# print("symmetric?        ", np.allclose(Sigma_new, Sigma_new.T))
# print("pos. definite?    ", np.all(np.linalg.eigvalsh(Sigma_new) > 0))
# print("pose before update:", mu, "-> after:", mu_new)

# import numpy as np
# from ukfUpdate import makeSigmaPoints

# rng = np.random.default_rng(0)
# A = rng.normal(size=(7, 7)); S = A @ A.T + np.eye(7); m = rng.normal(size=7)
# X, w = makeSigmaPoints(m, S)
# wm = w[:, 0]
# wc = w[:, 1]
# mean = wm @ X
# cov  = (X - mean).T @ np.diag(wc) @ (X - mean)
# print(X.shape, np.allclose(mean, m), np.allclose(cov, S), wm.sum())
# # expect: (15, 7) True True 1.0

# import numpy as np
# from ekfUpdate import g as g_ekf
# from ukfUpdate import ukfPredict

# alphas = np.array([0.05, 0.001, 0.05, 0.01])**2
# beta = np.radians(20)
# data = np.load('data/data.npz')['data']                      # adjust the path

# for idx in (0, 19):                                      # 19 is the first turning step
#     u = data[idx, 3:6]; d1, dt, d2 = u
#     M = np.diag([alphas[0]*d1**2 + alphas[1]*dt**2,
#                  alphas[2]*dt**2 + alphas[3]*(d1**2 + d2**2),
#                  alphas[0]*d2**2 + alphas[1]*dt**2])
#     Q = np.array([[beta**2]])
#     mu = np.array([180., 50., 0.5]); Sigma = np.diag([1., 1., 0.01])

#     mu_e, G, V = g_ekf(u, mu)
#     Sigma_e = G @ Sigma @ G.T + V @ M @ V.T
#     mu_u, Sigma_u = ukfPredict(mu, Sigma, u, M, Q)[:2]
#     print(idx, np.abs(mu_u - mu_e).max(), np.abs(Sigma_u - Sigma_e).max(),
#           Sigma_u.shape, np.allclose(Sigma_u, Sigma_u.T))

import numpy as np, helpers
from ekfUpdate import ekfUpdate
from ukfUpdate import ukfUpdate

alphas = np.array([0.05, 0.001, 0.05, 0.01])**2; beta = np.radians(20)
data = np.load('data/data.npz')['data']                  # adjust the path

def MQ(u):
    d1, dt, d2 = u
    return (np.diag([alphas[0]*d1**2 + alphas[1]*dt**2,
                     alphas[2]*dt**2 + alphas[3]*(d1**2 + d2**2),
                     alphas[0]*d2**2 + alphas[1]*dt**2]), np.array([[beta**2]]))

# one step from the same starting belief
row = data[0]; u = row[3:6]; M, Q = MQ(u)
mu = np.array([180., 50., 0.]); S = np.diag([1., 1., 0.01]); z = np.array([[row[1]]])
me, Se = ekfUpdate(mu, S, u, M, z, Q, int(row[0]))
mu_, Su = ukfUpdate(mu, S, u, M, z, Q, int(row[0]))
print("one step:", me, mu_, "| max|dmu| =", np.abs(me-mu_).max(), " max|dSigma| =", np.abs(Se-Su).max(),
      Su.shape, np.allclose(Su, Su.T), np.all(np.linalg.eigvalsh(Su) > 0))

# whole run on the data
ekf = (np.array([180., 50., 0.]), np.diag([1., 1., 0.01])); ukf = (ekf[0].copy(), ekf[1].copy())
gap = []; errs = {'ekf': [], 'ukf': []}; ins = {'ekf': [], 'ukf': []}
for r in data:
    u = r[3:6]; M, Q = MQ(u); z = np.array([[r[1]]]); m = int(r[0])
    ekf = ekfUpdate(*ekf, u, M, z, Q, m); ukf = ukfUpdate(*ukf, u, M, z, Q, m)
    gap.append(np.hypot(*(ekf[0][:2] - ukf[0][:2])))
    for name, (mu_, S_) in (('ekf', ekf), ('ukf', ukf)):
        e = mu_[:2] - r[6:8]
        errs[name].append(e @ e)
        ins[name].append(e @ np.linalg.solve(S_[:2, :2], e) <= 9)
for name in errs:
    print(name.upper(), "RMS position error: %.2f cm" % np.sqrt(np.mean(errs[name])),
          "| truth inside the 3-sigma ellipse: %.1f%%" % (100*np.mean(ins[name])))
print("max distance between EKF and UKF position estimates: %.2f cm" % max(gap))