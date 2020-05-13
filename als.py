import numpy as np
from scipy.linalg import orth
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def gen_k_rank_mat(m, n, k):
    L = np.random.normal(0, 1, (m, k))
    R = np.random.normal(0, 1, (k, n))
    return np.dot(L, R)


def gen_mask(m, n, p):
    return np.random.binomial(1, p, (m, n))


def solve_als(X, Z, mask):
    n = Z.shape[1]
    k = X.shape[1]
    res = np.zeros((k, n))
    for i in range(n):
        X_ = X.copy()
        X_[1-mask[:, i].astype(bool)] = 0
        res[:, i] = np.linalg.lstsq(X_, Z[:, i], rcond=None)[0]
    return res


def altmincomplete(M, T, mu, p, k, mask):
    m, n = M.shape
    omegas = [np.zeros((m, n)) for _ in range(2*T+1)]
    for i in range(m):
        for j in range(n):
            omegas[np.random.randint(0, 2*T+1)][i, j] = mask[i, j]
    Mk = M.copy()
    Mk[1-omegas[0].astype(bool)] = 0
    Mk = Mk/p
    U, S, V = np.linalg.svd(Mk, full_matrices=False)
    clip = (2*mu*np.sqrt(k))/np.sqrt(max(m, n))
    U[U > clip] = 0
    Ut = orth(U)
    err_log = []
    for t in range(T):
        print('iter: ', t)
        Vt = solve_als(Ut, M, omegas[t+1].astype(bool))
        Ut = solve_als(Vt.T, M.T, omegas[t+T+1].astype(bool).T).T
        err = np.dot(Ut, Vt) - M
        err[1-mask] = 0
        err = np.linalg.norm(err)
        err_log.append(err)
        print(np.linalg.norm(err))
    return np.dot(Ut, Vt), err_log


m = 100
n = 100
p = 0.1
k = 2
T = 5
mu = 0.1

N = [100, 200, 500, 1000, 2000, 10000]
P = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
plt.figure(figsize=(15, 10))
plt.title("AltMinComplete convergence with p")
n = 100
mat = gen_k_rank_mat(m, n, k)
for p in P:
    M = mat.copy()
    mask = gen_mask(n, n, p)
    M[1-mask.astype(bool)] = 0
    xopt, err = altmincomplete(M, T, mu, p, k, mask)
    plt.plot(err, label="p: {}".format(p))
plt.legend()
plt.yscale('log')
plt.savefig("alswithp.png", dpi=300)
