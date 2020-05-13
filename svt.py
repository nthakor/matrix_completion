import numpy as np
from sklearn.utils.extmath import randomized_svd as rsvd


def svd_approax(M, k):
    U, S, V = rsvd(M, k)
    return np.linalg.multi_dot((U, np.diag(S), V))


def recon_error(mask, x, xhat):
    err = x-xhat
    err[1-mask.astype(int)] = 0
    return np.linalg.norm(err)


def test_error(xhat, test):
    err = []
    for u, m, r in test:
        err.append((xhat[u-1, m-1]-r)**2)
    return np.sqrt(np.mean(err))


def svt(M, mask, delta, eps, tau, l, kmax, test=None):
    Yk = 0
    rk = 0
    tr_err = []
    te_err = []
    for k in range(kmax):
        if k == 0:
            Xk = np.zeros_like(M)
        else:
            sk = rk + 1
            U, S, V = rsvd(Yk, sk)
            while min(S) >= tau:
                sk += l
                U, S, V = rsvd(Yk, sk)
            rk = sum(S >= tau)
            S = np.maximum(S-tau, 0)
            Xk = np.linalg.multi_dot((U, np.diag(S), V))
        Yk = mask*(Yk+delta*(M-Xk))
        tr_e = recon_error(mask, M, Xk)
        if test:
            te_e = test_error(Xk, test)
            te_err.append(te_e)
        tr_err.append(tr_e)
        if k % 100 == 0:
            print(k, tr_e)
        if tr_e <= eps:
            break
    return Xk, tr_err, te_err
