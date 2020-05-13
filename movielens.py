import numpy as np
from sklearn.utils.extmath import randomized_svd as rsvd

from algo import svd_approax, svt_solve
from altmin import altmincomplete

data = np.loadtxt('ml-100k/u.data')
train = np.loadtxt('ml-100k/ua.base')
test = np.loadtxt('ml-100k/ua.test')


def process_data(data):
    data = data[:, :3].astype(np.int)
    users = data[:, 0]
    movies = data[:, 1]
    rating = data[:, 2]
    mat = np.zeros((max(users), max(movies))).astype(np.int)
    for u, m, r in data:
        mat[u-1, m-1] = r
    return mat


tmat = process_data(train)
test = test[:, :3].astype(np.int)
mask = np.ones_like(tmat).astype(np.int)
mask[tmat == 0] = 0


def train_error(mask, x, xhat):
    return np.linalg.norm(mask*(x-xhat))/np.linalg.norm(mask*x)


def test_error(xhat, test):
    err = []
    for u, m, r in test:
        err.append((xhat[u-1, m-1]-r)**2)
    return np.sqrt(np.mean(err))

