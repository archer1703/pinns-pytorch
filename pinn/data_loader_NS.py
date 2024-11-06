import scipy.io as sio
import numpy as np

N_train = 5000
def load_data(file_name, isTrain=True):
    data = sio.loadmat(file_name)

    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    x = np.tile(X_star[:, 0:1], (1, T)).reshape((N * T, 1))  # N x T
    y = np.tile(X_star[:, 1:2], (1, T)).reshape((N * T, 1))   # N x T
    t = np.tile(t_star, (1, N)).T.reshape((N * T, 1))   # N x T

    u = U_star[:, 0, :].reshape((N * T, 1))   # N x T
    v = U_star[:, 1, :].reshape((N * T, 1))   # N x T
    p = P_star.reshape((N * T, 1))   # N x T

    if isTrain:
        idx = np.random.choice(N * T, N_train, replace=True)
        x = x[idx, :]
        y = y[idx, :]
        t = t[idx, :]
        u = u[idx, :]
        v = v[idx, :]
        p = p[idx, :]

    return x, y, t, u, v, p



