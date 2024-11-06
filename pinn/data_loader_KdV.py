import scipy.io as sio
import numpy as np
import os

import torch

def load_data(file_name):
    data = sio.loadmat(file_name)
    tt = data['tt'].flatten()
    uu = data['uu']
    x = data['x'].flatten()
    row = uu.shape[0]
    row = np.random.choice(row, 150, replace=True)
    column = uu.shape[1]
    column = np.random.choice(column, 150, replace=True)

    return tt[column], uu[row, column], x[row]

def load_data0(file_name):
    data = sio.loadmat(file_name)
    tt = data['tt'].flatten()
    uu = data['uu']
    x = data['x'].flatten()
    row = uu.shape[0]
    row = np.random.choice(row, 199, replace=True)
    column = 40

    return 0.2 * np.ones_like(x[row]), uu[row, column], x[row]

def load_data1(file_name):
    data = sio.loadmat(file_name)
    tt = data['tt'].flatten()
    uu = data['uu']
    x = data['x'].flatten()
    row = uu.shape[0]
    row = np.random.choice(row, 201, replace=True)
    column = 160

    return 0.8 * np.ones_like(x[row]), uu[row, column], x[row]

def load_weights(root, q):
    file_name = "Butcher_IRK{}.txt".format(q)
    dataroot = os.path.join(root, file_name)
    temp = np.float32(np.loadtxt(dataroot))
    IRK_weights = torch.tensor(np.reshape(temp[:q**2+q], (q + 1, q)), dtype=torch.float32)
    IRK_times = torch.tensor(temp[q**2 + q:], dtype=torch.float32)
    return IRK_weights, IRK_times



