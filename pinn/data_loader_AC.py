import scipy.io as sio
import numpy as np
import os

import torch


# 随机选150个数据点
def load_data0(file_name):
    data = sio.loadmat(file_name)
    tt = data['tt'].flatten()
    uu = data['uu']
    x = data['x'].flatten()
    row = uu.shape[0]
    row = np.random.choice(row, 150, replace=True)
    column = uu.shape[1]
    column = np.random.choice(column, 150, replace=True)

    return tt[column], uu[row, column], x[row]

# 随机选150个数据点
def load_data(file_name):
    data = sio.loadmat(file_name)
    tt = data['tt'].flatten()
    uu = data['uu']
    x = data['x'].flatten()
    row = uu.shape[0]
    row = np.random.choice(row, 200, replace=True)
    column = 20

    return 0.1 * np.ones_like(x[row]), uu[row, column], x[row]

def load_weights(root, q):
    file_name = "Butcher_IRK{}.txt".format(q)
    dataroot = os.path.join(root, file_name)
    temp = np.float32(np.loadtxt(dataroot))
    IRK_weights = torch.tensor(np.reshape(temp[:q**2+q], (q + 1, q)), dtype=torch.float32)
    IRK_times = torch.tensor(temp[q**2 + q:], dtype=torch.float32)
    return IRK_weights, IRK_times

if __name__ == '__main__':
    tt, uu, x = load_data('AC.mat')

    tt = np.array(tt.T, dtype=np.float32)
    uu = np.array(uu.T)
    x = np.array(x.T, dtype=np.float32)
    weight, time = load_weights('E:\计算物理\pinn\weights', 100)
    print(weight.size())
    print(time)

    print(tt)
    print(uu)
    print(x)
    print(tt.shape, uu.shape, x.shape)