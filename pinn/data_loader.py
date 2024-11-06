import scipy.io as sio
import numpy as np

# 随机选150个数据点
def load_data(file_name):
    data = sio.loadmat(file_name)
    tt = data['tt'].flatten()
    uu = data['uu']
    x = data['x'].flatten()
    row = uu.shape[0]
    row = np.random.choice(row, 100, replace=True)
    column = uu.shape[1]
    column = np.random.choice(column, 100, replace=True)
    
    return tt[column], uu[row,column], x[row]

if __name__ == '__main__':
    tt, uu, x = load_data('NLS.mat')

    tt = np.array(tt.T, dtype=np.float32)
    uu = np.array(uu.T)
    x = np.array(x.T, dtype=np.float32)


    print(tt.shape, uu.shape, x.shape)