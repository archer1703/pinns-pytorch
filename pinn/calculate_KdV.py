import numpy as np
import torch

from model_KdV import MLP
from data_loader_KdV import load_data, load_data0, load_data1, load_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t, u, x = load_data("KdV.mat")
t0, u0, x0 = load_data0("KdV.mat")
t1, u1, x1 = load_data1("KdV.mat")

t = torch.from_numpy(np.array(t, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)
x = torch.from_numpy(np.array(x, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)
u = torch.from_numpy(np.array(u, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)
t0 = torch.from_numpy(np.array(t0, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)
x0 = torch.from_numpy(np.array(x0, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)
u0 = torch.from_numpy(np.array(u0, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)
t1 = torch.from_numpy(np.array(t1, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)
x1 = torch.from_numpy(np.array(x1, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)
u1 = torch.from_numpy(np.array(u1, dtype=np.float32)).reshape(-1, 1).to(device).requires_grad_(True)



q = 100
dt = 0.6
weight, time = load_weights('E:\computational physics\pinn\weights', q)
weight = weight.to(device)
time = time.to(device)
net = MLP([50, 50, 50, 50], 2, q+1, 0.001, u0, u1).to(device)
net.train(5000, t, x, u, t0, x0, t1, x1, dt, weight)
net.plot()
