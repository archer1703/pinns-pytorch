import torch
from model_AC import MLP
import numpy as np
import torch.nn as nn
from data_loader import load_data
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample(n):
    x = (torch.rand(n, 1) * 2 - 1).to(device)
    y = torch.rand(n, 1).to(device)
    return x.requires_grad_(True), y.requires_grad_(True)

loss = torch.nn.MSELoss()

def gradient(u, x, order=1):
    grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u).to(device), create_graph=True)[0]
    if order == 1:
        return grad
    else:
        return gradient(grad, x, order-1)

def left(n):
    # x = torch.linspace(-5, 5, n).reshape(-1, 1).to(device)
    x = (torch.rand(n, 1) * 2 - 1).to(device)
    t = torch.zeros_like(x).to(device)
    cond = (x ** 2 * torch.cos(np.pi * x)).to(device)
    return x.requires_grad_(True), t.requires_grad_(True), cond

def t_zero(n):
    # t = torch.linspace(0, 1.6, n).reshape(-1, 1).to(device)
    t = torch.rand(n, 1).to(device)
    x1 = torch.ones_like(t).to(device)
    x2 = - torch.ones_like(t).to(device)
    return t.requires_grad_(True), x1.requires_grad_(True), x2.requires_grad_(True)

def t_one(n):
    # t = torch.linspace(0, 1.6, n).reshape(-1, 1).to(device)
    t = torch.rand(n, 1).to(device)
    x1 = torch.ones_like(t).to(device)
    x2 = - torch.ones_like(t).to(device)
    return t.requires_grad_(True), x1.requires_grad_(True), x2.requires_grad_(True)

def l_left(u):
    x, t, cond = left(50)
    uxy = u(torch.cat([t, x], dim=1)).reshape(-1, 1)
    return loss(uxy, cond)

def l_t_zero(u):
    t, x1, x2 = t_zero(50)
    uxy1 = u(torch.cat([t, x1], dim=1))
    uxy2 = u(torch.cat([t, x2], dim=1))
    return loss(uxy1, uxy2)

def l_t_one(u):
    t, x1, x2 = t_one(50)
    uxy1 = u(torch.cat([t, x1], dim=1))
    uxy2 = u(torch.cat([t, x2], dim=1))
    uxy1 = gradient(uxy1, x1)
    uxy2 = gradient(uxy2, x2)
    return loss(uxy1, uxy2)

def l_u(u):
    t, x = sample(20000)
    uxy = u(torch.cat([t, x], dim=1))
    ut = gradient(uxy, t)
    uxx = gradient(uxy, x, 2)
    f = ut - 0.0001 * uxx + 5 * uxy ** 3 - 5 * uxy
    return loss(f, torch.zeros_like(f))

def l_real(u, t, x, uu):
    uxy = u(torch.cat([t, x], dim=1))
    return loss(uxy, uu)

net = MLP([100, 100, 100, 100, 100], 2, 1).to(device)

optim = torch.optim.Adam(params=net.parameters(), lr=0.001)

t, uu, x = load_data('AC.mat')
t = torch.from_numpy(np.array(t.T, dtype=np.float32).reshape(-1, 1)).to(device)
uu = torch.from_numpy(np.array(uu.T, dtype=np.float32).reshape(-1, 1)).to(device)
x = torch.from_numpy(np.array(x.T, dtype=np.float32).reshape(-1, 1)).to(device)


loss1 = 0
for i in range(5000):
    optim.zero_grad()
    l = l_left(net) / 50 + l_t_zero(net) / 50 + l_t_one(net) / 50 + l_u(net) / 20000 + l_real(net, t, x, uu) / 150
    loss1 += l
    l.backward()
    optim.step()

    if i % 100 == 0:
        print('iter:{},loss:{}'.format(i, loss1/100))
        loss1 = 0

# Inference

tc = torch.linspace(0, 1, 2000).to(device)
xc = torch.linspace(-1, 1, 2000).to(device)
tt, xx = torch.meshgrid(tc, xc, indexing='ij')
xx = xx.reshape(-1, 1)
tt = tt.reshape(-1, 1)
tx = torch.cat([tt, xx], dim=1).to(device)
u_pred = net(tx).detach().cpu().reshape(2000, 2000).T



plt.figure(figsize=(18, 6))
# plt.contourf(tc.cpu(), xc.cpu(), u_pred, levels=200, cmap='seismic')
plt.imshow(u_pred, extent = [0, 1, -1, 1], cmap='seismic', aspect='auto')
plt.colorbar()

plt.show()


