import torch
from model_schrodinger import MLP
import numpy as np
import torch.nn as nn
from data_loader import load_data
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample(n):
    x = (torch.rand(n, 1) * 10 - 5).to(device)
    y = (torch.rand(n, 1) * 1.6).to(device)
    return x.requires_grad_(True), y.requires_grad_(True)

loss = torch.nn.MSELoss()

def gradient(u, x, order=1):
    grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u).to(device), create_graph=True)[0]
    if order == 1:
        return grad
    else:
        return gradient(grad, x, order-1)

# 生成t=0时的边界数据
# u(0, x) = 2 sech(x)
def left(n):
    x = (torch.rand(n, 1) * 10 - 5).to(device)
    t = torch.zeros_like(x).to(device)
    cond = (4 / (torch.exp(x) + torch.exp(-x))).to(device)
    return x.requires_grad_(True), t.requires_grad_(True), cond

# u(t, 5) = u(t, -5)
def t_zero(n):
    t = (torch.rand(n, 1) * 1.6).to(device)
    x1 = (5 * torch.ones_like(t)).to(device)
    x2 = -5 * torch.ones_like(t).to(device)
    return t.requires_grad_(True), x1.requires_grad_(True), x2.requires_grad_(True)

# ux(t, 5) = ux(t, -5)
def t_one(n):
    t = (torch.rand(n, 1) * 1.6).to(device)
    x1 = (5 * torch.ones_like(t)).to(device)
    x2 = (-5 * torch.ones_like(t)).to(device)
    return t.requires_grad_(True), x1.requires_grad_(True), x2.requires_grad_(True)



# def l_interior(u):
#     x, y, cond = interior(1000)
#     uxy = u(torch.cat([x, y], dim=1))
#     return loss(uxy, cond)

def l_left(u):
    x, t, cond = left(50)
    u_real, u_image = u(torch.cat([t, x], dim=1))
    uxy = torch.sqrt(u_real ** 2 + u_image ** 2).reshape(-1, 1)
    return loss(uxy, cond)

def l_t_zero(u):
    t, x1, x2 = t_zero(50)
    u_real_1, u_image_1 = u(torch.cat([t, x1], dim=1))
    u_real_2, u_image_2 = u(torch.cat([t, x2], dim=1))
    uxy1 = torch.sqrt(u_real_1 ** 2 + u_image_1 ** 2)
    uxy2 = torch.sqrt(u_real_2 ** 2 + u_image_2 ** 2)
    return loss(uxy1, uxy2)

def l_t_one(u):
    t, x1, x2 = t_one(50)
    u_real_1, u_image_1 = u(torch.cat([t, x1], dim=1))
    u_real_2, u_image_2 = u(torch.cat([t, x2], dim=1))
    uxy1 = torch.sqrt(u_real_1 ** 2 + u_image_1 ** 2)
    uxy2 = torch.sqrt(u_real_2 ** 2 + u_image_2 ** 2)
    uxy1 = gradient(uxy1, x1)
    uxy2 = gradient(uxy2, x2)
    return loss(uxy1, uxy2)

def l_u(u):
    t, x = sample(20000)
    u_real, u_image = u(torch.cat([t, x], dim=1))
    uxt = u_real + 1j * u_image
    uu = u_real ** 2 + u_image ** 2
    L = 1j * (gradient(u_real, t).squeeze() + 1j * gradient(u_image, t).squeeze()) + \
        0.5 * (gradient(u_real, x, 2).squeeze() +1j * gradient(u_image, x, 2).squeeze()) +\
        uu * uxt
    h1 = L.real
    h2 = L.imag
    L = torch.sqrt(h1 ** 2 + h2 ** 2)
    return loss(L, torch.zeros_like(L))

def l_real(u, t, x, uu1, uu2):
    u_real, u_image = u(torch.cat([t, x], dim=1))
    return loss(u_real, uu1) + loss(u_image, uu2)


net = MLP([100, 100, 100, 100, 100], 2, 2).to(device)
optim = torch.optim.Adam(params=net.parameters(), lr=0.001)

t, uu, x = load_data('NLS.mat')
t = torch.from_numpy(np.array(t.T, dtype=np.float32).reshape(-1, 1)).to(device)
uu = torch.from_numpy(np.array(uu.T)).to(torch.complex64).to(device)
x = torch.from_numpy(np.array(x.T, dtype=np.float32).reshape(-1, 1)).to(device)
uu1 = torch.real(uu)
uu2 = torch.imag(uu)

loss1 = 0
start = time.time()
for i in range(5000):
    optim.zero_grad()
    l = l_left(net) / 50 + \
        l_t_zero(net) / 50 + \
        l_t_one(net) / 50 + \
        l_u(net) / 20000 + \
        l_real(net, t, x, uu1, uu2) / 100
    loss1 += l
    l.backward()
    optim.step()

    if i % 100 == 0:
        print('iter:{},loss:{}'.format(i + 100, loss1/100))
        loss1 = 0

end = time.time()
print('total time:{} s'.format(end - start))
# Inference

tc = torch.linspace(0, 1.6, 2000).to(device)
xc = torch.linspace(-5, 5, 2000).to(device)
tt, xx = torch.meshgrid(tc, xc, indexing='ij')
xx = xx.reshape(-1, 1)
tt = tt.reshape(-1, 1)
tx = torch.cat([tt, xx], dim=1).to(device)
u_real, u_image = net(tx)
u_pred = torch.sqrt(u_real ** 2 + u_image ** 2).detach().cpu().reshape(2000, 2000).T


plt.figure(figsize=(18, 6))
plt.contourf(tc.cpu(), xc.cpu(), u_pred, levels=200, cmap='YlGnBu')
# plt.imshow(u_pred, interpolation='nearest', cmap='YlGnBu', aspect='auto')
plt.colorbar()

plt.show()
