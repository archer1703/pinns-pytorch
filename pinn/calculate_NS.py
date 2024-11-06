import random
import torch.nn as nn
import numpy as np
import torch
from model_NS import MLP
from data_loader_NS import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (only if needed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用函数设置随机种子
set_random_seed(42)

# def net_NS(net, x, y, t, lambda_1, lambda_2):
#     X = torch.cat([x,y,t], dim=1)
#     psi, p = net(X)
#     psi = psi.requires_grad_(True)
#     p = p.requires_grad_(True)
#     u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
#     v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
#
#     u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#     u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#     u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#     u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
#     u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
#
#     v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
#     v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
#     v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
#     v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
#     v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
#
#     p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
#     p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
#
#     f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
#     f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
#
#     return u, v, p, f_u, f_v
#
#
# loss = nn.MSELoss()
#
# def l_f(f_u, f_v):
#     return loss(f_u, 0) + loss(f_v, 0)
#
#
xx, yy, tt, uu, vv, pp = load_data('cylinder_nektar_wake.mat')
xx = torch.from_numpy(np.array(xx, dtype=np.float32)).to(device).requires_grad_(True)
yy = torch.from_numpy(np.array(yy, dtype=np.float32)).to(device).requires_grad_(True)
tt = torch.from_numpy(np.array(tt, dtype=np.float32)).to(device).requires_grad_(True)
uu = torch.from_numpy(np.array(uu, dtype=np.float32)).to(device).requires_grad_(True)
vv = torch.from_numpy(np.array(vv, dtype=np.float32)).to(device).requires_grad_(True)
pp = torch.from_numpy(np.array(pp, dtype=np.float32)).to(device).requires_grad_(True)

net = MLP([20, 20, 20, 20, 20, 20, 20, 20], 3, 2, xx, yy, tt, uu, vv).to(device)
net.train(5000, xx, yy, tt, pp)
net.plot()
# net = MLP([100, 100, 100, 100], 3, 2).to(device)
# optim = torch.optim.Adam(list(net.parameters()) + [lambda_1, lambda_2], lr=0.001)
#
# loss1 = 0
# for i in range(5000):
#     optim.zero_grad()
#     u, v, p, f_u, f_v = net_NS(net, xx, yy, tt, lambda_1, lambda_2)
#     l = l_f(f_u, f_v) + loss(u, uu) + loss(v, vv)
#     loss1 += l
#     l.backward()
#     optim.step()
#
#     if i % 100 == 0:
#         print('iter:{},loss:{}'.format(i, loss1 / 100))
#         loss1 = 0