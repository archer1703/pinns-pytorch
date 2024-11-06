import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, size, input_size, output_size, learning_rate, u1, u2, init_weights=True):
        super(MLP, self).__init__()
        self.layer = []
        self.layer.append(nn.Linear(input_size, size[0]))
        self.layer.append(nn.ReLU())
        for i in range(1, len(size)):
            self.layer.append(nn.Linear(size[i - 1], size[i]))
            self.layer.append(nn.ReLU())
        self.layer.append(nn.Linear(size[-1], output_size))
        self.net = nn.Sequential(*self.layer)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.u1 = u1.to(self.device)
        self.u2 = u2.to(self.device)
        if init_weights:
            self.init_weights()

        # Use nn.Parameter for lambda1 and lambda2
        self.lambda1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.optim = torch.optim.LBFGS(list(self.net.parameters()) +
                                      [self.lambda1, self.lambda2], lr=learning_rate)

        self.mse_loss = nn.MSELoss()
    def forward(self, x):
        # avg = torch.mean(x, dim=1)
        # std = torch.std(x, dim=1)
        # x = (x - avg) / std

        output = self.net(x)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def sample(self, n):
        x = (torch.rand(n, 1) * 2 - 1).to(self.device)
        y = torch.rand(n, 1).to(self.device)
        return x.requires_grad_(True), y.requires_grad_(True)

    def KdV(self, t, x, dt, weight):
        u = self.net(torch.cat([t, x], dim=1))
        u_q = u[:, :-1].to(self.device)
        ux = torch.autograd.grad(u_q, x, grad_outputs=torch.ones_like(u_q), create_graph=True)[0]
        ux = ux.expand_as(u_q)
        uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(u_q), create_graph=True)[0]
        uxx = uxx.expand_as(u_q)
        uxxx = torch.autograd.grad(uxx, x, grad_outputs=torch.ones_like(u_q), create_graph=True)[0]
        uxxx = uxxx.expand_as(u_q)
        f = - self.lambda1 * u_q * ux - self.lambda2 * uxxx
        u0 = u - dt * torch.matmul(f, weight.T)

        return u0

    def l_u(self, t, x):
        u = torch.reshape(self.net(torch.cat([t, x], dim=1))[:, -1], (-1, 1))
        ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # ut = ut.expand_as(u)
        ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # ux = ux.expand_as(u)
        uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
        # uxxx = uxxx.expand_as(u)
        uxxx = torch.autograd.grad(uxx, x, grad_outputs=torch.ones_like(uxx), create_graph=True)[0]
        # uxxx = uxxx.expand_as(u)
        f = ut + self.lambda1 * u * ux + self.lambda2 * uxxx
        return f

    def l_0(self, n):
        x = (torch.rand(n, 1) * 2 - 1).to(self.device)
        t = torch.zeros_like(x).to(self.device)
        cond = torch.cos(np.pi * x)
        return x.requires_grad_(True), t.requires_grad_(True), cond



    def L(self, t, x, u, t1, x1, t2, x2, dt, weight):  # t = 0.2, t = 0.6
        xx, tt = self.sample(20000)
        x0, t0, cond = self.l_0(50)
        l = self.mse_loss(self.u1, torch.reshape(self.KdV(t1, x1, dt, weight)[:, -1], self.u1.shape)) / 199 + \
            self.mse_loss(self.u2, torch.reshape(self.KdV(t2, x2, -dt, weight)[:, -1], self.u2.shape)) / 201 + \
            self.mse_loss(torch.reshape(self.net(torch.cat([t0, x0], dim=1))[:, -1], (-1, 1)), cond) / 50 + \
            self.mse_loss(self.l_u(tt, xx), torch.zeros_like(self.l_u(tt, xx))) / 20000 + \
            self.mse_loss(torch.reshape(self.net(torch.cat([t, x], dim=1))[:, -1], (-1, 1)), u) / 150
        return l

    def train(self, epoch, t, x, u, t1, x1, t2, x2, dt, weight):
        def closure():
            self.optim.zero_grad()
            loss = self.L(t, x, u, t1, x1, t2, x2, dt, weight)
            loss.backward()
            return loss

        for it in range(epoch):
            loss_value = self.optim.step(closure).item()

            if it % 100 == 0:
                print(
                    f'Iteration {it + 100}, Loss: {loss_value:.3e}, λ1: {self.lambda1.item():.3f}, λ2: {self.lambda2.item():.5f}, error1: {abs(self.lambda1.item() - 1.0):.4f}, error2: {abs(self.lambda2.item() - 0.0025)/0.0025:.4f}')

    def plot(self):
        xc = torch.linspace(-5, 25, 250).to("cuda")
        yc = torch.linspace(-5, 5, 250).to("cuda")
        xx, yy = torch.meshgrid(xc, yc, indexing='ij')
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        u_pred = self.net(torch.cat([xx, yy], dim=1))[:, -1].detach().cpu().reshape(250, 250).T

        plt.figure(figsize=(18, 6))
        # plt.contourf(tc.cpu(), xc.cpu(), u_pred, levels=200, cmap='Seismic')
        plt.imshow(u_pred, interpolation='nearest', cmap='rainbow', extent=[0, 1, -1, 1], origin='lower', aspect='auto')
        plt.colorbar()

        plt.show()





