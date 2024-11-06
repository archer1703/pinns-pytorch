import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class MLP(nn.Module):
    def __init__(self, size, input_size, output_size, x, y, t, u, v, init_weights=True):
        super(MLP, self).__init__()
        self.layer = []
        self.layer.append(nn.Linear(input_size, size[0]))
        self.layer.append(nn.Tanh())
        for i in range(1, len(size)):
            self.layer.append(nn.Linear(size[i - 1], size[i]))
            self.layer.append(nn.Tanh())
        self.layer.append(nn.Linear(size[-1], output_size))
        self.net = nn.Sequential(*self.layer)

        # Use nn.Parameter for lambda1 and lambda2
        self.lambda1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        self.x = x
        self.y = y
        self.t = t
        self.u = u
        self.v = v

        # Adam optimizer
        self.optim = torch.optim.Adam(list(self.net.parameters()) + [self.lambda1, self.lambda2], lr=0.001)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        output = self.net(x)
        psi, p = output[:, 0], output[:, 1]
        return psi, p

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def net_NS(self, x, y, t):
        X = torch.cat([x, y, t], dim=1)
        psi, p = self.forward(X)
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f_u = u_t + self.lambda1 * (u * u_x + v * u_y) + p_x - self.lambda2 * (u_xx + u_yy)
        f_v = v_t + self.lambda1 * (u * v_x + v * v_y) + p_y - self.lambda2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss_function(self, x, y, t, p):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
        loss = torch.mean((self.u - u_pred) ** 2) + \
               torch.mean((self.v - v_pred) ** 2) + \
               torch.mean(f_u_pred ** 2) + \
               torch.mean(f_v_pred ** 2) + \
               torch.mean((p_pred - p) ** 2)
        return loss

    def train(self, epoch, x, y, t, p):
        writer = SummaryWriter()
        for it in range(epoch):
            def closure():
                self.optim.zero_grad()
                loss = self.loss_function(x, y, t, p)
                loss.backward()
                return loss

            self.optim.step(closure)
            loss_value = self.loss_function(x, y, t, p).item()
            print(
                f'Iteration {it}, Loss: {loss_value:.3e}, λ1: {self.lambda1.item():.3f}, λ2: {self.lambda2.item():.5f}, error1: {abs(self.lambda1.item() - 1.0):.4f}, error2: {abs(self.lambda2.item() - 0.01) / 0.01:.4f}')

            if it % 100 == 0 and it != 0:

                writer.add_scalar('mean_loss', loss_value, it + 100)
                writer.add_scalar('λ1', self.lambda1, it + 100)
                writer.add_scalar('λ2', self.lambda2, it + 100)
                writer.add_scalar('error1', abs(self.lambda1.item() - 1.0), it + 100)
                writer.add_scalar('error2', (self.lambda2.item() - 0.01)/0.01, it + 100)
                print(
                    f'Iteration {it + 100}, Loss: {loss_value:.3e}, λ1: {self.lambda1.item():.3f}, λ2: {self.lambda2.item():.5f}, error1: {abs(self.lambda1.item() - 1.0):.4f}, error2: {abs(self.lambda2.item() - 0.01)/0.01:.4f}')

    def plot(self):
        xc = torch.linspace(-5, 25, 250).to('cuda')
        yc = torch.linspace(-5, 5, 250).to('cuda')
        xx, yy = torch.meshgrid(xc, yc, indexing='ij')
        xx = xx.reshape(-1, 1).requires_grad_(True)
        yy = yy.reshape(-1, 1).requires_grad_(True)
        tt = torch.zeros_like(xx).to('cuda').requires_grad_(True)
        u, v, p, f_u, f_v = self.net_NS(xx, yy, tt)
        u = u.requires_grad_(True)
        v = v.requires_grad_(True)
        du_dy = torch.autograd.grad(u, yy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dv_dx = torch.autograd.grad(v, xx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        vorticity = (dv_dx - du_dy).detach().cpu().reshape(250, 250).T


        plt.figure(figsize=(10, 4))
        contour = plt.contourf(xc.cpu(), yc.cpu(), vorticity, levels=100, cmap='bwr')
        plt.colorbar(contour, label="Vorticity")

        plt.show()


if __name__ == '__main__':
    # Dummy data for testing
    x = torch.rand(100, 1)
    y = torch.rand(100, 1)
    t = torch.rand(100, 1)
    u = torch.rand(100, 1)
    v = torch.rand(100, 1)

    net = MLP([100, 100, 100], 3, 2, x, y, t, u, v)
    print(list(net.parameters()))
