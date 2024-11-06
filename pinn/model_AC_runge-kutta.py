import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, size, input_size, output_size, init_weights=True):
        super(MLP, self).__init__()
        self.layer = []
        self.layer.append(nn.Linear(input_size, size[0]))
        self.layer.append(nn.ReLU())
        for i in range(1, len(size)):
            self.layer.append(nn.Linear(size[i - 1], size[i]))
            self.layer.append(nn.ReLU())
        self.layer.append(nn.Linear(size[-1], output_size))
        self.net = nn.Sequential(*self.layer)
        if init_weights:
            self.init_weights()

    def forward(self, x):
        output = self.net(x)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

