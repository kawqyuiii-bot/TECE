import torch, torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(0)
        self.c = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x, t):
        return 0.1 * self.c(x) + 0.5 * x * (t.view(-1, 1, 1, 1) / 1000)
