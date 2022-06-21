import torch
import torch.nn as nn

USE_SIGMOID = True


class Generator(nn.Module):
    """
        Generator model
    """
    def __init__(self, in_features, out_features, mode, sig=False):
        super().__init__()

        #torch.manual_seed(3)

        self.hidden_0 = nn.Sequential(
            nn.Linear(in_features + mode, 256),
            nn.BatchNorm1d(256),
            nn.ReLU() if sig == False else nn.Sigmoid()
        )
        self.hidden_1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU() if sig == False else nn.Sigmoid()
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU() if sig == False else nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(1024, out_features),
            nn.Sigmoid() if USE_SIGMOID else nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x



