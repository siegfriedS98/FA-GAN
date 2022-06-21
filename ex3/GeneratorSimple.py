import torch
import torch.nn as nn

USE_SIGMOID = True


class GeneratorSimple(nn.Module):
    """
        Generator model
    """
    def __init__(self, in_features, out_features, mode, sig=False):
        super().__init__()

        #torch.manual_seed(3)

        self.hidden = nn.Sequential(
            nn.Linear(in_features + mode, 100),
            nn.BatchNorm1d(100),
            nn.ReLU() if not sig else nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(100, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return x



