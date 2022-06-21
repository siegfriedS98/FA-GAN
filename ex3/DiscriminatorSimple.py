import torch
import torch.nn as nn
import torch.nn.functional as F
from biotorch.layers.fa import Conv2d, Linear


class DiscriminatorSimple(nn.Module):
    '''
        Discriminator model
    '''
    def __init__(self, in_features, out_features, mode, sig=False):
        super(DiscriminatorSimple, self).__init__()

        #torch.manual_seed(3)

        self.hidden = nn.Sequential(
            nn.Linear(in_features + mode, 100),
            nn.ReLU() if not sig else nn.Sigmoid(),
            nn.Dropout(0.5),
        )

        self.out = nn.Sequential(
            torch.nn.Linear(100, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return x

